# SFT Warm-up Pipeline for TTT-RTL

Collects diverse Verilog solutions from strong models (Claude/GPT/Gemini) via OpenRouter, filters through the existing iverilog+yosys reward pipeline, and trains using verl's built-in FSDP SFT trainer. The resulting checkpoint gives the model high syntax/functional pass rates so RL can focus on PPA optimization.

## Quick Start

```bash
export OPENROUTER_API_KEY="sk-or-..."
bash recipe/ttt_rtl/sft/run_pipeline.sh
```

## Pipeline Steps

### 1. Collect Responses

Queries OpenRouter models for Verilog solutions to all RTLLM problems. Uses the same prompt format as the TTT-RTL RL loop (reference implementation + design spec + optimization instruction).

```bash
# All problems, all models
python -m recipe.ttt_rtl.sft.collect_responses

# Single problem, single model (for testing)
python -m recipe.ttt_rtl.sft.collect_responses --problems adder_16bit --models claude-sonnet
```

Proxy options:

```bash
# Option 1: standard environment variables
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
python -m recipe.ttt_rtl.sft.collect_responses --problems adder_16bit --models claude-sonnet

# Option 2: explicit flag
python -m recipe.ttt_rtl.sft.collect_responses \
  --problems adder_16bit \
  --models claude-sonnet \
  --proxy http://127.0.0.1:7890

# Option 3: dedicated environment variable
export OPENROUTER_PROXY=http://127.0.0.1:7890
python -m recipe.ttt_rtl.sft.collect_responses --problems adder_16bit --models claude-sonnet
```

- Async HTTP with per-model rate limiting
- Resumable: skips already-collected `(problem_id, model_name, sample_idx)` tuples
- Saves to `recipe/ttt_rtl/sft/data/raw/{model_name}.jsonl`

### 2. Filter Responses

Runs each response through iverilog compile → simulation → Yosys synthesis, deduplicates within each problem, and keeps only functionally correct samples.

```bash
python -m recipe.ttt_rtl.sft.filter_responses
```

- Parallel evaluation via `multiprocessing.Pool`
- Prints per-model and per-problem statistics (syntax%, func%, mean PPA)
- Outputs `recipe/ttt_rtl/sft/data/filtered.jsonl`

### 3. Format for SFT

Converts filtered records to parquet with the `messages` format expected by verl's `MultiTurnSFTDataset`. Performs a stratified train/val split by problem ID.

```bash
python -m recipe.ttt_rtl.sft.format_sft_data
```

- Outputs `recipe/ttt_rtl/sft/data/train.parquet` and `val.parquet`
- Each row has a `messages` column: `[{role: system, ...}, {role: user, ...}, {role: assistant, ...}]`

### 3b. Merge with CoDeV-R1 Data (Optional)

Merges the 87K CoDeV-R1 distillation dataset ([zhuyaoyu/CodeV-R1-dataset](https://huggingface.co/datasets/zhuyaoyu/CodeV-R1-dataset)) with the locally collected RTLLM data. Automatically converts CoDeV-R1's `<think>`/`<answer>` tags to `<thinking>` format.

```bash
# Download CoDeV-R1 SFT data (login node, requires internet)
huggingface-cli download zhuyaoyu/CodeV-R1-dataset codev_r1_sft.jsonl \
  --local-dir recipe/ttt_rtl/sft/data/

# Merge all 87K CoDeV-R1 + 451 RTLLM records
python -m recipe.ttt_rtl.sft.merge_codev_data

# Or subsample CoDeV-R1 to balance with RTLLM data
python -m recipe.ttt_rtl.sft.merge_codev_data --codev-sample 5000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--codev-sample N` | all 87K | Randomly sample N CoDeV-R1 records for balancing |
| `--val-ratio` | 0.05 | Fraction held out for validation (monitors overfitting) |
| `--seed` | 42 | Random seed for reproducibility |

Existing `train.parquet` and `val.parquet` are backed up to `.parquet.bak` before overwriting.

### 4. Train

Launches verl's `FSDPSFTTrainer` via `torchrun` with multi-turn SFT configuration.

```bash
# Full training (8 GPUs)
python -m recipe.ttt_rtl.sft.train_sft

# Single-GPU smoke test
python -m recipe.ttt_rtl.sft.train_sft --n-gpus 1

# Preview the command without running
python -m recipe.ttt_rtl.sft.train_sft --dry-run
```

The checkpoint is saved to `checkpoints/ttt-rtl-sft/global_step_N/` in HF format, directly usable as `MODEL_PATH` for `run_ttt_rtl.sh`.

## Configuration

All settings are in `config.yaml`. Key parameters:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `models` | `id` | 3 models | OpenRouter model IDs |
| `collection` | `num_samples` | 10 | Samples per (problem, model) |
| `collection` | `temperature` | 0.8 | Sampling temperature |
| `collection` | `concurrency` | 8 | Max concurrent requests per model |
| `filtering` | `require_functional` | true | Only keep functionally correct samples |
| `filtering` | `dedup` | true | Deduplicate within each problem |
| `formatting` | `val_split` | 0.1 | Fraction of problems for validation |
| `training` | `lr` | 1e-5 | Learning rate |
| `training` | `total_epochs` | 3 | Training epochs |
| `training` | `max_length` | 8192 | Max sequence length |
| `training` | `train_batch_size` | 64 | Global batch size |

## Pipeline Flags

The orchestration script supports flags to skip stages:

```bash
# Skip collection (use existing raw data)
bash recipe/ttt_rtl/sft/run_pipeline.sh --skip-collect

# Skip training (just collect + filter + format)
bash recipe/ttt_rtl/sft/run_pipeline.sh --skip-train

# Custom config and GPU count
N_GPUS=4 bash recipe/ttt_rtl/sft/run_pipeline.sh --config my_config.yaml
```

## Using the SFT Checkpoint for RL

After training, point the RL pipeline to the SFT checkpoint:

```bash
export MODEL_PATH=checkpoints/ttt-rtl-sft/global_step_N/
bash recipe/ttt_rtl/run_ttt_rtl.sh
```

## Requirements

- `aiohttp` and `pyyaml` for collection (`pip install aiohttp pyyaml`)
- `iverilog` and `yosys` on PATH for filtering
- `OPENROUTER_API_KEY` environment variable for collection
