# TTT-Discover for Verilog RTL Generation

This recipe implements TTT-Discover (Test-Time Training with Discover algorithm) for Verilog RTL code generation and PPA (Power-Performance-Area) optimization using the RTLLM benchmark.

The core idea: start from a reference implementation, then iteratively refine it via PUCT-guided tree search, using a hierarchical reward (syntax → functional → A·D·P) to drive the model toward designs with better Area × Delay × Power. ADP (Area × Delay only) is supported as a fallback when power reporting is disabled.

## Quick Start

### Prerequisites

```bash
# Install verl with vLLM
pip install -e .[test,vllm]

# Install Icarus Verilog (for simulation)
sudo apt-get install iverilog

# Install Yosys (for synthesis / PPA reward)
sudo apt-get install yosys

# Install OpenSTA (for accurate static timing analysis — optional but recommended)
# Path to the OpenSTA `sta` binary
# Or build from source: https://github.com/The-OpenROAD-Project/OpenSTA

# Install SwanLab (for experiment tracking)
pip install swanlab
```

### Single-Problem Training (recommended starting point)

```bash
# Train on the multi_pipe_4bit problem only (default)
export MODEL_PATH=~/models/Qwen2.5-7B-Instruct
bash recipe/ttt_rtl/run_multi_pipe_4bit.sh

# Train on any other single problem — override RTL_PROBLEMS
RTL_PROBLEMS=adder_16bit bash recipe/ttt_rtl/run_multi_pipe_4bit.sh
RTL_PROBLEMS=counter_12 bash recipe/ttt_rtl/run_multi_pipe_4bit.sh

# Override experiment name to avoid state-pool collision
EXP_NAME=adder_exp1 RTL_PROBLEMS=adder_16bit bash recipe/ttt_rtl/run_multi_pipe_4bit.sh
```

### FloatPoint Problems (fp_add, etc.)

Large multi-module designs (e.g. `fp_add` — 4021 lines, IEEE 754 dual-path pipelined adder) need a dedicated config with longer context windows and TP=2 rollout to fit on 8×A800 80 GB:

```bash
CONFIG_NAME=ttt_rtl_fp_trainer \
EXP_NAME=fp_add_exp1 \
MODEL_PATH=checkpoints/ttt-rtl-sft/global_step_18/huggingface \
  bash recipe/ttt_rtl/run_ttt_rtl.sh
```

Key differences from the base config (`ttt_rtl_fp_trainer.yaml` vs `ttt_rtl_trainer.yaml`):

| Parameter | Base | FloatPoint | Why |
|-----------|------|------------|-----|
| `max_prompt_length` | 4096 | 16384 | Discover requires full previous code in prompt; fp_add generates thousands of lines |
| `max_response_length` | 6144 | 12288 | thinking (4K) + code (8K); behavioral FP adder can reach 500-2000 lines |
| `tensor_model_parallel_size` | 1 | 2 | Model halved across 2 GPUs (8 GB/GPU vs 16 GB) → KV cache doubles to 32 GB/GPU |
| `gpu_memory_utilization` | 0.4 | 0.5 | More KV cache budget for longer sequences |
| `ppo_max_token_len_per_gpu` | 12288 | 32768 | FSDP dynamic batching budget matches 28K total context |
| `max_num_batched_tokens` | 16384 | 32768 | vLLM prefill handles 16K prompts without excessive chunking |
| `max_model_len` | null | 32768 | Explicit vLLM context window (prompt + response) |
| `timeout_s` | 30 | 120 | 4021-line Verilog through Yosys+OpenSTA needs 60-90s |
| `opensta_timeout_s` | 30 | 60 | STA on large gate-level netlists takes longer |
| `problems` | multi_pipe_4bit | fp_add | Target problem |
| `total_training_steps` | 100 | 500 | fp_add is 50-130x harder than typical RTLLM problems |

### Multi-Problem Training

```bash
export MODEL_PATH=~/models/Qwen2.5-7B-Instruct
export RTLLM_DIR=RTLLM
export RTL_PROBLEMS="adder_8bit,counter_12,fsm"
bash recipe/ttt_rtl/run_ttt_rtl.sh
```

### Multi-Node Training (SLURM)

#### Single experiment

Allocate 2 nodes interactively, then run:

```bash
salloc -N 2 --gres=gpu:4 --cpus-per-task=16 -t 04:00:00
conda activate verl

MODEL_PATH=checkpoints/ttt-rtl-sft/global_step_18/huggingface \
RTL_PROBLEMS=fsm \
EXP_NAME=fsm_exp1 \
SAVE_FREQ=-1 \
bash recipe/ttt_rtl/debug_multinode.sh
```

Or submit as a batch job (non-interactive):

```bash
MODEL_PATH=checkpoints/ttt-rtl-sft/global_step_18/huggingface \
RTL_PROBLEMS=fsm \
EXP_NAME=fsm_exp1 \
SAVE_FREQ=-1 \
sbatch recipe/ttt_rtl/run_ttt_rtl_multinode.slurm
```

#### Many experiments in one go

Create an `experiments.conf` file (one experiment per line):

```
# PROBLEMS           | EXP_NAME               | MODEL_PATH
fsm                  | fsm_new                | checkpoints/ttt-rtl-sft/global_step_18/huggingface
counter_12           | counter_12_new         | checkpoints/ttt-rtl-sft/global_step_18/huggingface
alu                  | alu_new                | checkpoints/ttt-rtl-sft/global_step_18/huggingface
```

A ready-made config covering all 50 RTLLM problems is provided at `experiments.conf` in the repo root.

Then submit all experiments at once:

```bash
bash recipe/ttt_rtl/submit_experiments.sh experiments.conf
```

Each experiment becomes an independent `sbatch` job. Jobs queue up and run in parallel as nodes become available.

```bash
squeue -u $USER          # monitor jobs
scancel <job_id>         # cancel one
scancel -u $USER         # cancel all
```

Logs go to `verl_rtl_<jobid>.out/.err` in the working directory, and training logs to `outputs/ttt_rtl_logs/`.

### SFT Warm-up (optional)

A supervised fine-tuning pipeline is provided to warm-start the model before RL:

```bash
# See recipe/ttt_rtl/sft/README.md for full details
bash recipe/ttt_rtl/sft/run_pipeline.sh
```

## Algorithm Overview

Training proceeds by iterative PUCT-guided tree search over Verilog code:

1. **Initial state**: The reference (verified) implementation for each problem is shown to the model, with the baseline cell count as a target to beat.
2. **PUCT sampling**: Parent states are sampled from the state pool using a UCB-based exploration bonus.
3. **Rollout**: The model generates `n` candidate Verilog implementations per parent state.
4. **Hierarchical reward**: Each candidate is evaluated by the EDA toolchain:
   - Level 1 — Syntax: compiles with `iverilog` → `r_syntax`
   - Level 2 — Functional: passes testbench simulation (`vvp`) → `r_func`
   - Level 3 — PPA: Yosys technology-mapped synthesis against the NanGate 45nm PDK → OpenSTA static timing + activity-less power → `r_ppa = baseline / actual`. The optimization metric is **A·D·P** (Area × Delay × Power) when `include_power=true`, or **ADP** (Area × Delay) when disabled. Falls back to ABC timing if OpenSTA is unavailable, and to generic cell-count synthesis when no Liberty file is configured.
5. **Advantage estimation**: Entropic adaptive-β advantage with binary search for β satisfying KL ≈ ln(2).
6. **Policy update**: Importance-sampling gradient (`-log_prob × advantage`) via the `gpg` loss mode.
7. **State pool update**: Top-k children per parent are kept; deduplicated by normalized Verilog hash.

## Configuration

### Config Files

Two config files are provided in `recipe/ttt_rtl/config/`:

| Config | Use Case | Key Settings |
|--------|----------|--------------|
| `ttt_rtl_trainer.yaml` | Standard problems (adder, counter, fsm, etc.) | 4K prompt, 6K response, TP=1 |
| `ttt_rtl_fp_trainer.yaml` | Large designs (fp_add, etc.) on 8×A800 80 GB | 16K prompt, 12K response, TP=2 |

Select a config via the `CONFIG_NAME` env var (default: `ttt_rtl_trainer`):

```bash
CONFIG_NAME=ttt_rtl_fp_trainer bash recipe/ttt_rtl/run_ttt_rtl.sh
```

### Key Parameters (`ttt_rtl_trainer.yaml`)

```yaml
data:
  train_batch_size: 4           # Parent states sampled per step
  max_prompt_length: 4096       # Prompt token budget
  max_response_length: 6144     # Max tokens for generated Verilog
  ttt_rtl:
    problems: "multi_pipe_4bit" # RTLLM problems (comma-separated)
    max_buffer_size: 500        # Max states in PUCT pool
    topk_children: 2            # Keep best k children per parent
    puct_c: 1.0                 # PUCT exploration constant
    lineage_blocking: true      # Block ancestor/descendant states in same batch
    seed: 42

custom_reward_function:
  reward_kwargs:
    timeout_s: 30.0             # Timeout per EDA tool invocation
    w_syntax: 0.1               # Weight for syntax correctness
    w_func: 1.0                 # Weight for functional correctness
    w_ppa: 10.0                 # Weight for PPA reward
    failure_reward: 0.0         # Reward when no Verilog code is extracted
    liberty_file: recipe/ttt_rtl/pdk/Nangate45_typ.lib  # NanGate 45nm PDK
    opensta_binary: sta  # set null to use ABC only
    opensta_timeout_s: 30.0     # Timeout for OpenSTA STA run
    clk_period_ns: 10.0         # Target clock period (10 ns = 100 MHz)
    include_power: true         # Fold OpenSTA power into PPA: ppa_score = area*delay*power_w

algorithm:
  adv_estimator: ttt_entropic_adaptive
  use_kl_in_advantage: true     # Entropic KL-based advantage estimation
  ttt_kl_penalty_coef: 0.1
  ttt_entropic_kl_budget: 0.693 # Target KL ≈ ln(2)
  ttt_remove_constant_reward_groups: true

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gpg             # Importance-sampling gradient
  rollout:
    n: 8                         # Rollouts per parent state
    temperature: 1.0
    top_p: 1.0

trainer:
  total_training_steps: 100
  save_freq: 50
  logger: ["console", "swanlab"]
```

### Environment Variable Overrides

```bash
# Model and data
export MODEL_PATH=~/models/Qwen2.5-7B-Instruct
export RTLLM_DIR=RTLLM
export RTL_PROBLEMS="adder_8bit,adder_16bit,counter_12"

# Config selection (default: ttt_rtl_trainer)
export CONFIG_NAME=ttt_rtl_trainer  # or ttt_rtl_fp_trainer for FloatPoint

# Training
export NNODES=1
export NGPUS_PER_NODE=8
export ROLLOUT_TP=2              # only set to override YAML; unset uses YAML default
export SAVE_FREQ=50
export RESUME_MODE=disable       # or "auto" to resume from checkpoint

# Sampling
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0

# Reward
export REWARD_TIMEOUT_S=30.0
export FAILURE_REWARD=0.0
export LIBERTY_FILE=recipe/ttt_rtl/pdk/Nangate45_typ.lib  # NanGate 45nm
export INCLUDE_POWER=true        # true: A*D*P reward, false: A*D only
# OpenSTA for accurate STA (set empty to disable, use ABC timing instead)
export OPENSTA_BINARY=sta

# Logging
export SWANLAB_MODE=local        # or "cloud"
export SWANLAB_LOG_DIR=~/swanlog

bash recipe/ttt_rtl/run_ttt_rtl.sh
```

## Prompt Design

Each prompt is built from the current state's context. The displayed metric label adapts to the active reward formula — `A*D*P` when `include_power=true`, `ADP` otherwise:

- **Initial state** (no prior attempt): Shows the verified reference implementation with its baseline PPA metrics (area μm², delay ps, power μW when enabled, and the active metric), then asks the model to produce a correct design with a lower metric value.
- **Non-initial state**: Shows the previous Verilog attempt, its syntax/functional/synthesis feedback, and a delta comparison (e.g., "A*D*P improved by 1.23e-5"). The instruction adapts:
  - If functionally correct with PPA: "produce a design with <metric> lower than N"
  - If functionally correct without PPA: "optimize for better PPA in synthesis"
  - If failing: "improve based on feedback"

## SwanLab Integration

SwanLab is enabled by default (`trainer.logger: ["console", "swanlab"]`).

**Environment Variables:**

```bash
# Local mode (default) — stores logs locally
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=~/swanlog

# Cloud mode — syncs to SwanLab cloud
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=your_api_key_here
```

**Disable SwanLab:**

```bash
export LOGGER='["console"]'
bash recipe/ttt_rtl/run_ttt_rtl.sh
```

### Logged Metrics

#### Core Training Metrics

| Metric | Description |
|--------|-------------|
| `train/policy_loss` | Policy gradient loss |
| `train/entropy` | Policy entropy |
| `train/approx_kl` | Approximate KL divergence |
| `train/learning_rate` | Current learning rate |
| `train/grad_norm` | Gradient norm |

#### TTT-Discover Metrics

| Metric | Description |
|--------|-------------|
| `ttt/success_rate` | Fraction of rollouts with valid Verilog (passed extraction) |
| `ttt/syntax_pass_rate` | Fraction passing iverilog compilation |
| `ttt/func_pass_rate` | Fraction passing testbench simulation |
| `ttt/func_pass_with_ppa_rate` | Fraction that passed functional and have PPA > 0 |
| `ttt/syntax_only_rate` | Fraction that passed syntax but failed functional |
| `ttt/reward_mean` | Mean reward across rollouts |
| `ttt/reward_max` | Best reward in current batch |
| `ttt/reward_min` | Worst reward in current batch |
| `ttt/reward_std` | Reward standard deviation |
| `ttt/non_constant_group_ratio` | Fraction of groups with non-constant rewards (effective batch ratio) |
| `ttt/effective_batch_ratio` | Same as above — groups yielding gradient signal |
| `ttt/unique_code_ratio` | Fraction of rollouts with distinct Verilog (diversity) |
| `ttt/group_reward_std_mean` | Mean within-group reward std |
| `ttt/num_groups` | Number of parent states sampled |
| `ttt/rollouts_per_step` | Total rollouts per training step |
| `ttt/best_reward_ever` | Best reward seen across all training |
| `ttt/best_reward_per_problem_mean` | Mean of per-problem best rewards |
| `ttt/tree_depth_max` | Maximum depth of search tree |
| `ttt/tree_depth_mean` | Mean depth of states in pool |
| `ttt/new_states_added` | States added to pool last step |

#### PUCT State Pool Metrics

| Metric | Description |
|--------|-------------|
| `puct/buffer_size` | Current number of states in pool |
| `puct/T` | Total PUCT visit count |
| `puct/value_max` | Best state value in pool |
| `puct/value_mean` | Mean state value in pool |
| `puct/syntax_ok_rate` | Fraction of pool states with valid syntax |
| `puct/func_ok_rate` | Fraction of pool states passing tests |

#### Per-Problem Metrics

| Metric | Description |
|--------|-------------|
| `ttt_problem/<pid>/best_reward` | Best reward ever achieved for problem `pid` |

#### Sample Outputs (logged every 10 steps)

| Metric | Description |
|--------|-------------|
| `ttt/sample_outputs` | Text preview of generated Verilog + feedback for 3 rollouts |

### Viewing Results

**Local mode:**
```bash
swanlab watch --logdir ~/swanlog
# Open http://localhost:5092 in browser
```

**Cloud mode:**
```bash
# Visit https://swanlab.cn/@your_username/verl-ttt-rtl
```

## Algorithm Components

1. **PUCT State Pool** (`rtl_state.py`) — Rank-based prior, UCB exploration bonus, lineage blocking, top-k children pruning, max-buffer-size pruning, construction deduplication via SHA-256 of normalized Verilog.
2. **Entropic Adaptive-β Advantage** (`recipe/ttt_math_construct/algorithms.py`) — Binary search for β satisfying KL(q‖uniform) ≈ ln(2), leave-one-out normalization, constant-group filtering.
3. **GPG Policy Loss** — Importance-sampling gradient: `-log_prob × advantage`.
4. **Hierarchical Reward** (`rtl_reward.py`) — Syntax (iverilog) → Functional (vvp) → PPA (Yosys + NanGate 45nm PDK + OpenSTA). When a Liberty file is configured, synthesis is technology-mapped producing real physical area (μm²). Critical-path delay (ps) comes from OpenSTA static timing analysis when `opensta_binary` is set, otherwise falls back to ABC's `stime` estimate. OpenSTA also reports activity-less power (default 10% toggle / 50% duty). The PPA metric is `area × delay × power_w` (A·D·P) when `include_power=true`, else `area × delay` (ADP). Reward is ratio-based: `r_ppa = baseline / actual`. Without Liberty, falls back to generic cell-count synthesis. Total: `reward = w_syntax·r_syntax + w_func·r_func + w_ppa·r_ppa`.
5. **Auto PPA Baseline** (`dataset.py`) — On startup, synthesizes each problem's reference implementation (with the same Liberty file) to obtain per-problem PPA baselines (ADP or A·D·P, matching the active metric) used in reward and prompt.
6. **Reference Injection** (`dataset.py`) — Initial-state prompts include the verified reference implementation so the model has a working starting point.
7. **Construction Deduplication** (`rtl_state.py`) — Verilog normalized by stripping comments and collapsing whitespace before hashing; duplicates are rejected from the pool.
8. **State Persistence** (`rtl_state.py`) — Pool saved atomically as JSON after each batch; optional step-level history snapshots for analysis.

## PPA Reward with NanGate 45nm PDK

By default, the PPA reward uses **technology-mapped synthesis** against the NanGate 45nm open-source standard-cell library (`recipe/ttt_rtl/pdk/Nangate45_typ.lib`), producing physically meaningful metrics instead of abstract cell counts. When `include_power=true` (the default), OpenSTA activity-less power is folded into the reward so the optimization target is **A·D·P (Area × Delay × Power)** rather than plain ADP.

### How It Works

The PPA pipeline runs in two stages:

**Stage 1 — Yosys technology mapping:**

```tcl
read_verilog design.v
synth -top <module>
dfflibmap -liberty Nangate45_typ.lib   # map DFFs to real cells
abc -liberty Nangate45_typ.lib          # technology mapping
write_verilog -noattr gate_netlist.v    # gate-level netlist for STA
stat -liberty Nangate45_typ.lib         # report physical area
```

**Stage 2 — OpenSTA static timing + activity-less power** (when `opensta_binary` is configured):

```tcl
read_liberty Nangate45_typ.lib
read_verilog gate_netlist.v
link_design <module>
# auto-detects clock port (clk/clock/CLK/…); creates create_clock at clk_period_ns
# combinational circuits use report_checks -unconstrained
report_checks -path_delay max -digits 4
report_worst_slack -digits 4
report_power -digits 6                  # internal + switching + leakage (activity-less)
```

This produces:
- **Area** — real physical area in μm² from Yosys `stat -liberty`
- **Delay** — critical-path arrival time in ps from OpenSTA; falls back to ABC estimate when OpenSTA is unavailable or fails
- **WNS** — worst negative slack in ps; reported in training feedback (negative = timing violated at `clk_period_ns`)
- **Power** — total power in W from OpenSTA `report_power` (internal + switching + leakage, default 10% toggle / 50% duty)
- **Primary metric** — `A·D·P = area × delay × power_w` when `include_power=true`, else `ADP = area × delay`

The reward is: `r_ppa = baseline / actual`, where `baseline` is auto-computed from the reference implementation at startup using the same formula.

### Why activity-less power

`report_power` without a VCD uses propagated default activity (10% toggle rate, 50% duty). This gives leakage + a consistent internal/switching estimate without adding VCD-generation cost to every rollout. On a `multi_pipe_8bit` comparison the best RL candidate reduced total power by ~40% relative to the RTLLM reference, driven primarily by accurate leakage and internal power accounting — showing the signal is discriminative even without real switching activity.

### Why OpenSTA instead of ABC stime

ABC's `stime -p` estimates delay during technology mapping — it lacks complete timing-arc context and does not perform proper STA. OpenSTA runs full static timing analysis on the mapped netlist using the Liberty model, accounting for all cell timing arcs, input/output delays, and setup constraints. The difference grows with circuit depth.

### Synthesis overhead

Yosys tech mapping: ~0.2–2s per design. OpenSTA STA + power: ~1–3s. Both are negligible relative to the vLLM rollout time.

### Benchmarking

Use `benchmark_yosys.py` to profile synthesis on reference designs:

```bash
python recipe/ttt_rtl/benchmark_yosys.py \
    --rtllm-dir RTLLM \
    --problems "adder_8bit,counter_12,multi_pipe_4bit" \
    --liberty-file recipe/ttt_rtl/pdk/Nangate45_typ.lib
```

Output includes Area (μm²), Delay (ps), ADP, and recommended `timeout_s` values.

### Switching to ADP (Area × Delay only)

To disable the power term and optimize plain ADP:

```bash
INCLUDE_POWER=false bash recipe/ttt_rtl/run_ttt_rtl.sh
```

Or in the YAML config:

```yaml
data:
  ttt_rtl:
    include_power: false
custom_reward_function:
  reward_kwargs:
    include_power: false  # must match data.ttt_rtl.include_power so units agree
```

> Both flags must agree — the baseline (computed in `dataset.py`) and the online reward (`rtl_reward.py`) need to use the same formula, otherwise `r_ppa = baseline / actual` is unit-inconsistent.

### Disabling OpenSTA (use ABC timing only)

```bash
# via env (add to run script)
OPENSTA_BINARY="" bash recipe/ttt_rtl/run_ttt_rtl.sh
```

Or in the YAML config:

```yaml
custom_reward_function:
  reward_kwargs:
    opensta_binary: null   # null or empty string disables OpenSTA; ABC stime is used
```

Note: disabling OpenSTA also disables power reporting, so `include_power=true` silently falls back to ADP in that case.

### Disabling technology-mapped synthesis entirely

To fall back to generic cell-count synthesis (no Liberty):

```bash
LIBERTY_FILE="" bash recipe/ttt_rtl/run_ttt_rtl.sh
```

Or in the YAML config:

```yaml
custom_reward_function:
  reward_kwargs:
    liberty_file: ""   # empty string disables tech-mapped synthesis
```

## RTLLM Benchmark

The implementation supports the RTLLM benchmark with a 3-level directory structure:
```
RTLLM/
  Category/
    Subcategory/
      <problem>/
        makefile               ← defines TEST_DESIGN (used as problem_id)
        design_description.txt
        testbench.v
        verified_<name>.v      ← reference implementation
        [*.txt | *.dat]        ← data files for $readmemh testbenches
```

**Problem Selection:**

```yaml
# In config
data:
  ttt_rtl:
    problems: "adder_8bit,counter_12,fsm"

# Or via environment
export RTL_PROBLEMS="adder_8bit,adder_16bit,counter_12,fsm,edge_detect"
```

If `problems` is not set, all problems found in `rtllm_dir` are loaded.

## State Pool Checkpointing

The state pool is saved to `checkpoints/<project>/<exp>/ttt_rtl_state_pool/`:

- `latest.json` — current pool state (overwritten each save)
- `state_pool_step_NNNNNN.json` — periodic snapshots (interval: `save_freq`)

Training auto-resumes from `latest.json` if present. To start fresh:

```bash
rm -rf checkpoints/<project>/<exp>/ttt_rtl_state_pool/
# or use run_multi_pipe_4bit.sh which does this automatically
```

## Troubleshooting

### SwanLab Issues

**Transaction bug in local mode:**
- Automatically patched in `main_ttt_rtl.py`
- If you see `peewee.OperationalError: cannot commit - no transaction is active`, the patch suppresses it; metric logging continues normally

**Cloud sync fails:**
```bash
export SWANLAB_MODE=local
```

### EDA Tool Issues

**iverilog / vvp not found:**
```bash
sudo apt-get install iverilog
# macOS: brew install icarus-verilog
```

**Yosys not found:**
```bash
sudo apt-get install yosys
# Disable PPA reward as a fallback:
# custom_reward_function.reward_kwargs.w_ppa=0.0
```

**PPA baseline is 0 for some problems:**
- Yosys synthesis of the reference failed at startup — PPA baseline will be 0 for affected problems
- Check that `yosys` is installed and the reference `.v` file exists
- The fallback PPA reward formula `1 / (1 + area/100)` is used automatically when baseline is 0

### Memory Issues

**OOM during rollout:**
```bash
# Reduce batch size
python -m recipe.ttt_rtl.main_ttt_rtl data.train_batch_size=2

# Reduce rollouts per parent
python -m recipe.ttt_rtl.main_ttt_rtl actor_rollout_ref.rollout.n=4

# Reduce max token budget
python -m recipe.ttt_rtl.main_ttt_rtl actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{discover2024,
  title={TTT-Discover: Test-Time Training for Algorithmic Discovery},
  author={NVIDIA Research},
  year={2024}
}

@misc{verl2024,
  title={verl: Volcano Engine Reinforcement Learning for LLMs},
  author={ByteDance},
  year={2024}
}
```

## License

Apache-2.0 (same as verl)
