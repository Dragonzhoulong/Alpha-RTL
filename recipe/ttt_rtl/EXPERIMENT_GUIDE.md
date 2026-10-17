# TTT-Discover RTL Experiment Guide

## Quick Start: Switching Problems

The recipe is fully generic. To run on any RTLLM problem, just set two environment variables:

```bash
cd verl

# Example: run multi_pipe_8bit
RTL_PROBLEMS=multi_pipe_8bit EXP_NAME=multi_pipe_8bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh

# Example: run adder_pipe_64bit
RTL_PROBLEMS=adder_pipe_64bit EXP_NAME=adder_pipe_64bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh

# Example: run alu
RTL_PROBLEMS=alu EXP_NAME=alu_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh
```

No Python code or YAML changes needed. The `RTL_PROBLEMS` value must match either the
directory name (e.g. `multi_pipe_4bit`) or `TEST_DESIGN` in the problem's makefile.

### What happens automatically

| Component | How it resolves the problem |
|-----------|---------------------------|
| `dataset.py` | Reads `design_description.txt` for prompt |
| `dataset.py` | Reads `verified_*.v` as reference implementation |
| `dataset.py` | Extracts correct module name from `testbench.v` |
| `rtl_reward.py` | Uses problem-specific `testbench.v` for functional test |
| `rtl_utils.py` | Extracts actual module name from generated code for Yosys |
| `rtl_state.py` | Tracks per-problem best reward and PPA |

### Setting PPA baseline per problem

Each problem needs a `ppa_baseline` (the reference implementation's cell count) to
enable ratio-based PPA reward. Pass it as a hydra override:

```bash
# multi_pipe_8bit: reference = 519 cells
RTL_PROBLEMS=multi_pipe_8bit EXP_NAME=multi_pipe_8bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh \
  custom_reward_function.reward_kwargs.ppa_baseline=519

# adder_pipe_64bit: reference = 809 cells
RTL_PROBLEMS=adder_pipe_64bit EXP_NAME=adder_pipe_64bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh \
  custom_reward_function.reward_kwargs.ppa_baseline=809

# multi_pipe_4bit: reference = 93 cells
RTL_PROBLEMS=multi_pipe_4bit EXP_NAME=multi_pipe_4bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh \
  custom_reward_function.reward_kwargs.ppa_baseline=93
```

To get a problem's baseline, synthesize its verified implementation:
```bash
cd /tmp && mkdir synth_test && cd synth_test
sed 's/verified_\w\+/MODULE_NAME/' RTLLM/.../verified_*.v > design.v
echo "read_verilog design.v; synth -top MODULE_NAME; stat" > synth.tcl
yosys -s synth.tcl 2>&1 | grep cells
```

### Optional: Override reward weights per problem

```bash
RTL_PROBLEMS=multi_pipe_8bit EXP_NAME=multi_pipe_8bit_exp1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh \
  custom_reward_function.reward_kwargs.w_syntax=0.1 \
  custom_reward_function.reward_kwargs.w_func=1.0 \
  custom_reward_function.reward_kwargs.w_ppa=10.0 \
  custom_reward_function.reward_kwargs.ppa_baseline=519
```

### Optional: Override training parameters

```bash
RTL_PROBLEMS=multi_pipe_8bit EXP_NAME=multi_pipe_8bit_exp1 \
  SAVE_FREQ=-1 \
  bash recipe/ttt_rtl/run_ttt_rtl.sh \
  data.train_batch_size=8 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=200
```

## multi_pipe_4bit Experiment Results

### Conclusion: Not suitable for TTT-Discover

PPA optimization headroom is only **6.5%** (93 → 87 cells). The model converges
to the optimal solution within ~20 steps and then produces identical outputs.

| Implementation | Cells | r_ppa | Functional |
|---------------|-------|-------|------------|
| Verified reference (generate+shift) | 93 | 0.518 | PASS |
| Model output (uses `*` operator) | 86-87 | 0.535 | PASS |
| Theoretical minimum (Yosys optimized) | ~87 | 0.535 | PASS |

### Observed training behavior

- **Steps 1-10**: syntax pass rate ~75%, func pass rate ~30-60%
- **Steps 10-30**: func pass rate reaches 80-100%
- **Steps 30+**: All 32 samples identical score (6.476), advantage=0, no gradient
- **Conclusion**: Model quickly memorizes the reference and generates near-identical code

### Bugs found and fixed during this experiment

1. **Yosys module name mismatch** (`rtl_utils.py`): `synth -top` used `design_name` from
   makefile (`multi_pipe`) but generated code has module name `multi_pipe_4bit`.
   Fixed: extract actual module name from generated code with regex.

2. **Checkpoint save failure on Lustre**: 8 GPU workers writing 2-3GB files concurrently
   causes `PytorchStreamWriter failed`. Workaround: set `SAVE_FREQ=-1` to disable,
   or save to local SSD first.

## Choosing a Good Problem for TTT-Discover

### Criteria

A good problem should have:
1. **High func pass rate** (>50%) — model must be able to generate correct code
2. **Wide PPA range** — different valid implementations should have very different cell counts
3. **Moderate complexity** — achievable for 7B model (reference <100 lines)
4. **Multiple algorithmic approaches** — not just one canonical solution

### Evaluation method

```bash
# Check PPA of reference implementation
cd /tmp && mkdir test && cd test
sed 's/verified_MODULE/MODULE/' RTLLM/.../verified_MODULE.v > MODULE.v
echo "read_verilog MODULE.v; synth -top MODULE; stat" > synth.tcl
yosys -s synth.tcl 2>&1 | grep cells
```

### Evaluated problems

| Problem | Ref cells | Best cells | Headroom | ppa_baseline | Verdict |
|---------|-----------|------------|----------|--------------|---------|
| multi_pipe_4bit | 93 | 87 | 6.5% | 93 | Too narrow |
| multi_pipe_8bit | 519 | 469 | 9.6% | 519 | Marginal |
| adder_pipe_64bit | 809 | 752 | 7.0% | 809 | Marginal |

#### multi_pipe_8bit (519 → 469 cells, 9.6% headroom)

| Implementation | Cells | Functional |
|---------------|-------|------------|
| Reference (manual partial products) | 519 | PASS |
| Generate + 4-group partial sums | 527 | PASS |
| `*` operator, 4-stage pipeline | 488 | PASS |
| `*` operator, 2-stage + pass-through | 472 | PASS |
| Minimal `*` with 3 regs | 469 | PASS |

#### adder_pipe_64bit (809 → 752 cells, 7.0% headroom)

| Implementation | Cells | Functional |
|---------------|-------|------------|
| 2-stage split (16+48) | 833 | PASS |
| Direct +, 4-stage delay | 817 | PASS |
| Reference (4-stage, 16-bit segments) | 809 | PASS |
| Simple + with 3 regs | 752 | PASS |

### Candidates still to evaluate

| Problem | Ref lines | Notes |
|---------|-----------|-------|
| radix2_div | 83 | Multiple division algorithms |
| alu | 111 | 17 operations, complex but many optimizations |
| float_multi | 164 | May be too hard for 7B |

## Configuration Reference

### Key files (no problem-specific code)

```
recipe/ttt_rtl/
├── config/ttt_rtl_trainer.yaml  # Training config (problems, reward weights, batch size)
├── dataset.py                   # Prompt construction (auto reads reference + testbench)
├── rtl_reward.py                # Hierarchical reward (syntax → func → PPA)
├── rtl_utils.py                 # EDA tool wrappers (iverilog, yosys)
├── rtl_state.py                 # PUCT state pool
├── sampler.py                   # Curriculum sampler + SwanLab metrics
├── run_ttt_rtl.sh               # Main launch script (env vars → hydra overrides)
└── main_ttt_rtl.py              # Hydra entry point
```

### Reward formula

```
reward = w_syntax × r_syntax + w_func × r_func + w_ppa × r_ppa

r_syntax = 1.0 if iverilog compiles, else 0.0
r_func   = 1.0 if testbench prints "Your Design Passed", else 0.0
r_ppa    = ppa_baseline / yosys_cell_count   (when ppa_baseline > 0, only if func_ok)
         = 1.0 / (1.0 + yosys_cell_count / 100.0)   (fallback when ppa_baseline = 0)
```

**Why ratio-based PPA reward?** The old formula `1/(1+area/100)` becomes very flat for
large cell counts. A 10% cell reduction gives very different reward deltas depending on
absolute size:

| Problem | Optimization | OLD Δreward | NEW Δreward | Improvement |
|---------|-------------|-------------|-------------|-------------|
| multi_pipe_4bit | 93→87 | 0.166 | 0.690 | 4.1x |
| multi_pipe_8bit | 519→469 | 0.142 | 1.066 | 7.5x |
| adder_pipe_64bit | 809→752 | 0.074 | 0.758 | 10.2x |

The new formula `baseline/actual` gives consistent reward signal: the same percentage
of cell reduction always produces a proportional reward increase, regardless of problem size.

### Suggested reward weights by problem difficulty

| Difficulty | w_syntax | w_func | w_ppa | Rationale |
|-----------|----------|--------|-------|-----------|
| Easy (counter, shifter) | 0.1 | 1.0 | 10.0 | Focus on PPA optimization |
| Medium (multiplier, FSM) | 1.0 | 5.0 | 5.0 | Balance func and PPA |
| Hard (ALU, divider) | 1.0 | 10.0 | 1.0 | Prioritize functional correctness |

### SwanLab metrics

| Metric | Meaning | Healthy range |
|--------|---------|--------------|
| `ttt/func_pass_rate` | Functional test pass rate | >50% |
| `ttt/effective_batch_ratio` | Groups with non-constant rewards | >0.3 |
| `ttt/best_reward_ever` | Historical best reward | Should increase |
| `ttt/unique_code_ratio` | Code diversity per step | >0.5 |
| `ttt/new_states_added` | New states entering pool per step | >0 |
| `actor/grad_norm` | Gradient magnitude | >0 (must be non-zero) |
| `actor/pg_loss` | Policy gradient loss | Non-zero |
| `puct/tree_depth_max` | Search tree depth | Should grow |
| `ttt_problem/{pid}/best_reward` | Per-problem best | Should increase |
