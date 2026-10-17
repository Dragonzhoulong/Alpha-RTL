# TTT-C910 FPU Framework (NeurIPS Figure Spec)

This document describes the end-to-end framework of `recipe/ttt_c910_fpu/` and provides a layout/labeling specification for the framework figure of the NeurIPS submission.

> One-line caption (suggested):
> **TTT-Discover applied to industrial RTL: a PUCT-curriculum LLM agent rewrites OpenC910 FPU components, scored by a hierarchical golden-model + PPA reward, and improved with an entropic-β policy gradient whose KL budget δ is driven by a metric-based adaptive controller.**

---

## 1. High-level Pipeline

The framework is a closed RL loop with five stages, repeated for `T` training steps:

```
   ┌──────────────┐   parent state s_t       ┌─────────────────┐
   │ PUCT State   │ ───────────────────────▶ │  Prompt Builder │
   │  Pool  𝒮     │                          │   (state→ctx)   │
   │   (per       │ ◀─── top-k children ──── └────────┬────────┘
   │   problem)   │     {(s_t, c_i, r_i)}             │
   └──────┬───────┘                                   │ rendered prompt
          ▲                                           ▼
          │                                  ┌─────────────────┐
          │                                  │   LLM Policy    │
          │  rollout batch B                 │  πθ  (vLLM,     │
          │  + advantages                    │   n=8 / parent) │
          │                                  └────────┬────────┘
          │                                           │ verilog candidates
          │                                           ▼
   ┌──────┴─────────┐                        ┌─────────────────┐
   │  Policy Update │ ◀── (advantage, kl) ── │  Hierarchical   │
   │ (FSDP, GPG)    │                        │  Reward (EDA)   │
   │  + Adaptive-β  │                        │  iverilog/yosys │
   │   Advantage    │                        │  /OpenSTA       │
   └────────────────┘                        └─────────────────┘
```

The five blocks (in the figure they should be five large nodes plus the EDA reward subgraph):

1. **PUCT State Pool 𝒮** — per-problem search tree of candidate designs.
2. **Prompt Builder** — turns a sampled parent state into a structured Verilog-design prompt.
3. **LLM Rollout Policy πθ** — long-context generator (Qwen3-8B, YaRN-extended to 65k).
4. **Hierarchical EDA Reward** — three-level filter: syntax → functional (golden model) → PPA.
5. **Entropic Adaptive Policy Update** — KL-regularized advantage on top of the GPG policy loss.

---

## 2. Block-by-block Specification

### 2.1 Targets — OpenC910 FPU Components

Industrial-grade RTL drawn from the OpenC910 RISC-V CPU's Vector Floating-point Multiply-Accumulate Unit (VFMAU). Each problem lives under `problems/<id>/` and ships with:
`spec.yaml` · `reference.v` · `reference_renamed.v` (golden) · `testbench.v` · `deps/` · `prompt_template.txt`.

| Problem | Module | What it does | Baseline (sky130, OpenSTA) |
|---|---|---|---|
| `mult_compressor` | `ct_vfmau_mult_compressor` | Booth-radix-4 + 5-level Wallace tree, 27→2 partial products, EX1→EX2 pipeline | area 122,122 μm², delay 9,903 ps, ADP 1.21×10⁹ |
| `lza` | leading-zero anticipator (double) | predicts MSB of `a−b` from operands |  |
| `lza_simd_half` | LZA, half-precision SIMD lane |  |  |
| `fadd_close_s0_d` | close-path FP adder, sign=0, double | special-case adder datapath |  |

> **Figure note:** show one schematic block "OpenC910 FPU" with dotted boundary, with chips for the 4 problems. Use the `mult_compressor` Wallace-tree mini-diagram as an inset to make the industrial nature concrete.

### 2.2 PUCT State Pool 𝒮 (`recipe/ttt_rtl/rtl_state.py`, reused)

A per-problem archive of candidate states `s = (verilog_code, value, ppa_area, ppa_delay, parents)` with the following operations:

- **Sampling** (`sample_parent_state_ids`): for batch size `B` it returns `B` parent states by the score
  $$\text{PUCT}(s) = Q(s) \;+\; c\,\sigma\,P(s)\,\frac{\sqrt{1+T}}{1+n(s)},$$
  with `Q(s)` the running mean child reward, `P(s)` a rank-based prior over `value`, `σ` the value-range scale, `T` the global visit count, `n(s)` per-state visits, `c=1.0`.
- **Lineage blocking**: when one ancestor is picked, all its descendants are blocked from the same batch (forces diversity across the batch).
- **Top-k children** (`k=2`): per parent, only the best 2 children by reward are inserted to limit branching factor.
- **Construction dedup**: SHA-256 over whitespace/comment-normalized Verilog prevents re-inserting identical designs.
- **Seeded init**: at step 0, the pool injects all *verified* SFT solutions from `problems/<id>/solutions/reward_measurements.json` as permanent initial states.
- **Buffer pruning** to `max_buffer_size=500`.
- **Persistence**: `state_pool_dir` snapshots are saved every `save_freq` steps for restart-safety.

> **Figure note:** draw 𝒮 as a tree. Root = initial state (reference seed). Edges to children are labeled with reward; an "active batch" is highlighted across non-overlapping subtrees (lineage blocking). Three small Q/P/n badges next to each node make the PUCT formula tangible.

### 2.3 Prompt Builder (`dataset.py:_build_user_prompt`)

Two prompt regimes, deterministically chosen by whether the parent state has code:

- **Initial states** — `Reference Implementation` block (with baseline ADP) + `Design Specification` (full template) + ADP optimization goal.
- **Iterative states** — full spec (with `## Baseline Implementation` and `## Optimization Goals` sections stripped) + `## Previous Attempt` (parent's Verilog) + `## Feedback` (syntax/functional/PPA results, last-800-char tail of simulator log) + targeted hint ("try a different tree topology / encoding / grouping").

Long-context budget: `max_prompt_length = 24,576` tokens; reference + spec for `mult_compressor` is ~16k tokens, leaving headroom for chat template + iterative feedback.

> **Figure note:** show two prompt cards side-by-side, `t=0` vs `t>0`, to make the iterative-improvement nature visible. The `t>0` card should highlight the simulator-log feedback as the closing block, since this is the dominant signal that pushes the LLM toward fixes.

### 2.4 LLM Rollout Policy πθ

- Backbone: **Qwen3-8B** (default) with **YaRN-extended 65,536 context**.
- Engine: **vLLM** with `tensor_model_parallel_size=2`, `max_num_batched_tokens=65,536`, chunked prefill.
- Sampling: `n = 8` rollouts per parent state, `temperature=1.0`, `top_p=1.0`, `max_response_length=24,576` (≈5k thinking + ≈16k code).
- "Thinking" mode is enabled in the chat template (`enable_thinking=True`); `extract_verilog_code` strips `<think>...</think>` before grading.
- Per step: `train_batch_size=2` parents × `n=8` ≈ 16 candidates per step on 2 nodes × 8 GPUs.

> **Figure note:** the LLM block should expose three sub-arrows from one prompt: `<think>`, `<verilog>` (kept), and `<other>` (discarded). Mark the YaRN-65k context window explicitly; reviewers care about that detail.

### 2.5 Hierarchical EDA Reward (`c910_reward.py:compute_score`)

Three sequential gates, each with smooth partial credit so the gradient is non-zero even on failure:

| Level | Tool | Pass condition | Reward (default weights) |
|---|---|---|---|
| **L1 Syntax** | `iverilog -g2005-sv` over `{candidate, reference_renamed, testbench, deps}` | exit 0 + `simv` produced | `w_syntax · r_syntax`, `w_syntax = 0.1`, partial credit `1/(1+n_errors)` (×0.3 for port errors) |
| **L2 Functional** | `vvp simv` (dual-instance testbench: candidate vs `_ref` golden) | testbench prints `"Your Design Passed"` | `w_func · r_func`, `w_func = 1.0`, partial = `passed/total` ratio |
| **L3 PPA** | `yosys` (sky130 liberty `dfflibmap` + `abc`) → `gate_netlist.v` → **OpenSTA** static timing | synthesizable + valid timing | `w_ppa · r_ppa`, `w_ppa = 10.0`, **`r_ppa = ADP_baseline / ADP_candidate`** |

Total: `r = w_s·r_s + w_f·r_f + w_p·r_p`. `ADP = area(μm²) × delay(ps)`. `ADP_baseline` is auto-computed at startup by synthesizing each problem's `reference.v` once.

The functional level is the key novelty for industrial RTL: the testbench instantiates **both** the candidate and the renamed reference (`_ref`) and checks **mathematical equivalence** of `sum+carry` (not bit-exactness on each), enabling the LLM to explore alternative compressor-tree topologies.

> **Figure note:** draw L1/L2/L3 as a vertical filter chain with the candidate flowing top-down; failures escape sideways with their partial reward. For L2, render the dual-instance comparison as two parallel boxes (candidate / `_ref`) feeding an `==` checker. For L3, show `yosys → netlist → OpenSTA → (area, delay) → ADP` as a small inset.

### 2.6 Entropic-β Advantage + Discover KL Penalty (`recipe/ttt_math_construct/algorithms.py`)

Per-group advantage estimator (`adv_estimator: ttt_entropic_adaptive`):

1. **Group rewards** `r_1, …, r_k` by parent state (`k = n = 8`).
2. **Drop constant-reward groups** (`ttt_remove_constant_reward_groups=True`) — they contribute zero gradient.
3. **Solve β by binary search** so the per-group Boltzmann weights match a target KL budget δ:
   $$q_i \;=\; \frac{\exp(\beta\,r_i)}{\sum_j \exp(\beta\,r_j)},\qquad \mathrm{KL}\!\left(q\,\|\,\mathrm{Uniform}_k\right) \;\approx\; \delta.$$
   `δ` is the **KL budget** (described in §2.7 — this is the user-facing knob). Higher δ → higher β → more peaked `q` → **exploitative**; lower δ → β→0 → near-uniform `q` → **exploratory**.
4. **Leave-one-out normalization** within the group:
   $$\hat A_i \;=\; \frac{\exp(\beta\,r_i)}{\frac{1}{k-1}\sum_{j\neq i}\exp(\beta\,r_j)} \;-\; 1.$$
5. **Discover-style per-token KL penalty toward π_ref** (activated by `use_kl_in_advantage=True`, requires a frozen reference-policy worker):
   $$\hat A_t \;\leftarrow\; \hat A_t \;+\; \kappa \cdot \big(\overline{\mathrm{KL}} - \mathrm{KL}_t\big),\qquad \mathrm{KL}_t = \log\pi_\theta(a_t) - \log\pi_{\mathrm{ref}}(a_t),$$
   with `κ = ttt_kl_penalty_coef = 0.1`. The redistribution is mean-zero across a rollout — it pulls high-divergence tokens toward π_ref without changing the rollout-level advantage.

Policy loss: **GPG** (`-log_prob × advantage`), `actor.policy_loss.loss_mode: gpg`.

Per-step diagnostics published to SwanLab (`adv/ttt/*`): `delta_used`, `beta_solved_{mean,max}`, `kl_achieved_mean`, `effective_group_size_mean = k·exp(−KL)`, `beta_max_hit_rate`, `constant_groups_frac`, `kl_penalty_avg_kl`.

> **Figure note:** advantage inset = 8 rollouts sorted by reward → softmax `q_β` (a peaked curve whose width is controlled by δ) → leave-one-out → `+κ·(KL̄−KL_t)` ribbon along the token axis. Frozen π_ref is a separate dashed box feeding the KL ribbon.

### 2.7 Adaptive KL Budget δ (the contribution) — `_update_adaptive_delta`

**Why adaptive?** The single hyperparameter δ controls the entire exploration/exploitation balance of TTT-Discover (it sets the per-group advantage temperature). A *fixed* δ = ln 2 (Discover's default) over- or under-shoots in different regimes:

- early in training: many constant-reward groups → fixed δ wastes a budget on flat advantages,
- mid-training when one rollout dominates: β saturates and the policy collapses onto a single trajectory,
- on plateaus: gradient signal stays equally peaked around stale optima.

The C910 recipe ships three modes; the **`adaptive`** mode is the one we recommend and report:

| `ttt_entropic_kl_budget_mode` | Behavior |
|---|---|
| `fixed` | constant `δ = ln 2` (Discover baseline) |
| `schedule` | linear or cosine interp `δ_start → δ_end` over `total_training_steps` |
| **`adaptive`** | metric-driven priority controller (below) |

#### 2.7.1 Signals (all EMAs, computed at end of step `t`, used at step `t+1` — natural one-step lag)

| EMA | Source | α |
|---|---|---|
| `kl_ref` | per-token `KL(π_θ ‖ π_ref)`, batch mean | 0.3 |
| `beta_max_rate` | fraction of groups where the binary-search β hit the upper bound `β_max=10⁶` (saturation) | 0.3 |
| `eff_n` | effective sample size per group `= k · exp(−KL_achieved)`, batch mean | 0.3 |
| `const_frac` | fraction of constant-reward groups in batch | 0.3 |
| `reward_mean` | batch-mean reward (logging only) | 0.1 |
| `best_seen` | running max of **batch-max reward** (not EMA — a hard "best-ever" with absolute-improvement threshold to avoid drift from a slowly-filling pool) | — |
| `steps_no_improve` | consecutive steps without a new `best_seen` | — |

#### 2.7.2 Priority ladder (first rule fires; defaults shown)

| Rule | Condition (EMA) | Action on δ | Reason code |
|---|---|---|---|
| **P1 KL Brake** | `kl_ref > 0.4` | `δ ← δ_init`, reset `steps_no_improve` | `1` |
| **P2 Winner-Take-All** | `beta_max_rate > 0.6` | `δ ← max(δ · 0.85, δ_min)` (down) | `2` |
| **P3 Stagnation** | `steps_no_improve ≥ 15` **AND** `const_frac < 0.7` | `δ ← max(δ · 0.85, δ_min)` (down) | `3` |
| **P4 Over-Exploring** | `eff_n > 5.5` **AND** `const_frac < 0.3` **AND** `steps_no_improve < 5` (plateau guard) | `δ ← min(δ · 1.2, δ_max)` (up) | `4` |
| **P0 Hold** | else | unchanged | `0` |

Bounds: `δ ∈ [0.25·ln 2, 4·ln 2] ≈ [0.173, 2.773]`. Step sizes: `up = 1.2`, `down = 0.85`.

#### 2.7.3 The four design subtleties (worth calling out in the paper)

1. **P1 reset, not decrement.** A KL-spike vs π_ref is a collapse signal, not a tuning signal — the controller jumps back to `δ_init` instead of slowly stepping down, to give the policy a wide-advantage step before drift compounds.
2. **P2 fires on β-saturation alone**, not jointly with low `eff_n` as the original draft did. Reason: `eff_n` is a batch *mean* and gets inflated by the non-saturated groups, so the joint condition essentially never fires. β hitting `β_max` already proves a single sample dominates within most groups.
3. **P3 uses batch-MAX reward, not batch-mean.** Batch-mean drifts upward as the PUCT pool fills with marginally-better states even when the policy is stuck — it would spuriously reset the stagnation counter. Tracking `max(batch_max_t)` over training ensures only a *genuinely new best* clears stagnation. The improvement threshold is `0.02 · max(|best_seen|, 1)` — absolute when `best_seen` is small to avoid relative-improvement degeneracy near zero.
4. **P4 plateau guard.** Late in training, `const_frac → 0` and `eff_n` inflates because most groups have small reward spreads — naively this triggers "over-exploring" and δ increases, which flattens the gradient and blocks breakthrough. The `steps_no_improve < 5` guard suppresses P4 during plateaus so P3 can drive δ *down* toward sharper exploitation instead.

Telemetry (logged to SwanLab as `adv/ttt/adaptive_*`): `delta_next`, `ema_{const_frac, eff_n, kl_ref, beta_max_rate, reward}`, `best_seen`, `steps_no_improve`, `reason_code` (numeric for plotting).

> **Figure note:** the adaptive controller deserves its own inset (top-right of the framework figure or a separate sub-figure). Two visualizations work well:
> 1. **Priority ladder graphic** — five horizontal "lanes" P1→P4→P0, each showing its EMA signal threshold and δ-action arrow (↓ / ↓ / ↑ / reset / —).
> 2. **δ trajectory plot** — a real training run's `delta_used` over steps, color-coded by the active rule (`reason_code`), overlaid with the `best_seen` reward curve. This makes the controller's behavior tangible and is also a natural ablation target ("fixed δ" vs "schedule" vs "adaptive").

### 2.8 State-Pool Update (closing the loop)

For each parent: insert top-k=2 surviving (= `success=True`) children carrying full lineage `(parent_ids, parent_values)`. Update PUCT statistics:
- `n(s) ← n(s) + n_rollouts`
- `T ← T + n_rollouts`
- `Q(s) ← Q(s)` updated by best child's reward (favors expanding promising regions).

> **Figure note:** the closing arrow from the reward block back to 𝒮 is the visual anchor of "test-time training". Label it `(s_t, c_i, r_i)` and annotate `top-k=2`, `lineage-block`, `dedup`.

---

## 3. Compute & Engineering Stack (for reviewer context)

- **Verl backend**: `verl/trainer/main_ppo.py`, FSDP actor + frozen FSDP reference.
- **Distributed**: 2 nodes × 8 × A800-80G via SLURM (`run_ttt_c910_multinode.slurm`) or 1 node interactive (`run_ttt_c910.sh`).
- **EDA toolchain**: `iverilog/vvp` (Icarus 11+) → `yosys` 0.34+ with sky130 (`pdk/sky130_fd_sc_hd__tt_025C_1v80.lib`) → **OpenSTA** at `sta` for accurate critical path.
- **Long-context training tricks**: `use_remove_padding`, `use_dynamic_bsz`, `entropy_from_logits_with_chunking`, `entropy_checkpointing` (all enabled in `ttt_c910_trainer.yaml`).
- **Logger**: SwanLab in `local` mode (compute nodes have no internet); a small `add_multi_chart` patch is shipped in `main_ttt_c910.py`.

> **Figure note (optional inset):** a small "infrastructure" stripe at the bottom of the figure listing `vLLM | FSDP | Ray | Yosys | OpenSTA | sky130 PDK` is helpful for systems-leaning reviewers.

---

## 4. Suggested Figure Layout

A single landscape figure, ~7" × 4", with three horizontal bands:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Band A (Targets)        Band B (Agent loop)              Band C        │
│  OpenC910 FPU            ┌──────────────┐                 (Reward)      │
│  ┌──────────┐            │ State Pool 𝒮 │                              │
│  │mult_comp │            │  PUCT tree   │ ─→ Prompt → πθ ─→ candidate ─→│ L1 │ L2 │ L3 │
│  │  (Wallace│            └──────┬───────┘                               │
│  │   tree)  │                   ▲                                       │   reward r
│  └──────────┘                   │                                       │
│  ┌──────────┐                   │           (advantage, KL)             │
│  │   lza    │                   ┕────────── Entropic Adv ←──────────────┤
│  └──────────┘                                ↑    ↑                     │
│  ┌──────────┐                                │    │                     │
│  │ fadd ... │                            π_ref (frozen)                 │
│  └──────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────┘
        Band D (compute stack stripe): vLLM | FSDP | Ray | Yosys | OpenSTA | sky130
```

Layout principles:
- **Left = problem definition** (industrial), **center = agent loop**, **right = environment** (EDA).
- Make the **EDA chain horizontal and labeled** (most papers gloss over this — concrete tool names build credibility).
- Use **two distinct arrow styles**: solid for forward data flow (prompt/candidate/reward), dashed for gradient/back-update (state-pool insert, policy gradient, KL).
- Color code by stage: blue for state-pool ops, orange for LLM, green for EDA, red for advantage/policy update. Stay color-blind safe.
- Include the **PUCT formula** and the **adv-update equation** as small math callouts pinned to their respective blocks; reviewers love being able to verify the exact algorithm at a glance.

---

## 5. Asset Checklist (for the figure illustrator)

- [ ] OpenC910 mult_compressor schematic snippet (Wallace tree, EX1→EX2) — derive from `problems/mult_compressor/spec.yaml` description.
- [ ] PUCT tree mini-graphic (root + ~3 levels, k=2 branching, lineage-block highlight).
- [ ] Two prompt cards (initial vs iterative).
- [ ] LLM block with thinking/code split arrows; annotate Qwen3-8B + YaRN 65k.
- [ ] Reward filter chain (L1 iverilog / L2 vvp dual-instance comparator / L3 yosys+OpenSTA → ADP).
- [ ] Advantage inset: 8 rollouts → β-softmax → leave-one-out → +κ·(KL̄−KL_t).
- [ ] Frozen π_ref block with dashed arrow into KL inset.
- [ ] Compute-stack stripe.

---

## 6. Notation Glossary (use consistently in figure and paper)

| Symbol | Meaning |
|---|---|
| `𝒮` | per-problem PUCT state pool |
| `s, s_t` | a state (parent / sampled at step `t`) |
| `c_i` | the `i`-th child rollout from a parent |
| `n(s), Q(s), T` | PUCT visit count, mean child value, global step counter |
| `δ` | **adaptive KL budget** for the entropic-β advantage; controlled per step by the priority ladder in §2.7. Range `[0.25·ln 2, 4·ln 2]`, init `ln 2`. |
| `β` | per-group inverse-temperature solved by binary search so `KL(q_β ‖ Uniform_k) ≈ δ` |
| `κ` | KL-penalty coefficient (`ttt_kl_penalty_coef = 0.1`) |
| `π_ref` | frozen reference policy (initial SFT checkpoint) |
| `ADP` | Area × Delay product on sky130; `r_ppa = ADP_ref / ADP_cand` |
