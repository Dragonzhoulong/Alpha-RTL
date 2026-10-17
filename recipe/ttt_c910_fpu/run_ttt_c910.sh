#!/usr/bin/env bash
# Launch TTT-Discover for C910 FPU component optimization.
#
# Usage:
#   bash recipe/ttt_c910_fpu/run_ttt_c910.sh [hydra overrides...]
#
# Environment variables (with defaults):
#   MODEL_PATH          Path to base model (default: ~/models/Qwen3-8B)
#   EXP_NAME            Experiment name (default: mult_compressor)
#   PROJECT_NAME        Project name (default: verl-ttt-c910)
#   LOGGER              Logger config (default: '["console","swanlab"]')
#   C910_PROBLEMS       Comma-separated problem list (default: config value)
#   NNODES              Number of nodes (default: 1)
#   NGPUS_PER_NODE      GPUs per node (default: 8)
#   ROLLOUT_TP          Rollout tensor parallelism (default: config value)
#   ROLLOUT_TEMPERATURE Sampling temperature (default: 1.0)
#   ROLLOUT_TOP_P       Sampling top_p (default: 1.0)
#   REWARD_TIMEOUT_S    Per-tool timeout (default: 30.0)
#   SIM_TIMEOUT_S       Simulation timeout (default: 120.0)
#   FAILURE_REWARD      Reward for failures (default: 0.0)
#   LIBERTY_FILE        Liberty timing library (default: sky130 PDK)
#   OPENSTA_BINARY      OpenSTA binary path (default: repo setting)
#   SAVE_FREQ           Checkpoint frequency (default: 50)
#   RESUME_MODE         Resume mode (default: disable)
#   CONFIG_NAME         Hydra config name without .yaml (default: ttt_c910_trainer)
#   REWARD_FN_PATH      Reward function path (default: recipe/ttt_c910_fpu/c910_reward.py)
#   SWANLAB_MODE        SwanLab mode: cloud or local (default: local)
#   SWANLAB_LOG_DIR     SwanLab log directory (default: ~/swanlog)
#   KL_BUDGET_MODE      Entropic δ mode: fixed | schedule | adaptive (default: config value)
#   KL_BUDGET_START     δ at step 0   when mode=schedule (default: config value)
#   KL_BUDGET_END       δ at final step when mode=schedule (default: config value)
#   KL_BUDGET_SCHEDULE  cosine | linear (default: config value)
#   KL_BUDGET_ADAPTIVE_MIN   δ lower bound for adaptive mode (default: 0.1733)
#   KL_BUDGET_ADAPTIVE_MAX   δ upper bound for adaptive mode (default: 2.7726)
#   KL_ADAPTIVE_EMA_ALPHA    EMA weight on new obs (default: 0.3)
#   KL_ADAPTIVE_KL_BRAKE     ref-KL EMA threshold for emergency reset (default: 0.4)
#   KL_ADAPTIVE_WINNER_BETA  β-max-hit-rate threshold for winner-take-all (default: 0.6)
#   KL_ADAPTIVE_WINNER_EFF_N eff-n threshold for winner-take-all (default: 2.5)
#   KL_ADAPTIVE_WARMUP       hold δ at init for first N steps (default: 20)
#   KL_ADAPTIVE_STAG_STEPS   rolling-window length for stagnation (default: 15)
#   KL_ADAPTIVE_STAG_EFF_N   only fire P3 if eff_n EMA < this (default: 4.0)
#   KL_ADAPTIVE_PLATEAU_GUARD P4 guard: only over-explore if steps_no_improve<N (default: 5)
#   KL_ADAPTIVE_UP_FACTOR    δ multiplicative increase factor (default: 1.2)
#   KL_ADAPTIVE_DOWN_FACTOR  δ multiplicative decrease factor (default: 0.85)
#   REUSE_MODE          State-reuse mode: puct | epsilon_greedy | none (default: config value)
#   EPS_GREEDY_EPS      ε for reuse_mode=epsilon_greedy (default: config value)
#   ADV_ESTIMATOR       Advantage estimator: ttt_entropic_adaptive | grpo | ... (default: config value)
#   USE_KL_IN_ADV       Enable Discover-style KL penalty in advantage (default: config value)
#   ACTOR_LR            Actor optimizer learning rate (default: config value)

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3-8B-65k"}
CONFIG_NAME=${CONFIG_NAME:-"ttt_c910_trainer"}
EXP_NAME=${EXP_NAME:-"mult_compressor"}
PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-c910"}
LOGGER=${LOGGER:-'["console","swanlab"]'}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Leave ROLLOUT_TP unset by default so the YAML config controls TP.
ROLLOUT_TP=${ROLLOUT_TP:-}

RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-50}
FAILURE_REWARD=${FAILURE_REWARD:-0.0}
REWARD_TIMEOUT_S=${REWARD_TIMEOUT_S:-30.0}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-120.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
LIBERTY_FILE=${LIBERTY_FILE:-"recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib"}
OPENSTA_BINARY=${OPENSTA_BINARY:-"sta"}

# SwanLab configuration
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${HOME}/swanlog"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_ROOT}/recipe/ttt_c910_fpu/c910_reward.py"}

cd "${REPO_ROOT}"

# Log file setup
LOG_DIR="${REPO_ROOT}/outputs/ttt_c910_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[info] Training log will be written to: ${LOG_FILE}"

# Clean environment
unset RAY_ADDRESS || true
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
export HYDRA_FULL_ERROR=1

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[warn] Unsetting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for vLLM compatibility."
  unset PYTORCH_CUDA_ALLOC_CONF || true
fi

EXTRA_ARGS=()
if [[ -n "${C910_PROBLEMS:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_c910.problems=${C910_PROBLEMS}")
fi
# Only override TP when explicitly set; otherwise the YAML config value is used.
if [[ -n "${ROLLOUT_TP:-}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}")
fi
# KL-budget overrides (only emitted when the env var is set).
# mode: fixed | schedule | adaptive
if [[ -n "${KL_BUDGET_MODE:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_mode=${KL_BUDGET_MODE}")
fi
# schedule-mode params
if [[ -n "${KL_BUDGET_START:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_start=${KL_BUDGET_START}")
fi
if [[ -n "${KL_BUDGET_END:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_end=${KL_BUDGET_END}")
fi
if [[ -n "${KL_BUDGET_SCHEDULE:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_schedule=${KL_BUDGET_SCHEDULE}")
fi
# adaptive-mode params
if [[ -n "${KL_BUDGET_ADAPTIVE_MIN:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_adaptive_min=${KL_BUDGET_ADAPTIVE_MIN}")
fi
if [[ -n "${KL_BUDGET_ADAPTIVE_MAX:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_entropic_kl_budget_adaptive_max=${KL_BUDGET_ADAPTIVE_MAX}")
fi
if [[ -n "${KL_ADAPTIVE_EMA_ALPHA:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_ema_alpha=${KL_ADAPTIVE_EMA_ALPHA}")
fi
if [[ -n "${KL_ADAPTIVE_KL_BRAKE:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_kl_brake_threshold=${KL_ADAPTIVE_KL_BRAKE}")
fi
if [[ -n "${KL_ADAPTIVE_WINNER_BETA:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_winner_beta_rate_hi=${KL_ADAPTIVE_WINNER_BETA}")
fi
if [[ -n "${KL_ADAPTIVE_WINNER_EFF_N:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_winner_eff_n_lo=${KL_ADAPTIVE_WINNER_EFF_N}")
fi
if [[ -n "${KL_ADAPTIVE_WARMUP:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_warmup_steps=${KL_ADAPTIVE_WARMUP}")
fi
if [[ -n "${KL_ADAPTIVE_STAG_STEPS:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_stagnation_steps=${KL_ADAPTIVE_STAG_STEPS}")
fi
if [[ -n "${KL_ADAPTIVE_STAG_EFF_N:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_stagnation_eff_n_lo=${KL_ADAPTIVE_STAG_EFF_N}")
fi
if [[ -n "${KL_ADAPTIVE_PLATEAU_GUARD:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_plateau_guard=${KL_ADAPTIVE_PLATEAU_GUARD}")
fi
if [[ -n "${KL_ADAPTIVE_UP_FACTOR:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_up_factor=${KL_ADAPTIVE_UP_FACTOR}")
fi
if [[ -n "${KL_ADAPTIVE_DOWN_FACTOR:-}" ]]; then
  EXTRA_ARGS+=("algorithm.ttt_adaptive_down_factor=${KL_ADAPTIVE_DOWN_FACTOR}")
fi
# Ablation knobs (only emitted when the env var is set).
if [[ -n "${REUSE_MODE:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_c910.reuse_mode=${REUSE_MODE}")
fi
if [[ -n "${EPS_GREEDY_EPS:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_c910.epsilon_greedy_eps=${EPS_GREEDY_EPS}")
fi
if [[ -n "${ADV_ESTIMATOR:-}" ]]; then
  EXTRA_ARGS+=("algorithm.adv_estimator=${ADV_ESTIMATOR}")
fi
if [[ -n "${USE_KL_IN_ADV:-}" ]]; then
  EXTRA_ARGS+=("algorithm.use_kl_in_advantage=${USE_KL_IN_ADV}")
fi
if [[ -n "${ACTOR_LR:-}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.optim.lr=${ACTOR_LR}")
fi

python3 -m recipe.ttt_c910_fpu.main_ttt_c910 --config-name "${CONFIG_NAME}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.reward_kwargs.failure_reward="${FAILURE_REWARD}" \
    custom_reward_function.reward_kwargs.timeout_s="${REWARD_TIMEOUT_S}" \
    custom_reward_function.reward_kwargs.sim_timeout_s="${SIM_TIMEOUT_S}" \
    custom_reward_function.reward_kwargs.liberty_file="${LIBERTY_FILE}" \
    custom_reward_function.reward_kwargs.opensta_binary="${OPENSTA_BINARY}" \
    data.ttt_c910.liberty_file="${LIBERTY_FILE}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    "${EXTRA_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
