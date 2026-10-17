#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -J verl_c910_lza
#SBATCH -o verl_c910_lza_%j.out
#SBATCH -e verl_c910_lza_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#
# Single-node (8 GPU) launcher for TTT-C910 lza_simd_half optimization.
#
# Usage:
#   sbatch recipe/ttt_c910_fpu/run_ttt_c910_lza_simd_half.slurm
#
#   # With overrides:
#   MODEL_PATH=~/models/Qwen3-8B SAVE_FREQ=100 \
#       sbatch recipe/ttt_c910_fpu/run_ttt_c910_lza_simd_half.slurm
#
# Accepts the same environment variables as run_ttt_c910.sh.

set -uo pipefail

source ~/.bashrc || true
conda activate "${CONDA_ENV:-verl}"
set -e

# ===================== Ray single-node setup =====================

n_gpus_per_node=${SLURM_GPUS_PER_NODE:-8}

ray stop -f 2>/dev/null || true
pkill -9 -f 'raylet|gcs_server' 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true

ray start --head --include-dashboard=False --num-gpus=$n_gpus_per_node
sleep 5
echo "Ray head started (single-node, $n_gpus_per_node GPUs)."

# ===================== Training =====================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3-8B"}
CONFIG_NAME=${CONFIG_NAME:-"ttt_c910_lza_simd_half"}
EXP_NAME=${EXP_NAME:-"lza_simd_half"}
PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-c910"}
LOGGER=${LOGGER:-'["console","swanlab"]'}
NNODES=1
NGPUS_PER_NODE=$n_gpus_per_node
ROLLOUT_TP=${ROLLOUT_TP:-}

RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-"-1"}
FAILURE_REWARD=${FAILURE_REWARD:-0.0}
REWARD_TIMEOUT_S=${REWARD_TIMEOUT_S:-30.0}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-30.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
LIBERTY_FILE=${LIBERTY_FILE:-"recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib"}
OPENSTA_BINARY=${OPENSTA_BINARY:-"sta"}

export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${HOME}/swanlog"}

# Triton needs libcuda.so for torch.compile kernels; add CUDA compat dir to linker path
export LIBRARY_PATH=/usr/local/cuda/compat:${LIBRARY_PATH:-}

REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_ROOT}/recipe/ttt_c910_fpu/c910_reward.py"}

LOG_DIR="${REPO_ROOT}/outputs/ttt_c910_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_lza_$(date +%Y%m%d_%H%M%S).log"
echo "[info] Training log will be written to: ${LOG_FILE}"

export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=127.0.0.1:6379

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[warn] Unsetting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for vLLM compatibility."
  unset PYTORCH_CUDA_ALLOC_CONF || true
fi

EXTRA_ARGS=()
if [[ -n "${C910_PROBLEMS:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_c910.problems=${C910_PROBLEMS}")
fi
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
