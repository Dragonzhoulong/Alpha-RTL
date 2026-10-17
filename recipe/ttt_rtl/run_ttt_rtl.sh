#!/usr/bin/env bash
# Launch TTT-Discover for RTL generation.
#
# Usage:
#   bash recipe/ttt_rtl/run_ttt_rtl.sh [hydra overrides...]
#
# Environment variables (with defaults):
#   MODEL_PATH          Path to base model (default: ~/models/Qwen3-8B)
#   EXP_NAME            Experiment name (default: ttt-rtl-discover)
#   PROJECT_NAME        Project name (default: verl-ttt-rtl)
#   LOGGER              Logger config (default: '["console","swanlab"]')
#   NNODES              Number of nodes (default: 1)
#   NGPUS_PER_NODE      GPUs per node (default: 8)
#   ROLLOUT_TP          Rollout tensor parallelism (default: 1)
#   RTLLM_DIR           Path to RTLLM benchmark (default: RTLLM)
#   RTL_PROBLEMS        Comma-separated problem list (default: config value)
#   REWARD_TIMEOUT_S    Per-tool timeout (default: 30.0)
#   FAILURE_REWARD      Reward for failures (default: 0.0)
#   ROLLOUT_TEMPERATURE Sampling temperature (default: 1.0)
#   ROLLOUT_TOP_P       Sampling top_p (default: 1.0)
#   LIBERTY_FILE        Liberty timing library for tech-mapped synthesis (default: NanGate 45nm)
#   INCLUDE_POWER       Fold OpenSTA power into PPA (default: true => A*D*P reward)
#   CONFIG_NAME         Hydra config name without .yaml (default: ttt_rtl_trainer)
#                       Use ttt_rtl_fp_trainer for FloatPoint problems.
#   SWANLAB_API_KEY     SwanLab API key (optional, for cloud mode)
#   SWANLAB_MODE        SwanLab mode: cloud or local (default: cloud)
#   SWANLAB_LOG_DIR     SwanLab log directory (default: swanlog)

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3-8B"}
CONFIG_NAME=${CONFIG_NAME:-"ttt_rtl_trainer"}
EXP_NAME=${EXP_NAME:-"multi_pipe_4bit"}
PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-rtl"}
LOGGER=${LOGGER:-'["console","swanlab"]'}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Leave ROLLOUT_TP unset by default so the YAML config controls TP.
# Set explicitly (e.g. ROLLOUT_TP=2) only to override the config value.
ROLLOUT_TP=${ROLLOUT_TP:-}

RTLLM_DIR=${RTLLM_DIR:-"RTLLM"}
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-50}
FAILURE_REWARD=${FAILURE_REWARD:-0.0}
REWARD_TIMEOUT_S=${REWARD_TIMEOUT_S:-10.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
LIBERTY_FILE=${LIBERTY_FILE:-"recipe/ttt_rtl/pdk/Nangate45_typ.lib"}
OPENSTA_BINARY=${OPENSTA_BINARY:-"sta"}
INCLUDE_POWER=${INCLUDE_POWER:-true}

# SwanLab configuration
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${HOME}/swanlog"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_ROOT}/recipe/ttt_rtl/rtl_reward.py"}

cd "${REPO_ROOT}"

# Log file setup
LOG_DIR="${REPO_ROOT}/outputs/ttt_rtl_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[info] Training log will be written to: ${LOG_FILE}"

# Avoid inheriting stale external Ray endpoints from login environment.
unset RAY_ADDRESS || true
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
export HYDRA_FULL_ERROR=1

# vLLM's CuMemAllocator is not compatible with expandable_segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[warn] Unsetting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for vLLM compatibility."
  unset PYTORCH_CUDA_ALLOC_CONF || true
fi

EXTRA_ARGS=()
if [[ -n "${RTL_PROBLEMS:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_rtl.problems=${RTL_PROBLEMS}")
fi
# Only override TP when explicitly set; otherwise the YAML config value is used.
if [[ -n "${ROLLOUT_TP:-}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}")
fi

python3 -m recipe.ttt_rtl.main_ttt_rtl --config-name "${CONFIG_NAME}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.reward_kwargs.failure_reward="${FAILURE_REWARD}" \
    custom_reward_function.reward_kwargs.timeout_s="${REWARD_TIMEOUT_S}" \
    custom_reward_function.reward_kwargs.liberty_file="${LIBERTY_FILE}" \
    custom_reward_function.reward_kwargs.opensta_binary="${OPENSTA_BINARY}" \
    custom_reward_function.reward_kwargs.include_power="${INCLUDE_POWER}" \
    data.ttt_rtl.liberty_file="${LIBERTY_FILE}" \
    data.ttt_rtl.include_power="${INCLUDE_POWER}" \
    data.ttt_rtl.rtllm_dir="${RTLLM_DIR}" \
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
