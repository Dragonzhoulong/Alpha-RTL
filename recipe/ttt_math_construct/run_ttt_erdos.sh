#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen2.5-7B-Instruct"}
EXP_NAME=${EXP_NAME:-"ttt-erdos-mvp"}
PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-math-construct"}
LOGGER=${LOGGER:-'["console","swanlab"]'}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}

RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
FAILURE_REWARD=${FAILURE_REWARD:--1.0}
REWARD_TIMEOUT_S=${REWARD_TIMEOUT_S:-20.0}
REWARD_BUDGET_S=${REWARD_BUDGET_S:-8.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.2}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_ROOT}/recipe/ttt_math_construct/erdos_reward.py"}

cd "${REPO_ROOT}"

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
python3 -m recipe.ttt_math_construct.main_ttt_erdos \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.reward_kwargs.failure_reward="${FAILURE_REWARD}" \
    custom_reward_function.reward_kwargs.timeout_s="${REWARD_TIMEOUT_S}" \
    custom_reward_function.reward_kwargs.budget_s="${REWARD_BUDGET_S}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs=200 \
    "$@"
