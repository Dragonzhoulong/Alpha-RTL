#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -J verl_ttt_rtl
#SBATCH -o verl_rtl_%j.out
#SBATCH -e verl_rtl_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#
# Multi-node launcher for TTT-RTL.
# Sets up a Ray cluster across SLURM-allocated nodes, then runs training.
#
# Usage:
#   MODEL_PATH=... RTL_PROBLEMS=fsm EXP_NAME=fsm_exp1 SAVE_FREQ=-1 \
#       sbatch recipe/ttt_rtl/run_ttt_rtl_multinode.slurm
#
# Accepts the same environment variables as run_ttt_rtl.sh.

set -euo pipefail

set +e  # tolerate missing module (env drift 2026-07); re-enabled after env init
source ~/.bashrc
conda activate "${CONDA_ENV:-verl}"
set -e

# ===================== Ray cluster setup =====================

n_gpus_per_node=${SLURM_GPUS_PER_NODE:-4}
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}

echo "=== SLURM multi-node Ray setup ==="
echo "Nodes: ${nodes[*]}"
echo "Head:  $head_node"
echo "GPUs/node: $n_gpus_per_node"

# Head node IP — sbatch script runs on head node, so use hostname directly
all_ips=$(hostname -I 2>/dev/null || hostname -i 2>/dev/null || true)
head_ip=$(echo "$all_ips" | tr ' ' '\n' | grep -vE '^127\.' | grep -E '^[0-9]+\.' | head -n1 || true)
if [ -z "$head_ip" ]; then
  echo "ERROR: cannot resolve head node IP (all_ips=$all_ips)"
  exit 1
fi
echo "Head IP: $head_ip"

# Clean up stale Ray on all nodes
# Head node: run directly
ray stop -f 2>/dev/null || true
pkill -9 -f 'raylet|gcs_server' 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true
# Worker nodes: use srun
for n in "${nodes[@]:1}"; do
  srun -N1 -n1 -w "$n" bash -c "
    ray stop -f 2>/dev/null || true
    pkill -9 -f 'raylet|gcs_server' 2>/dev/null || true
    rm -rf /tmp/ray/session_* 2>/dev/null || true
  " &
done
wait
echo "Stale Ray processes cleaned."

# Must unset before ray start so Ray daemons and all actors they spawn don't inherit them.
# (sbatch --export=ALL can carry these in from the calling shell environment.)
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true

# Start Ray head (runs locally on this node)
ray start --head --node-ip-address=$head_ip --port=6379 \
  --include-dashboard=False --num-gpus=$n_gpus_per_node
sleep 5
echo "Ray head started on $head_node."

# Start Ray workers
for n in "${nodes[@]:1}"; do
  srun -N1 -n1 -w "$n" bash -c "
    source ~/.bashrc && conda activate "${CONDA_ENV:-verl}"
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    ray start --address=$head_ip:6379 --num-gpus=$n_gpus_per_node --block
  " &
  sleep 5
  echo "Ray worker started on $n."
done

sleep 10

# ===================== Training =====================

# SLURM_SUBMIT_DIR is set by sbatch to the directory where sbatch was called.
# Using BASH_SOURCE[0] would resolve to the script's spool copy (/var/spool/slurm/...),
# which gives a wrong REPO_ROOT and causes mkdir to fail with permission denied.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3-8B"}
CONFIG_NAME=${CONFIG_NAME:-"ttt_rtl_trainer"}
EXP_NAME=${EXP_NAME:-"multi_pipe_4bit"}
PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-rtl"}
LOGGER=${LOGGER:-'["console","swanlab"]'}
NNODES=${#nodes[@]}
NGPUS_PER_NODE=$n_gpus_per_node
# Leave ROLLOUT_TP unset by default so the YAML config controls TP.
ROLLOUT_TP=${ROLLOUT_TP:-}

RTLLM_DIR=${RTLLM_DIR:-"RTLLM"}
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-50}
FAILURE_REWARD=${FAILURE_REWARD:-0.0}
REWARD_TIMEOUT_S=${REWARD_TIMEOUT_S:-10.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
LIBERTY_FILE=${LIBERTY_FILE:-"recipe/ttt_rtl/pdk/Nangate45_typ.lib"}
INCLUDE_POWER=${INCLUDE_POWER:-true}
SEED_SOLUTIONS=${SEED_SOLUTIONS:-}

export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${HOME}/swanlog"}

REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_ROOT}/recipe/ttt_rtl/rtl_reward.py"}

LOG_DIR="${REPO_ROOT}/outputs/ttt_rtl_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[info] Training log will be written to: ${LOG_FILE}"

export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=$head_ip:6379

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
if [[ -n "${SEED_SOLUTIONS:-}" ]]; then
  EXTRA_ARGS+=("data.ttt_rtl.seed_solutions_file=${SEED_SOLUTIONS}")
fi

python3 -m recipe.ttt_rtl.main_ttt_rtl --config-name "${CONFIG_NAME}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.reward_kwargs.failure_reward="${FAILURE_REWARD}" \
    custom_reward_function.reward_kwargs.timeout_s="${REWARD_TIMEOUT_S}" \
    custom_reward_function.reward_kwargs.liberty_file="${LIBERTY_FILE}" \
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
