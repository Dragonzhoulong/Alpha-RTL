#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -J ttt_c910_sft
#SBATCH -o outputs/ttt_c910_logs/sft_%j.out
#SBATCH -e outputs/ttt_c910_logs/sft_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#
# 2-node × 8-GPU SFT warm-up for TTT-C910-FPU.
# Uses torchrun c10d rendezvous (no Ray needed for SFT).
#
# Usage:
#   sbatch recipe/ttt_c910_fpu/sft/run_sft_multinode.slurm
#   EXP_NAME=my_run sbatch recipe/ttt_c910_fpu/sft/run_sft_multinode.slurm

set -euo pipefail

source ~/.bashrc
conda activate "${CONDA_ENV:-verl}"

# ===================== Node discovery =====================
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
NNODES=${#nodes[@]}

all_ips=$(srun -N1 -n1 -w "$head_node" hostname -I 2>/dev/null || true)
head_ip=$(echo "$all_ips" | tr ' ' '\n' | grep -vE '^127\.' | grep -E '^[0-9]+\.' | head -n1)
if [ -z "$head_ip" ]; then
  echo "ERROR: cannot resolve head node IP"
  exit 1
fi
echo "=== SFT multi-node setup ==="
echo "Nodes:    ${nodes[*]}"
echo "Head IP:  $head_ip"
echo "Num nodes: $NNODES"

# ===================== Config =====================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-"recipe/ttt_c910_fpu/sft/config.yaml"}
EXP_NAME=${EXP_NAME:-"mult_compressor_warmup"}
MASTER_PORT=${MASTER_PORT:-12355}

export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${HOME}/swanlog"}
export HYDRA_FULL_ERROR=1

LOG_DIR="${REPO_ROOT}/outputs/ttt_c910_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sft_${SLURM_JOB_ID}.log"
echo "[info] Log: ${LOG_FILE}"

# ===================== Launch =====================
# srun launches one task per node; each task runs torchrun which spawns 8
# GPU worker processes. c10d rendezvous coordinates across nodes.
srun --ntasks="${NNODES}" --ntasks-per-node=1 \
  python -m recipe.ttt_c910_fpu.sft.train_sft \
    --config "${CONFIG}" \
    --nnodes "${NNODES}" \
    --master-addr "${head_ip}" \
    --master-port "${MASTER_PORT}" \
  2>&1 | tee "${LOG_FILE}"
