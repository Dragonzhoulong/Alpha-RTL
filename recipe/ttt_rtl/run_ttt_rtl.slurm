#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -J verl_ttt_rtl
#SBATCH -o verl_rtl_%j.out
#SBATCH -e verl_rtl_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G

source ~/.bashrc
conda activate "${CONDA_ENV:-verl}"

# --- 1) 清理环境污染（强烈建议） ---
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# 让 Ray 不去改 CUDA_VISIBLE_DEVICES（避免二次分配）
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# 每节点 GPU 数：--gres=gpu:4 时用 SLURM_GPUS_PER_NODE，否则回退到 4
n_gpus_per_node=${SLURM_GPUS_PER_NODE:-4}

# --- 2) 列出本次作业分到的节点 ---
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}

echo "Allocated nodes: ${nodes[@]}"
echo "Head node: $head_node"
echo "GPUs per node: $n_gpus_per_node"

# --- 3) head 的 IP：优先内网 IPv4（参考 ray_on_slurm.slurm）---
all_ips=$(srun -N 1 -n 1 -w "$head_node" hostname -I 2>/dev/null || srun -N 1 -n 1 -w "$head_node" hostname -i 2>/dev/null)
head_ip=$(echo "$all_ips" | grep -oE '\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3}\b|\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b' | head -n 1)
[ -z "$head_ip" ] && head_ip=$(echo "$all_ips" | grep -oE '\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b' | grep -v '^127\.' | head -n 1)
if [ -z "$head_ip" ]; then
  echo "ERROR: 无法获取 head 节点有效 IPv4，all_ips=$all_ips"
  exit 1
fi
echo "Head IP: $head_ip"

# --- 4) 每个节点彻底停掉残留 Ray（参考你跑通的单机流程）---
for n in "${nodes[@]}"; do
  srun -N 1 -n 1 -w "$n" bash -c "
    ray stop -f || true
    pkill -9 -f 'raylet|gcs_server|dashboard_agent|runtime_env_agent' 2>/dev/null || true
    rm -rf /tmp/ray/session_* 2>/dev/null || true
    rm -f ~/.ray/ray_current_cluster ~/.ray/current_cluster ~/.ray/ray_address 2>/dev/null || true
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  " &
done
wait

# --- 5) 启动 Ray head ---
srun -N 1 -n 1 -w "$head_node" bash -c "
  source ~/.bashrc
  conda activate "${CONDA_ENV:-verl}"
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  ray start --head --node-ip-address=$head_ip --port=6379 \
    --include-dashboard=False --num-gpus=$n_gpus_per_node --block
" &
sleep 10

# --- 6) 启动其他节点 Ray worker ---
for n in "${nodes[@]:1}"; do
  srun -N 1 -n 1 -w "$n" bash -c "
    source ~/.bashrc
    conda activate "${CONDA_ENV:-verl}"
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    ray start --address=$head_ip:6379 --num-gpus=$n_gpus_per_node --block
  " &
  sleep 5
done

sleep 10

# --- 7) 在 head 节点启动 verl driver（直接运行，不用 srun，脚本已在 head 上）---
export RAY_ADDRESS=$head_ip:6379
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# TTT-RTL environment variables
export MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen2.5-7B-Instruct"}
export EXP_NAME=${EXP_NAME:-"ttt-rtl-discover"}
export PROJECT_NAME=${PROJECT_NAME:-"verl-ttt-rtl"}
export LOGGER=${LOGGER:-'["console","swanlab"]'}
export NNODES=2
export NGPUS_PER_NODE=4
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export RTLLM_DIR=${RTLLM_DIR:-"RTLLM"}

python3 -m recipe.ttt_rtl.main_ttt_rtl \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.reward_kwargs.failure_reward="${FAILURE_REWARD}" \
    custom_reward_function.reward_kwargs.timeout_s="${REWARD_TIMEOUT_S}" \
    data.ttt_rtl.rtllm_dir="${RTLLM_DIR}" \
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
    "${EXTRA_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
