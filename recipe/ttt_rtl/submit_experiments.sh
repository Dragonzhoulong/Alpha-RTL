#!/bin/bash
# Submit multiple TTT-RTL experiments as independent SLURM jobs.
#
# Usage:
#   bash recipe/ttt_rtl/submit_experiments.sh [experiments.conf]
#
# If no config file is given, the built-in EXPERIMENTS list below is used.
#
# Config file format (one experiment per line, fields separated by |):
#   PROBLEMS | EXP_NAME | MODEL_PATH | extra_overrides...
#
# Fields beyond EXP_NAME are optional — MODEL_PATH and extras fall back to
# the defaults already in run_ttt_rtl_multinode.slurm.
#
# Examples (in the config file or the EXPERIMENTS array below):
#   fsm | fsm_exp2 | checkpoints/ttt-rtl-sft/global_step_18/huggingface
#   counter | counter_exp1 | checkpoints/ttt-rtl-sft/global_step_18/huggingface
#   alu | alu_exp1
#   fsm,counter | fsm_counter_joint
#
# Extra SLURM or env overrides can be appended after MODEL_PATH:
#   fsm | fsm_long | /abs/path/to/model | SAVE_FREQ=100 ROLLOUT_TEMPERATURE=0.8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_ttt_rtl_multinode.slurm"

# ── Default experiments (edit or pass a config file instead) ─────────────────
DEFAULT_MODEL="checkpoints/ttt-rtl-sft/global_step_18/huggingface"
DEFAULT_PROJECT_NAME="${PROJECT_NAME:-verl-ttt-rtl-2026q2}"

declare -a EXPERIMENTS=(
  # PROBLEMS         | EXP_NAME
  "fsm               | fsm_exp2"
  "counter           | counter_exp1"
  "alu               | alu_exp1"
)
# ─────────────────────────────────────────────────────────────────────────────

if [[ $# -ge 1 && -f "$1" ]]; then
  mapfile -t EXPERIMENTS < <(grep -v '^\s*#' "$1" | grep -v '^\s*$')
  echo "[submit] Using config: $1 (${#EXPERIMENTS[@]} experiments)"
else
  echo "[submit] Using built-in EXPERIMENTS list (${#EXPERIMENTS[@]} experiments)"
fi

submitted=0

for entry in "${EXPERIMENTS[@]}"; do
  # Split on '|', strip whitespace
  IFS='|' read -ra parts <<< "$entry"

  problems=$(echo "${parts[0]:-}" | xargs)
  expname=$(echo  "${parts[1]:-}" | xargs)
  model=$(echo    "${parts[2]:-${DEFAULT_MODEL}}" | xargs)
  extras=$(echo   "${parts[3]:-}" | xargs)   # e.g. "SAVE_FREQ=100 ROLLOUT_TEMPERATURE=0.8"

  if [[ -z "$expname" ]]; then
    echo "[skip] Empty EXP_NAME in entry: '$entry'"
    continue
  fi

  # Build env string for sbatch --export
  env_vars="ALL,MODEL_PATH=${model},SAVE_FREQ=-1,PROJECT_NAME=${DEFAULT_PROJECT_NAME}"
  [[ -n "$problems" ]] && env_vars+=",RTL_PROBLEMS=${problems}"
  [[ -n "$expname"  ]] && env_vars+=",EXP_NAME=${expname}"

  # Append any extra key=value pairs
  for kv in $extras; do
    env_vars+=",${kv}"
  done

  job_id=$(sbatch --export="${env_vars}" \
                  --job-name="ttt_${expname}" \
                  "${SLURM_SCRIPT}" | awk '{print $NF}')

  echo "[submit] Job ${job_id}  EXP=${expname}  PROBLEMS=${problems:-<all>}  MODEL=$(basename "$model")"
  (( ++submitted )) || true
done

echo ""
echo "[submit] Done. ${submitted} job(s) submitted."
echo "         Monitor with: squeue -u \$USER"
echo "         Cancel all:   scancel -u \$USER"
