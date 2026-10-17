#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end SFT warm-up pipeline for TTT-RTL
#
# Usage:
#   export OPENROUTER_API_KEY="sk-or-..."
#   bash recipe/ttt_rtl/sft/run_pipeline.sh [--config CONFIG] [--n-gpus N]
#
# Steps:
#   1. Collect responses from strong models via OpenRouter
#   2. Filter through iverilog/yosys pipeline
#   3. Format as parquet for MultiTurnSFTDataset
#   4. Train with verl FSDP SFT trainer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
N_GPUS="${N_GPUS:-8}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --n-gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --skip-collect)
            SKIP_COLLECT=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate API key
if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -z "${SKIP_COLLECT:-}" ]]; then
    echo "Error: OPENROUTER_API_KEY environment variable not set"
    echo "Set it with: export OPENROUTER_API_KEY='sk-or-...'"
    exit 1
fi

echo "============================================"
echo "TTT-RTL SFT Warm-up Pipeline"
echo "============================================"
echo "Config: ${CONFIG}"
echo "GPUs:   ${N_GPUS}"
echo ""

# Step 1: Collect responses
if [[ -z "${SKIP_COLLECT:-}" ]]; then
    echo ">>> Step 1/4: Collecting responses from strong models..."
    python -m recipe.ttt_rtl.sft.collect_responses --config "${CONFIG}"
    echo "    Done."
    echo ""
else
    echo ">>> Step 1/4: Skipped (--skip-collect)"
    echo ""
fi

# Step 2: Filter responses
echo ">>> Step 2/4: Filtering through iverilog/yosys pipeline..."
python -m recipe.ttt_rtl.sft.filter_responses --config "${CONFIG}"
echo "    Done."
echo ""

# Step 3: Format for SFT
echo ">>> Step 3/4: Formatting data for MultiTurnSFTDataset..."
python -m recipe.ttt_rtl.sft.format_sft_data --config "${CONFIG}"
echo "    Done."
echo ""

# Step 4: Train
if [[ -z "${SKIP_TRAIN:-}" ]]; then
    echo ">>> Step 4/4: Launching SFT training..."
    python -m recipe.ttt_rtl.sft.train_sft --config "${CONFIG}" --n-gpus "${N_GPUS}"
    echo "    Done."
    echo ""
else
    echo ">>> Step 4/4: Skipped (--skip-train)"
    echo ""
fi

echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
echo "The SFT checkpoint is saved in the directory specified in config.yaml"
echo "(default: checkpoints/ttt-rtl-sft/)"
echo ""
echo "To use the checkpoint for RL training, set MODEL_PATH to the latest"
echo "global_step_* directory when running run_ttt_rtl.sh."
