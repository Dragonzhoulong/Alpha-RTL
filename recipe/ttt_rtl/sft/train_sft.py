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
"""Launch verl FSDP SFT trainer for TTT-RTL warm-up.

Invokes verl's built-in FSDPSFTTrainer via torchrun with appropriate
Hydra overrides for multi-turn SFT on collected Verilog data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


def build_torchrun_command(config: dict, n_gpus: int | None = None) -> list[str]:
    """Build the torchrun command with Hydra overrides."""
    train_cfg = config["training"]
    paths = config["paths"]

    n_gpus = n_gpus or train_cfg.get("n_gpus", 8)
    train_file = os.path.abspath(os.path.expanduser(paths["train_file"]))
    val_file = os.path.abspath(os.path.expanduser(paths["val_file"]))
    model_path = os.path.abspath(os.path.expanduser(train_cfg["model_path"]))
    checkpoint_dir = os.path.abspath(os.path.expanduser(train_cfg["checkpoint_dir"]))

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        "-m", "verl.trainer.fsdp_sft_trainer",
    ]

    # Hydra overrides
    overrides = [
        # Data configuration
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        f"data.max_length={train_cfg.get('max_length', 8192)}",
        "data.truncation=right",
        f"data.train_batch_size={train_cfg.get('train_batch_size', 64)}",
        f"data.micro_batch_size_per_gpu={train_cfg.get('micro_batch_size_per_gpu', 2)}",
        "data.balance_dp_token=True",

        # Model configuration
        f"model.partial_pretrain={model_path}",
        f"model.strategy={train_cfg.get('strategy', 'fsdp2')}",
        "model.enable_gradient_checkpointing=True",
        "model.trust_remote_code=True",

        # Optimizer
        f"optim.lr={train_cfg.get('lr', 1e-5)}",
        f"optim.warmup_steps_ratio={train_cfg.get('warmup_steps_ratio', 0.1)}",
        "optim.lr_scheduler=cosine",

        # Trainer
        f"trainer.total_epochs={train_cfg.get('total_epochs', 3)}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.project_name={train_cfg.get('project_name', 'ttt-rtl-sft')}",
        f"trainer.experiment_name={train_cfg.get('experiment_name', 'warmup')}",
        "trainer.save_freq=50",
        "trainer.test_freq=50",
        "trainer.logger=[console,swanlab]",
        # Save only HF format (safetensors, rank-0 only write) to avoid
        # torch.save zip-writer failures on Lustre for large concurrent writes
        "trainer.checkpoint.save_contents=[hf_model]",
    ]

    cmd.extend(overrides)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch SFT training for TTT-RTL warm-up")
    parser.add_argument("--config", default="recipe/ttt_rtl/sft/config.yaml", help="Config file path")
    parser.add_argument("--n-gpus", type=int, default=None, help="Number of GPUs (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Verify data files exist
    train_file = os.path.expanduser(config["paths"]["train_file"])
    val_file = os.path.expanduser(config["paths"]["val_file"])
    if not os.path.isfile(train_file):
        print(f"Error: Training data not found: {train_file}")
        print("Run format_sft_data.py first.")
        sys.exit(1)
    if not os.path.isfile(val_file):
        print(f"Error: Validation data not found: {val_file}")
        print("Run format_sft_data.py first.")
        sys.exit(1)

    cmd = build_torchrun_command(config, n_gpus=args.n_gpus)

    print("Command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Use local mode on compute nodes (no internet access); logs saved to swanlog/
    os.environ.setdefault("SWANLAB_MODE", "local")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nTraining complete.")
    else:
        print(f"\nTraining failed with exit code {result.returncode}")

    sys.exit(result.returncode)



if __name__ == "__main__":
    main()
