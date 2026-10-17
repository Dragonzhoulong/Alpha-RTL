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
"""Launch verl FSDP SFT trainer for TTT-C910-FPU warm-up.

Invokes verl's built-in FSDPSFTTrainer via torchrun with Hydra overrides
for multi-turn SFT on the mult_compressor solution dataset.

Single-node (interactive):
    python -m recipe.ttt_c910_fpu.sft.train_sft
    python -m recipe.ttt_c910_fpu.sft.train_sft --dry-run

Multi-node (called by run_sft_multinode.slurm via srun, one task per node):
    python -m recipe.ttt_c910_fpu.sft.train_sft --nnodes 2 --master-addr <HEAD_IP>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


def build_torchrun_command(
    config: dict,
    n_gpus: int | None = None,
    nnodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 12355,
) -> list[str]:
    """Build the torchrun + fsdp_sft_trainer command with Hydra overrides."""
    train_cfg = config["training"]
    paths = config["paths"]

    n_gpus = n_gpus or int(train_cfg.get("n_gpus", 8))
    train_file = os.path.abspath(os.path.expanduser(paths["train_file"]))
    val_file = os.path.abspath(os.path.expanduser(paths["val_file"]))
    model_path = os.path.abspath(os.path.expanduser(train_cfg["model_path"]))
    checkpoint_dir = os.path.abspath(os.path.expanduser(train_cfg["checkpoint_dir"]))

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={n_gpus}",
    ]
    if nnodes > 1:
        # c10d rendezvous for multi-node: all nodes connect to the head's endpoint
        cmd += [
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
        ]
    cmd += ["-m", "verl.trainer.fsdp_sft_trainer"]

    overrides = [
        # Data
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        f"data.max_length={int(train_cfg.get('max_length', 32768))}",
        # Left-truncation: if a prompt is too long, cut from the left to keep
        # the interface spec and solution code at the end.
        "data.truncation=left",
        f"data.train_batch_size={int(train_cfg.get('train_batch_size', 16))}",
        f"data.micro_batch_size_per_gpu={int(train_cfg.get('micro_batch_size_per_gpu', 1))}",
        "data.balance_dp_token=True",

        # Model
        f"model.partial_pretrain={model_path}",
        f"model.strategy={train_cfg.get('strategy', 'fsdp2')}",
        "model.enable_gradient_checkpointing=True",
        "model.trust_remote_code=True",
        f"model.fsdp_config.model_dtype={train_cfg.get('model_dtype', 'bf16')}",

        # Optimizer
        f"optim.lr={train_cfg.get('lr', 5e-6)}",
        f"optim.warmup_steps_ratio={train_cfg.get('warmup_steps_ratio', 0.05)}",
        "optim.lr_scheduler=cosine",

        # Trainer
        f"trainer.total_epochs={int(train_cfg.get('total_epochs', 3))}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.project_name={train_cfg.get('project_name', 'ttt-c910-sft')}",
        f"trainer.experiment_name={train_cfg.get('experiment_name', 'mult_compressor_warmup')}",
        f"trainer.nnodes={nnodes}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.save_freq=50",
        "trainer.test_freq=50",
        "trainer.logger=[console,swanlab]",
        # Save only HF format to avoid Lustre zip-writer failures on large concurrent writes
        "trainer.checkpoint.save_contents=[hf_model]",
    ]

    cmd.extend(overrides)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch SFT training for TTT-C910-FPU warm-up")
    parser.add_argument("--config", default="recipe/ttt_c910_fpu/sft/config.yaml", help="Config YAML path")
    parser.add_argument("--n-gpus", type=int, default=None, help="GPUs per node (overrides config)")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes (>1 enables multi-node torchrun)")
    parser.add_argument("--master-addr", default="localhost", help="Head node IP for multi-node rdzv")
    parser.add_argument("--master-port", type=int, default=12355, help="Port for multi-node rdzv")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_file = os.path.expanduser(config["paths"]["train_file"])
    val_file = os.path.expanduser(config["paths"]["val_file"])
    if not os.path.isfile(train_file):
        print(f"Error: Training data not found: {train_file}")
        print("Run format_sft_data.py first:")
        print("  python -m recipe.ttt_c910_fpu.sft.format_sft_data")
        sys.exit(1)
    if not os.path.isfile(val_file):
        print(f"Error: Validation data not found: {val_file}")
        print("Run format_sft_data.py first.")
        sys.exit(1)

    cmd = build_torchrun_command(
        config,
        n_gpus=args.n_gpus,
        nnodes=args.nnodes,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )

    print("Command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run — not executing)")
        return

    # Compute nodes have no internet; use local SwanLab logging
    os.environ.setdefault("SWANLAB_MODE", "local")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
