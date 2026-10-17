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
"""Merge CoDeV-R1 SFT data with existing filtered RTLLM data into train/val parquet."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def load_codev_records(jsonl_path: str) -> list[dict]:
    """Load CoDeV-R1 SFT JSONL and convert to messages format.

    CoDeV-R1 format: {"system": "...", "prompt": "...", "response": "<think>...</think> <answer>...</answer>"}
    Target format:   [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    Tag conversion: <think> -> <thinking>, </think> -> </thinking>
                    <answer>...</answer> tags are stripped (keep inner content).
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = item.get("response", "")
            # Convert <think>/<answer> tags to <thinking> style
            response = response.replace("<think>", "<thinking>").replace("</think>", "</thinking>")
            response = re.sub(r"</?answer>", "", response).strip()

            messages = []
            if item.get("system"):
                messages.append({"role": "system", "content": item["system"]})
            messages.append({"role": "user", "content": item["prompt"]})
            messages.append({"role": "assistant", "content": response})

            records.append({"messages": messages, "source": "codev-r1"})
    return records


def load_rtllm_records(jsonl_path: str) -> list[dict]:
    """Load existing filtered RTLLM JSONL and convert to messages format."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages_sent = item.get("messages_sent", [])
            response_content = item.get("response_content", "")

            messages = [{"role": m["role"], "content": m["content"]} for m in messages_sent]
            messages.append({"role": "assistant", "content": response_content})

            records.append({
                "messages": messages,
                "source": "rtllm",
                "problem_id": item.get("problem_id", ""),
            })
    return records


def merge_and_split(
    codev_records: list[dict],
    rtllm_records: list[dict],
    val_ratio: float = 0.05,
    codev_sample: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge two data sources and create train/val split.

    For RTLLM data: stratified split by problem_id (no problem leaks into both sets).
    For CoDeV-R1 data: random split (no problem_id available).
    """
    rng = random.Random(seed)

    # Optionally subsample CoDeV-R1 data
    if codev_sample is not None and codev_sample < len(codev_records):
        rng.shuffle(codev_records)
        codev_records = codev_records[:codev_sample]

    # Split RTLLM by problem_id (stratified)
    rtllm_train, rtllm_val = [], []
    if rtllm_records:
        problem_ids = sorted({r["problem_id"] for r in rtllm_records})
        rng.shuffle(problem_ids)
        n_val = max(1, int(len(problem_ids) * val_ratio))
        val_pids = set(problem_ids[:n_val])
        rtllm_train = [r for r in rtllm_records if r["problem_id"] not in val_pids]
        rtllm_val = [r for r in rtllm_records if r["problem_id"] in val_pids]

    # Split CoDeV-R1 randomly
    rng.shuffle(codev_records)
    n_codev_val = max(1, int(len(codev_records) * val_ratio))
    codev_train = codev_records[n_codev_val:]
    codev_val = codev_records[:n_codev_val]

    # Merge and shuffle
    train_all = rtllm_train + codev_train
    val_all = rtllm_val + codev_val
    rng.shuffle(train_all)
    rng.shuffle(val_all)

    train_df = pd.DataFrame({"messages": [r["messages"] for r in train_all]})
    val_df = pd.DataFrame({"messages": [r["messages"] for r in val_all]})

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Merge CoDeV-R1 + RTLLM SFT data")
    parser.add_argument(
        "--codev-jsonl",
        default="recipe/ttt_rtl/sft/data/codev_r1_sft.jsonl",
        help="Path to CoDeV-R1 SFT JSONL file",
    )
    parser.add_argument(
        "--rtllm-jsonl",
        default="recipe/ttt_rtl/sft/data/filtered.jsonl",
        help="Path to existing filtered RTLLM JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="recipe/ttt_rtl/sft/data",
        help="Output directory for train.parquet and val.parquet",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction held out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--codev-sample",
        type=int,
        default=None,
        help="Subsample N records from CoDeV-R1 for balance (default: use all 87K)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load both sources
    print(f"Loading CoDeV-R1 data from {args.codev_jsonl} ...")
    codev_records = load_codev_records(args.codev_jsonl)
    print(f"  Loaded {len(codev_records)} CoDeV-R1 records")

    print(f"Loading RTLLM data from {args.rtllm_jsonl} ...")
    rtllm_records = load_rtllm_records(args.rtllm_jsonl)
    print(f"  Loaded {len(rtllm_records)} RTLLM records")

    if not codev_records and not rtllm_records:
        print("ERROR: No data loaded from either source.")
        return

    # Merge and split
    train_df, val_df = merge_and_split(
        codev_records,
        rtllm_records,
        val_ratio=args.val_ratio,
        codev_sample=args.codev_sample,
        seed=args.seed,
    )

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.parquet"
    val_path = out / "val.parquet"

    # Backup existing files
    for p in [train_path, val_path]:
        if p.exists():
            bak = p.with_suffix(".parquet.bak")
            p.rename(bak)
            print(f"  Backed up {p} -> {bak}")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved {train_path} ({len(train_df)} rows)")
    print(f"Saved {val_path} ({len(val_df)} rows)")

    # Verify
    for path in [train_path, val_path]:
        df = pd.read_parquet(path)
        assert "messages" in df.columns
        first = df.iloc[0]["messages"]
        if not isinstance(first, list):
            first = list(first)
        roles = [m["role"] for m in first]
        assert "assistant" in roles, f"No assistant turn in {path}"
        print(f"  Verified {path}: {len(df)} rows, first entry has {len(first)} messages")


if __name__ == "__main__":
    main()
