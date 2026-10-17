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
"""Convert filtered responses to parquet format for MultiTurnSFTDataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import yaml


def load_filtered_records(filtered_file: str) -> list[dict]:
    """Load filtered JSONL records."""
    records = []
    with open(filtered_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def record_to_messages(record: dict, include_thinking: bool = True) -> list[dict[str, str]]:
    """Convert a filtered record to the messages format expected by MultiTurnSFTDataset.

    The messages_sent field already contains the system + user messages.
    The response_content is the assistant response (possibly with <thinking> tags).
    """
    messages_sent = record.get("messages_sent", [])
    response_content = record.get("response_content", "")

    if not include_thinking:
        # Strip <thinking>...</thinking> blocks from response
        import re
        response_content = re.sub(r"<thinking>.*?</thinking>\s*", "", response_content, flags=re.DOTALL).strip()

    messages = []
    for msg in messages_sent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "assistant", "content": response_content})

    return messages


def format_data(config: dict) -> None:
    """Convert filtered records to train/val parquet files."""
    filtered_file = config["paths"]["filtered_file"]
    train_file = config["paths"]["train_file"]
    val_file = config["paths"]["val_file"]
    fmt_cfg = config["formatting"]

    if not os.path.isfile(filtered_file):
        print(f"Filtered file not found: {filtered_file}")
        print("Run filter_responses.py first.")
        return

    records = load_filtered_records(filtered_file)
    if not records:
        print("No filtered records found.")
        return

    print(f"Loaded {len(records)} filtered records")

    include_thinking = fmt_cfg.get("include_thinking", True)

    # Convert to messages format
    all_data = []
    for record in records:
        messages = record_to_messages(record, include_thinking=include_thinking)
        all_data.append({
            "messages": messages,
            "problem_id": record.get("problem_id", ""),
            "model_name": record.get("model_name", ""),
            "ppa_score": record.get("ppa_score", 0.0),
        })

    # Stratified train/val split by problem_id
    val_split = fmt_cfg.get("val_split", 0.1)
    problem_ids = sorted(set(d["problem_id"] for d in all_data))
    random.seed(42)
    random.shuffle(problem_ids)

    n_val_problems = max(1, int(len(problem_ids) * val_split))
    val_problem_ids = set(problem_ids[:n_val_problems])
    train_problem_ids = set(problem_ids[n_val_problems:])

    train_data = [d for d in all_data if d["problem_id"] in train_problem_ids]
    val_data = [d for d in all_data if d["problem_id"] in val_problem_ids]

    print(f"Split: {len(train_data)} train ({len(train_problem_ids)} problems), "
          f"{len(val_data)} val ({len(val_problem_ids)} problems)")

    # Save as parquet
    os.makedirs(os.path.dirname(train_file), exist_ok=True)

    train_df = pd.DataFrame({"messages": [d["messages"] for d in train_data]})
    val_df = pd.DataFrame({"messages": [d["messages"] for d in val_data]})

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    print(f"Saved {train_file} ({len(train_df)} rows)")
    print(f"Saved {val_file} ({len(val_df)} rows)")

    # Verify the parquet files can be read back
    _verify_parquet(train_file)
    _verify_parquet(val_file)


def _verify_parquet(path: str) -> None:
    """Quick verification that the parquet file has the expected structure."""
    df = pd.read_parquet(path)
    assert "messages" in df.columns, f"Missing 'messages' column in {path}"
    if len(df) > 0:
        first_messages = df.iloc[0]["messages"]
        if not isinstance(first_messages, list):
            first_messages = list(first_messages)
        assert isinstance(first_messages, list), f"messages should be a list, got {type(first_messages)}"
        assert len(first_messages) >= 2, f"messages should have at least 2 entries, got {len(first_messages)}"
        roles = [m["role"] for m in first_messages]
        assert "assistant" in roles, f"messages should contain an assistant turn, got roles: {roles}"
        print(f"  Verified {path}: {len(df)} rows, first entry has {len(first_messages)} messages")


def main():
    parser = argparse.ArgumentParser(description="Format filtered responses for SFT training")
    parser.add_argument("--config", default="recipe/ttt_rtl/sft/config.yaml", help="Config file path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    format_data(config)


if __name__ == "__main__":
    main()
