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
"""Filter collected responses through iverilog/yosys pipeline with deduplication."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path

import yaml

from recipe.ttt_rtl.rtl_utils import (
    RTLLMProblem,
    extract_verilog_code,
    load_rtllm_problems,
    run_iverilog_compile,
    run_iverilog_sim,
    run_yosys_synth,
    verilog_dedup_key,
)


def _evaluate_single(args: tuple) -> dict | None:
    """Evaluate a single response through the iverilog/yosys pipeline.

    Runs in a subprocess via multiprocessing.Pool.
    """
    record, problem_data, filter_cfg = args

    problem_id = record["problem_id"]
    testbench_path = problem_data["testbench_path"]
    design_name = problem_data["design_name"]
    data_files = problem_data.get("data_files", [])
    response_content = record["response_content"]

    compile_timeout = filter_cfg["compile_timeout_s"]
    sim_timeout = filter_cfg["sim_timeout_s"]
    synth_timeout = filter_cfg["synth_timeout_s"]

    # Extract Verilog code
    verilog_code = extract_verilog_code(response_content)
    if not verilog_code.strip():
        return {
            **record,
            "verilog_code": "",
            "syntax_ok": False,
            "func_ok": False,
            "ppa_score": 0.0,
            "dedup_key": "",
            "filter_msg": "No Verilog code found",
        }

    dedup = verilog_dedup_key(verilog_code)
    work_dir = tempfile.mkdtemp(prefix=f"sft_filter_{problem_id}_")

    try:
        # Level 1: Syntax check
        syntax_ok, compile_log = run_iverilog_compile(
            verilog_code=verilog_code,
            testbench_path=testbench_path,
            work_dir=work_dir,
            design_name=design_name,
            data_files=data_files,
            timeout_s=compile_timeout,
        )

        if not syntax_ok:
            return {
                **record,
                "verilog_code": verilog_code,
                "syntax_ok": False,
                "func_ok": False,
                "ppa_score": 0.0,
                "dedup_key": dedup,
                "filter_msg": f"Compile failed: {compile_log[:200]}",
            }

        # Level 2: Functional check
        func_ok, sim_output = run_iverilog_sim(
            work_dir=work_dir,
            timeout_s=sim_timeout,
        )

        # Level 3: PPA synthesis (even if func fails, for stats)
        ppa_score = 0.0
        synth_result = run_yosys_synth(
            verilog_code=verilog_code,
            design_name=design_name,
            work_dir=work_dir,
            timeout_s=synth_timeout,
        )
        if synth_result["success"]:
            ppa_score = float(synth_result["area"])

        return {
            **record,
            "verilog_code": verilog_code,
            "syntax_ok": True,
            "func_ok": func_ok,
            "ppa_score": ppa_score,
            "dedup_key": dedup,
            "filter_msg": "OK" if func_ok else f"Functional test failed: {sim_output[:200]}",
        }

    except Exception as e:
        return {
            **record,
            "verilog_code": verilog_code,
            "syntax_ok": False,
            "func_ok": False,
            "ppa_score": 0.0,
            "dedup_key": dedup,
            "filter_msg": f"Error: {e}",
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _load_raw_records(raw_dir: str) -> list[dict]:
    """Load all raw JSONL records."""
    records = []
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return records
    for jsonl_file in sorted(raw_path.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def filter_responses(config: dict) -> None:
    """Run the full filtering pipeline."""
    raw_dir = config["paths"]["raw_dir"]
    filtered_file = config["paths"]["filtered_file"]
    filter_cfg = config["filtering"]

    # Load problems for testbench paths
    problems = load_rtllm_problems(rtllm_dir=config["rtllm_dir"])
    problem_data_map = {}
    for pid, problem in problems.items():
        problem_data_map[pid] = {
            "testbench_path": problem.testbench_path,
            "design_name": problem.design_name,
            "data_files": problem.data_files,
        }

    # Load raw records
    records = _load_raw_records(raw_dir)
    if not records:
        print("No raw records found. Run collect_responses.py first.")
        return

    print(f"Loaded {len(records)} raw records")

    # Filter out records whose problem is not loaded
    valid_records = []
    for r in records:
        if r["problem_id"] in problem_data_map:
            valid_records.append(r)
        else:
            print(f"  Skipping unknown problem: {r['problem_id']}")
    records = valid_records

    # Prepare arguments for parallel evaluation
    eval_args = [(r, problem_data_map[r["problem_id"]], filter_cfg) for r in records]

    # Run evaluation in parallel
    max_workers = filter_cfg.get("max_workers", 8)
    print(f"Evaluating {len(eval_args)} responses with {max_workers} workers...")

    results: list[dict] = []
    with Pool(processes=max_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_evaluate_single, eval_args)):
            if result is not None:
                results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(eval_args):
                print(f"  Progress: {i + 1}/{len(eval_args)}")

    # Dedup within each problem
    if filter_cfg.get("dedup", True):
        seen_keys: dict[str, set[str]] = {}  # problem_id -> set of dedup_keys
        deduped = []
        for r in results:
            pid = r["problem_id"]
            dk = r.get("dedup_key", "")
            if not dk:
                continue
            if pid not in seen_keys:
                seen_keys[pid] = set()
            if dk in seen_keys[pid]:
                continue
            seen_keys[pid].add(dk)
            deduped.append(r)
        dup_count = len(results) - len(deduped)
        results = deduped
        print(f"Removed {dup_count} duplicates")

    # Filter to functionally correct only
    if filter_cfg.get("require_functional", True):
        func_results = [r for r in results if r.get("func_ok", False)]
        print(f"Functionally correct: {len(func_results)}/{len(results)}")
        kept = func_results
    else:
        kept = results

    # Write filtered output
    os.makedirs(os.path.dirname(filtered_file), exist_ok=True)
    with open(filtered_file, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(kept)} filtered records to {filtered_file}")

    # Print statistics
    _print_statistics(results)


def _print_statistics(results: list[dict]) -> None:
    """Print per-model and per-problem statistics."""
    from collections import defaultdict

    model_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "syntax": 0, "func": 0, "ppa_vals": []})
    problem_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "syntax": 0, "func": 0, "ppa_vals": []})

    for r in results:
        model = r.get("model_name", "unknown")
        pid = r.get("problem_id", "unknown")

        for stats in [model_stats[model], problem_stats[pid]]:
            stats["total"] += 1
            if r.get("syntax_ok"):
                stats["syntax"] += 1
            if r.get("func_ok"):
                stats["func"] += 1
            if r.get("ppa_score", 0) > 0:
                stats["ppa_vals"].append(r["ppa_score"])

    print("\n=== Per-Model Statistics ===")
    print(f"{'Model':<20} {'Total':>6} {'Syntax%':>8} {'Func%':>8} {'Unique':>7} {'Mean PPA':>10}")
    print("-" * 65)
    for model, s in sorted(model_stats.items()):
        syntax_pct = 100 * s["syntax"] / max(s["total"], 1)
        func_pct = 100 * s["func"] / max(s["total"], 1)
        mean_ppa = sum(s["ppa_vals"]) / max(len(s["ppa_vals"]), 1) if s["ppa_vals"] else 0
        print(f"{model:<20} {s['total']:>6} {syntax_pct:>7.1f}% {func_pct:>7.1f}% {s['func']:>7} {mean_ppa:>10.1f}")

    print("\n=== Per-Problem Statistics (top 10 by func count) ===")
    print(f"{'Problem':<30} {'Total':>6} {'Syntax%':>8} {'Func%':>8} {'Mean PPA':>10}")
    print("-" * 68)
    sorted_problems = sorted(problem_stats.items(), key=lambda x: x[1]["func"], reverse=True)
    for pid, s in sorted_problems[:10]:
        syntax_pct = 100 * s["syntax"] / max(s["total"], 1)
        func_pct = 100 * s["func"] / max(s["total"], 1)
        mean_ppa = sum(s["ppa_vals"]) / max(len(s["ppa_vals"]), 1) if s["ppa_vals"] else 0
        print(f"{pid:<30} {s['total']:>6} {syntax_pct:>7.1f}% {func_pct:>7.1f}% {mean_ppa:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Filter collected Verilog responses")
    parser.add_argument("--config", default="recipe/ttt_rtl/sft/config.yaml", help="Config file path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filter_responses(config)


if __name__ == "__main__":
    main()
