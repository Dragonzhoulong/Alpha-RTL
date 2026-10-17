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
"""Build SFT warm-up dataset for TTT-C910-FPU from verified solutions.

Creates two types of training examples that mirror the RL prompt construction
(TTTC910Dataset._build_user_prompt in dataset.py):

  Type 1 — Initial: reference implementation + full spec → solution Verilog
  Type 2 — Iterative: abbreviated spec + previous attempt + feedback → improved Verilog

Run from the repo root:
    python -m recipe.ttt_c910_fpu.sft.format_sft_data --config recipe/ttt_c910_fpu/sft/config.yaml
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re

import pandas as pd
import yaml


SYSTEM_PROMPT = "You are an expert Verilog RTL designer specializing in high-performance arithmetic units."


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_passing_solutions(solutions_dir: str, measurements_path: str) -> list[dict]:
    """Load functionally correct solutions with valid PPA metrics."""
    with open(measurements_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    solutions = []

    for sol_name, metrics in results.items():
        if not metrics.get("testbench_pass", False):
            continue
        if metrics.get("adp", 0.0) <= 0:
            continue

        verilog_path = os.path.join(solutions_dir, sol_name, "ct_vfmau_mult_compressor.v")
        if not os.path.isfile(verilog_path):
            continue

        with open(verilog_path, "r", encoding="utf-8") as f:
            code = f.read().strip()

        solutions.append({
            "name": sol_name,
            "code": code,
            "adp": float(metrics["adp"]),
            "area": float(metrics["area"]),
            "delay_ps": float(metrics["delay_ps"]),
            "reward": float(metrics["reward"]),
        })

    # Sort by ADP ascending (best ADP first)
    return sorted(solutions, key=lambda x: x["adp"])


def build_initial_prompt(reference_code: str, prompt_template: str, baseline_metrics: dict) -> str:
    """Build the initial state user prompt.

    Mirrors TTTC910Dataset._build_user_prompt when state.verilog_code is empty:
    reference implementation + ADP baseline info + full spec + optimization target.
    """
    parts: list[str] = []
    baseline_adp = baseline_metrics.get("adp", 0.0)
    baseline_area = baseline_metrics.get("area", 0.0)
    baseline_delay = baseline_metrics.get("delay_ps", 0.0)

    if reference_code:
        parts.append("## Reference Implementation\n")
        if baseline_adp > 0:
            parts.append(
                f"The reference synthesizes to area={baseline_area:.1f}\u03bcm\u00b2, "
                f"delay={baseline_delay:.1f}ps (ADP={baseline_adp:.1f}). "
                f"Produce a functionally correct implementation with lower ADP.\n"
            )
        parts.append("```verilog")
        parts.append(reference_code)
        parts.append("```\n")

    parts.append(prompt_template)

    if baseline_adp > 0:
        parts.append(f"\nOptimize for ADP (Area \u00d7 Delay Product) lower than {baseline_adp:.1f}.")

    parts.append("\nOutput only Verilog code within ```verilog fences.")
    return "\n".join(parts)


def build_iterative_prompt(
    prompt_template: str,
    prev_code: str,
    prev_adp: float,
    prev_area: float,
    prev_delay: float,
    baseline_metrics: dict,
) -> str:
    """Build the iterative improvement user prompt.

    Mirrors TTTC910Dataset._build_user_prompt when state.verilog_code is set:
    abbreviated spec + previous attempt code + synthesis feedback + optimization target.
    """
    parts: list[str] = []
    baseline_adp = baseline_metrics.get("adp", 0.0)

    # Abbreviated spec: keep everything up to "## Available Building Blocks"
    template_lines = prompt_template.split("\n")
    spec_end = len(template_lines)
    for idx, line in enumerate(template_lines):
        if idx > 0 and line.startswith("## Available Building Blocks"):
            spec_end = idx
            break
    parts.append("\n".join(template_lines[:spec_end]))

    parts.append("\n## Previous Attempt\n")
    parts.append("```verilog")
    parts.append(prev_code)
    parts.append("```\n")

    parts.append("## Feedback from Previous Attempt\n")
    parts.append("- Syntax: PASS")
    parts.append("- Functional Test: PASS")

    if prev_adp > 0 and prev_area > 0:
        ppa_line = (
            f"- Synthesis: area={prev_area:.1f}\u03bcm\u00b2, "
            f"delay={prev_delay:.1f}ps, ADP={prev_adp:.1f}"
        )
        if baseline_adp > 0:
            ppa_line += f" (reference ADP: {baseline_adp:.1f})"
        parts.append(ppa_line)

    if prev_adp > 0 and prev_area > 0:
        parts.append(
            f"\nOptimizing ADP (lower is better). "
            f"Current: area={prev_area:.1f}\u03bcm\u00b2, delay={prev_delay:.1f}ps, ADP={prev_adp:.1f}. "
            f"Produce a functionally correct design with ADP lower than {prev_adp:.1f}."
        )
    else:
        parts.append("\nThe design is functionally correct. Try to optimize for better PPA.")

    parts.append("\nOutput only Verilog code within ```verilog fences.")
    return "\n".join(parts)


def build_repair_prompt(
    prompt_template: str,
    prev_code: str,
    observation: str,
    syntax_ok: bool,
    func_ok: bool,
) -> str:
    """Build a repair prompt using a failed rollout sample and its real feedback."""
    parts: list[str] = []

    template_lines = prompt_template.split("\n")
    spec_end = len(template_lines)
    for idx, line in enumerate(template_lines):
        if idx > 0 and line.startswith("## Available Building Blocks"):
            spec_end = idx
            break
    parts.append("\n".join(template_lines[:spec_end]))

    parts.append("\n## Previous Attempt\n")
    parts.append("```verilog")
    parts.append(prev_code)
    parts.append("```\n")

    parts.append("## Feedback from Previous Attempt\n")
    parts.append(f"- Syntax: {'PASS' if syntax_ok else 'FAIL'}")
    parts.append(f"- Functional Test: {'PASS' if func_ok else ('FAIL' if syntax_ok else 'N/A')}")

    obs = (observation or "").strip()
    if obs:
        if len(obs) > 1200:
            obs = obs[:1200] + "\n...(truncated)"
        parts.append(f"\n{obs}")

    parts.append(
        "\nFix the design conservatively. Preserve the required interface and pipeline behavior, "
        "eliminate the reported errors first, and return a functionally correct implementation."
    )
    parts.append("\nOutput only Verilog code within ```verilog fences.")
    return "\n".join(parts)


def _derive_repair_plan(observation: str, syntax_ok: bool, func_ok: bool) -> str:
    """Create a short natural-language repair rationale from tool feedback."""
    plan: list[str] = []
    obs = observation or ""

    if not syntax_ok:
        if "has already been declared" in obs:
            plan.append("Remove duplicate declarations or instance names instead of redefining existing nets and modules.")
        if "macro " in obs and "undefined" in obs:
            plan.append("Avoid introducing undefined macros; use concrete widths and the existing module parameters/signals.")
        if "Unable to bind wire/reg/memory" in obs:
            plan.append("Restore missing signal declarations and keep bit-slices consistent with declared vector widths.")
        if "expects" in obs and ("got" in obs or "Pruning" in obs):
            plan.append("Match every instantiated port width to the dependency module interface exactly.")
        if "syntax error" in obs or "invalid module" in obs or "continuous assignment" in obs:
            plan.append("Use legal Verilog syntax only: continuous assignments outside always blocks, valid instantiations, and no placeholder fragments.")
        if not plan:
            plan.append("Repair the compilation issues first and keep the implementation close to a known-good structure.")
    elif not func_ok:
        plan.append("Keep the code compiling, then repair the arithmetic datapath so `sum + carry` matches the reference outputs.")
        if "output mismatch" in obs:
            plan.append("Preserve the reference Booth/compressor pipeline structure instead of simplifying it into placeholders.")
    else:
        plan.append("Preserve correctness while improving the implementation conservatively.")

    joined = "Repair plan:\n" + "\n".join(f"{idx}. {item}" for idx, item in enumerate(plan, start=1))
    return joined


def make_response(
    code: str,
    include_thinking_placeholder: bool,
    repair_plan: str | None = None,
) -> str:
    """Format the assistant response.

    If include_thinking_placeholder=True, prepend an empty <think> block so the
    model learns to emit thinking tokens before the code (matching RL generation
    format with enable_thinking=True).
    """
    verilog_block = f"```verilog\n{code}\n```"
    body = verilog_block if not repair_plan else f"{repair_plan}\n\n{verilog_block}"
    if include_thinking_placeholder:
        return f"<think>\n\n</think>\n\n{body}"
    return body


def _classify_failure(observation: str, syntax_ok: bool, func_ok: bool) -> str:
    obs = observation or ""
    if not syntax_ok:
        if "has already been declared" in obs:
            return "redeclared_signal"
        if "Unable to bind wire/reg/memory" in obs:
            return "missing_signal"
        if "macro " in obs and "undefined" in obs:
            return "undefined_macro"
        if "expects" in obs and ("got" in obs or "Pruning" in obs):
            return "port_width_mismatch"
        if "syntax error" in obs or "invalid module" in obs or "continuous assignment" in obs:
            return "verilog_syntax"
        return "other_compile"
    if syntax_ok and not func_ok:
        return "functional_fail"
    return "other"


def load_failed_rollouts(
    rollout_glob: str,
    max_failed_samples: int,
    max_per_category: int,
) -> list[dict]:
    """Load and deduplicate failed rollout samples for repair-style SFT."""
    if not rollout_glob:
        return []

    files = sorted(glob.glob(rollout_glob))
    if not files:
        return []

    failures: list[dict] = []
    category_counts: dict[str, int] = {}
    seen_hashes: set[str] = set()

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                syntax_ok = bool(sample.get("ttt_syntax_ok", False))
                func_ok = bool(sample.get("ttt_func_ok", False))
                if syntax_ok and func_ok:
                    continue

                code = (sample.get("ttt_code") or "").strip()
                observation = (sample.get("ttt_msg") or "").strip()
                if not code or not observation:
                    continue

                digest = hashlib.sha1(code.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    continue

                category = _classify_failure(observation, syntax_ok, func_ok)
                if category_counts.get(category, 0) >= max_per_category:
                    continue

                seen_hashes.add(digest)
                category_counts[category] = category_counts.get(category, 0) + 1
                failures.append({
                    "code": code,
                    "observation": observation,
                    "syntax_ok": syntax_ok,
                    "func_ok": func_ok,
                    "step": int(sample.get("step", -1)),
                    "parent_state_id": sample.get("ttt_parent_state_id", ""),
                    "failure_type": category,
                })

                if len(failures) >= max_failed_samples:
                    return failures

    return failures


def format_data(config: dict) -> None:
    """Build and save train/val parquet files from verified solutions."""
    cfg_paths = config["paths"]
    cfg_sft = config.get("sft_data", {})

    repo_root = cfg_paths.get("repo_root", ".")
    solutions_dir = os.path.join(repo_root, cfg_paths["solutions_dir"])
    measurements_path = os.path.join(repo_root, cfg_paths["reward_measurements"])
    reference_path = os.path.join(repo_root, cfg_paths["reference_v"])
    prompt_template_path = os.path.join(repo_root, cfg_paths["prompt_template"])
    train_file = os.path.join(repo_root, cfg_paths["train_file"])
    val_file = os.path.join(repo_root, cfg_paths["val_file"])

    include_thinking_placeholder = bool(cfg_sft.get("include_thinking_placeholder", False))
    val_fraction = float(cfg_sft.get("val_fraction", 0.15))
    iterative_max_pairs_per_source = int(cfg_sft.get("iterative_max_pairs_per_source", 3))
    min_adp_improvement_pct = float(cfg_sft.get("min_adp_improvement_pct", 1.0))
    random_seed = int(cfg_sft.get("random_seed", 42))
    include_failure_repair = bool(cfg_sft.get("include_failure_repair", False))
    max_failed_samples = int(cfg_sft.get("max_failed_samples", 0))
    max_failed_samples_per_category = int(cfg_sft.get("max_failed_samples_per_category", 8))

    reference_code = _read_file(reference_path)
    prompt_template = _read_file(prompt_template_path)

    with open(measurements_path, "r", encoding="utf-8") as f:
        measurements = json.load(f)
    baseline_metrics = measurements.get("baseline_metrics", {})

    rollout_glob = ""
    if include_failure_repair:
        rollout_glob = cfg_paths.get("failed_rollout_glob", "")
        if rollout_glob and not os.path.isabs(rollout_glob):
            rollout_glob = os.path.join(repo_root, rollout_glob)

    solutions = load_passing_solutions(solutions_dir, measurements_path)
    print(f"Loaded {len(solutions)} functionally correct solutions with valid PPA metrics")
    if not solutions:
        print("No valid solutions found. Exiting.")
        return

    random.seed(random_seed)
    all_examples: list[dict] = []

    # ------------------------------------------------------------------
    # Type 1: Initial prompt — reference + full spec → solution code
    # ------------------------------------------------------------------
    initial_user = build_initial_prompt(reference_code, prompt_template, baseline_metrics)
    for sol in solutions:
        response = make_response(sol["code"], include_thinking_placeholder)
        all_examples.append({
            "example_type": "initial",
            "solution_name": sol["name"],
            "source_name": "",
            "adp": sol["adp"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user},
                {"role": "assistant", "content": response},
            ],
        })
    n_type1 = len(solutions)
    print(f"  Type 1 (initial):   {n_type1} examples")

    # ------------------------------------------------------------------
    # Type 2: Iterative improvement — previous attempt + feedback → better
    # ------------------------------------------------------------------
    baseline_adp = baseline_metrics.get("adp", 0.0)
    baseline_area = baseline_metrics.get("area", 0.0)
    baseline_delay = baseline_metrics.get("delay_ps", 0.0)

    # Load the reference (baseline) Verilog — it lives in solutions/baseline/
    baseline_code = ""
    baseline_v = os.path.join(solutions_dir, "baseline", "ct_vfmau_mult_compressor.v")
    if os.path.isfile(baseline_v):
        with open(baseline_v, "r", encoding="utf-8") as f:
            baseline_code = f.read().strip()

    iterative_examples: list[dict] = []

    def _adp_improves(src_adp: float, tgt_adp: float) -> bool:
        if src_adp <= 0:
            return False
        return (src_adp - tgt_adp) / src_adp * 100 >= min_adp_improvement_pct

    # Pair A: baseline → each solution that beats the baseline
    if baseline_code:
        baseline_user = build_iterative_prompt(
            prompt_template=prompt_template,
            prev_code=baseline_code,
            prev_adp=baseline_adp,
            prev_area=baseline_area,
            prev_delay=baseline_delay,
            baseline_metrics=baseline_metrics,
        )
        for sol in solutions:
            if not _adp_improves(baseline_adp, sol["adp"]):
                continue
            iterative_examples.append({
                "example_type": "iterative",
                "solution_name": sol["name"],
                "source_name": "baseline",
                "adp": sol["adp"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": baseline_user},
                    {"role": "assistant", "content": make_response(sol["code"], include_thinking_placeholder)},
                ],
            })

    # Pair B: worse solutions → better solutions
    # For each source (sorted worst→best), find up to N targets that are meaningfully better
    sources_desc = sorted(solutions, key=lambda x: x["adp"], reverse=True)
    for src in sources_desc:
        targets = [
            t for t in solutions
            if t["name"] != src["name"] and _adp_improves(src["adp"], t["adp"])
        ]
        if not targets:
            continue
        # Take the best improvements (lowest ADP targets)
        targets = sorted(targets, key=lambda x: x["adp"])[:iterative_max_pairs_per_source]

        src_user = build_iterative_prompt(
            prompt_template=prompt_template,
            prev_code=src["code"],
            prev_adp=src["adp"],
            prev_area=src["area"],
            prev_delay=src["delay_ps"],
            baseline_metrics=baseline_metrics,
        )
        for tgt in targets:
            iterative_examples.append({
                "example_type": "iterative",
                "solution_name": tgt["name"],
                "source_name": src["name"],
                "adp": tgt["adp"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": src_user},
                    {"role": "assistant", "content": make_response(tgt["code"], include_thinking_placeholder)},
                ],
            })

    # Deduplicate by (source_name, solution_name)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for ex in iterative_examples:
        key = (ex["source_name"], ex["solution_name"])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    all_examples.extend(deduped)
    print(f"  Type 2 (iterative): {len(deduped)} examples")

    # ------------------------------------------------------------------
    # Type 3: Repair — failed rollout + real feedback → known-good code
    # ------------------------------------------------------------------
    repair_examples: list[dict] = []
    if include_failure_repair and max_failed_samples > 0:
        failed_rollouts = load_failed_rollouts(
            rollout_glob=rollout_glob,
            max_failed_samples=max_failed_samples,
            max_per_category=max_failed_samples_per_category,
        )

        target_solution = None
        if baseline_code:
            target_solution = {
                "name": "baseline",
                "code": baseline_code,
                "adp": baseline_adp,
            }
        elif solutions:
            target_solution = solutions[0]

        if target_solution:
            for failed in failed_rollouts:
                user_prompt = build_repair_prompt(
                    prompt_template=prompt_template,
                    prev_code=failed["code"],
                    observation=failed["observation"],
                    syntax_ok=failed["syntax_ok"],
                    func_ok=failed["func_ok"],
                )
                repair_examples.append({
                    "example_type": "repair",
                    "solution_name": target_solution["name"],
                    "source_name": failed["parent_state_id"],
                    "adp": target_solution.get("adp", 0.0),
                    "failure_type": failed["failure_type"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "content": make_response(
                                target_solution["code"],
                                include_thinking_placeholder,
                                repair_plan=_derive_repair_plan(
                                    failed["observation"],
                                    failed["syntax_ok"],
                                    failed["func_ok"],
                                ),
                            ),
                        },
                    ],
                })

        all_examples.extend(repair_examples)
    print(f"  Type 3 (repair):    {len(repair_examples)} examples")
    print(f"  Total:              {len(all_examples)} examples")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    print(f"  Split:              {len(train_examples)} train / {len(val_examples)} val")

    # ------------------------------------------------------------------
    # Save parquet
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)

    train_df = pd.DataFrame({"messages": [ex["messages"] for ex in train_examples]})
    val_df = pd.DataFrame({"messages": [ex["messages"] for ex in val_examples]})
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    print(f"Saved: {train_file} ({len(train_df)} rows)")
    print(f"       {val_file} ({len(val_df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Format C910 FPU solutions as SFT training data")
    parser.add_argument("--config", default="recipe/ttt_c910_fpu/sft/config.yaml", help="Config file path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    format_data(config)


if __name__ == "__main__":
    main()
