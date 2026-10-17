#!/usr/bin/env python3
"""Run the current RTL reward pipeline on RTLLM benchmark reference cases."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recipe.ttt_rtl.rtl_reward import compute_score
from recipe.ttt_rtl.rtl_utils import (
    RTLLMProblem,
    extract_tb_param_overrides,
    load_rtllm_problems,
    run_opensta_timing,
    run_yosys_synth,
)


def _extract_tb_module_name(problem: RTLLMProblem) -> str:
    """Extract the DUT module name expected by the testbench."""
    tb_path = problem.testbench_path
    if not tb_path or not os.path.isfile(tb_path):
        return ""

    skip = {
        "module",
        "reg",
        "wire",
        "input",
        "output",
        "assign",
        "always",
        "initial",
        "begin",
        "end",
        "integer",
        "parameter",
        "localparam",
        "genvar",
        "generate",
        "endgenerate",
        "if",
        "else",
        "for",
        "case",
        "endcase",
        "endmodule",
        "function",
        "task",
        "timescale",
    }

    try:
        with open(tb_path, "r", encoding="utf-8") as f:
            tb_code = f.read()
    except OSError:
        return ""

    for line in tb_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("`"):
            continue
        match = re.match(r"\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\(", stripped)
        if match:
            mod_name = match.group(1)
            if mod_name not in skip and not mod_name.startswith("$"):
                return mod_name
    return ""


def _read_reference(problem: RTLLMProblem) -> str:
    """Read the verified reference implementation and rename the module if needed."""
    if not problem.reference_path or not os.path.isfile(problem.reference_path):
        return ""

    try:
        with open(problem.reference_path, "r", encoding="utf-8") as f:
            code = f.read().strip()
    except OSError:
        return ""

    expected_name = _extract_tb_module_name(problem)
    if expected_name:
        code = re.sub(
            r"\bmodule\s+verified_\w+",
            f"module {expected_name}",
            code,
            count=1,
        )
    return code


def _compute_baseline(problem: RTLLMProblem, liberty_file: str | None, opensta_binary: str | None, clk_period_ns: float) -> dict[str, float]:
    """Compute the per-problem baseline used by the online reward."""
    ref_code = _read_reference(problem)
    if not ref_code:
        return {"area": 0.0, "delay": 0.0, "adp": 0.0}

    work_dir = tempfile.mkdtemp(prefix=f"ttt_rtl_reward_baseline_{problem.problem_id}_")
    try:
        tb_param_overrides = extract_tb_param_overrides(problem.testbench_path, problem.design_name)
        result = run_yosys_synth(
            verilog_code=ref_code,
            design_name=problem.design_name,
            work_dir=work_dir,
            timeout_s=60.0,
            liberty_file=liberty_file,
            param_overrides=tb_param_overrides or None,
        )
        if not result["success"] or float(result.get("area", 0.0)) <= 0:
            return {"area": 0.0, "delay": 0.0, "adp": 0.0}

        area = float(result["area"])
        delay = float(result.get("delay", 0.0))
        adp = float(result.get("adp", 0.0))

        if (
            opensta_binary
            and liberty_file
            and result.get("netlist_path")
            and os.path.isfile(result["netlist_path"])
        ):
            sta = run_opensta_timing(
                netlist_path=result["netlist_path"],
                top_module=result.get("top_module", problem.design_name),
                liberty_file=liberty_file,
                timeout_s=60.0,
                opensta_binary=opensta_binary,
                clk_period_ns=clk_period_ns,
            )
            if sta["success"] and float(sta["delay_ps"]) > 0:
                delay = float(sta["delay_ps"])
                adp = area * delay

        return {"area": area, "delay": delay, "adp": adp}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _format_score(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_benchmark(
    rtllm_dir: str,
    problems: list[str] | None,
    liberty_file: str | None,
    opensta_binary: str | None,
    timeout_s: float,
    opensta_timeout_s: float,
    clk_period_ns: float,
    w_syntax: float,
    w_func: float,
    w_ppa: float,
    failure_reward: float,
) -> int:
    """Run the full reward pipeline on RTLLM reference designs."""
    problem_dict = load_rtllm_problems(rtllm_dir=rtllm_dir, problems=problems)
    if not problem_dict:
        print(f"No RTLLM problems found in {rtllm_dir}.")
        return 1

    print("=" * 100)
    print("RTL Reward Pipeline Benchmark")
    print(f"RTLLM dir:         {os.path.abspath(rtllm_dir)}")
    print(f"Problems:          {', '.join(problem_dict.keys())}")
    print(f"Liberty file:      {liberty_file or '(disabled)'}")
    print(f"OpenSTA binary:    {opensta_binary or '(disabled)'}")
    print(f"Weights:           syntax={w_syntax}, func={w_func}, ppa={w_ppa}")
    print("=" * 100)
    print()

    results: list[dict[str, Any]] = []

    for problem_id, problem in problem_dict.items():
        print(f"[{problem_id}] category={problem.category}")
        ref_code = _read_reference(problem)
        if not ref_code:
            print("  reference: missing or unreadable")
            print()
            results.append({"problem": problem_id, "success": False, "reason": "missing reference"})
            continue

        baseline = _compute_baseline(problem, liberty_file, opensta_binary, clk_period_ns)
        solution_str = f"```verilog\n{ref_code}\n```"
        extra_info = {
            "ttt_problem_id": problem.problem_id,
            "ttt_testbench_path": problem.testbench_path,
            "ttt_design_name": problem.design_name,
            "ttt_parent_state_id": "benchmark_root",
            "ttt_data_files": problem.data_files,
            "ttt_ppa_baseline": baseline["adp"],
        }
        result = compute_score(
            data_source="ttt/rtl_benchmark",
            solution_str=solution_str,
            ground_truth="",
            extra_info=extra_info,
            timeout_s=timeout_s,
            w_syntax=w_syntax,
            w_func=w_func,
            w_ppa=w_ppa,
            failure_reward=failure_reward,
            liberty_file=liberty_file,
            opensta_binary=opensta_binary,
            opensta_timeout_s=opensta_timeout_s,
            clk_period_ns=clk_period_ns,
        )
        results.append({"problem": problem_id, "success": bool(result.get("ttt_success")), "result": result, "baseline": baseline})

        print(
            "  baseline: "
            f"area={baseline['area']:.1f}, delay={baseline['delay']:.1f}ps, adp={baseline['adp']:.1f}"
        )
        print(
            "  reward:   "
            f"score={_format_score(result.get('score'))}, "
            f"syntax_ok={result.get('ttt_syntax_ok')}, "
            f"func_ok={result.get('ttt_func_ok')}, "
            f"ppa_score={_format_score(result.get('ttt_ppa_score'))}, "
            f"area={_format_score(result.get('ttt_ppa_area'))}, "
            f"delay={_format_score(result.get('ttt_ppa_delay'))}"
        )
        msg = str(result.get("ttt_msg", "")).strip()
        if msg:
            msg_lines = msg.splitlines()
            preview = "\n".join(f"    {line}" for line in msg_lines[:8])
            print("  feedback:")
            print(preview)
            if len(msg_lines) > 8:
                print("    ...(truncated)")
        print()

    print("=" * 100)
    print("Summary")
    print("=" * 100)
    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    print(f"Total:    {len(results)}")
    print(f"Success:  {len(succeeded)}")
    print(f"Failed:   {len(failed)}")
    if succeeded:
        scores = [float(r["result"]["score"]) for r in succeeded]
        print(f"Score min/mean/max: {min(scores):.4f} / {sum(scores) / len(scores):.4f} / {max(scores):.4f}")
    if failed:
        print("Failed cases:")
        for item in failed:
            print(f"  - {item['problem']}: {item.get('reason', item['result'].get('ttt_msg', 'unknown failure'))}")
    return 0 if not failed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full RTL reward pipeline on RTLLM benchmark cases.")
    parser.add_argument("--rtllm-dir", default="RTLLM", help="RTLLM root directory")
    parser.add_argument(
        "--problems",
        default="adder_8bit,adder_16bit,counter_12,fsm,edge_detect",
        help="Comma-separated RTLLM problem IDs or problem directory names",
    )
    parser.add_argument(
        "--liberty-file",
        default="recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib",
        help="Path to Liberty timing library",
    )
    parser.add_argument(
        "--opensta-binary",
        default="sta",
        help="Path to OpenSTA binary; empty string disables OpenSTA",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Timeout for iverilog/yosys/vvp")
    parser.add_argument("--opensta-timeout-s", type=float, default=60.0, help="Timeout for OpenSTA")
    parser.add_argument("--clk-period-ns", type=float, default=10.0, help="Clock period used by OpenSTA")
    parser.add_argument("--w-syntax", type=float, default=0.1, help="Syntax reward weight")
    parser.add_argument("--w-func", type=float, default=1.0, help="Functional reward weight")
    parser.add_argument("--w-ppa", type=float, default=10.0, help="PPA reward weight")
    parser.add_argument("--failure-reward", type=float, default=0.0, help="Reward when no Verilog is extracted")
    args = parser.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    liberty_file = args.liberty_file if args.liberty_file and os.path.isfile(args.liberty_file) else None
    opensta_binary = args.opensta_binary.strip() or None

    return run_benchmark(
        rtllm_dir=args.rtllm_dir,
        problems=problems or None,
        liberty_file=liberty_file,
        opensta_binary=opensta_binary,
        timeout_s=args.timeout_s,
        opensta_timeout_s=args.opensta_timeout_s,
        clk_period_ns=args.clk_period_ns,
        w_syntax=args.w_syntax,
        w_func=args.w_func,
        w_ppa=args.w_ppa,
        failure_reward=args.failure_reward,
    )


if __name__ == "__main__":
    raise SystemExit(main())
