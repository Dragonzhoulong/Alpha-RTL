"""Benchmark all RTLLM reference designs — area / delay / power / ADP / ADPP.

Synthesizes each ``verified_*.v`` with Yosys against the recipe's default
standard-cell library, then runs OpenSTA for accurate critical-path delay and
activity-less power. Reports results as both a stdout table and a CSV file.

Usage (from repo root):

    python -m recipe.ttt_rtl.benchmark_reference_adp \
        --rtllm-dir RTLLM \
        --output-csv recipe/ttt_rtl/reference_adp.csv

All defaults match ``recipe/ttt_rtl/config/ttt_rtl_trainer.yaml``:
    - liberty: recipe/ttt_rtl/pdk/Nangate45_typ.lib
    - opensta: sta
    - clk_period_ns: 10.0
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from recipe.ttt_rtl.rtl_utils import (
    RTLLMProblem,
    extract_tb_param_overrides,
    load_rtllm_problems,
    run_opensta_timing,
    run_yosys_synth,
)


def _extract_tb_module_name(problem: RTLLMProblem) -> str:
    """Return the DUT module name instantiated in the testbench (or '')."""
    tb_path = problem.testbench_path
    if not tb_path or not os.path.isfile(tb_path):
        return ""
    try:
        with open(tb_path, "r", encoding="utf-8") as f:
            tb_code = f.read()
    except Exception:
        return ""
    skip = {
        "module", "reg", "wire", "input", "output", "assign", "always", "initial",
        "begin", "end", "integer", "parameter", "localparam", "genvar", "generate",
        "endgenerate", "if", "else", "for", "case", "endcase", "endmodule",
        "function", "task", "timescale",
    }
    for line in tb_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("`"):
            continue
        m = re.match(r"\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\(", stripped)
        if m:
            mod_name = m.group(1)
            if mod_name not in skip:
                return mod_name
    return ""


def _read_reference(problem: RTLLMProblem) -> str:
    if not problem.reference_path or not os.path.isfile(problem.reference_path):
        return ""
    try:
        with open(problem.reference_path, "r", encoding="utf-8") as f:
            code = f.read().strip()
    except Exception:
        return ""
    expected = _extract_tb_module_name(problem)
    if expected:
        code = re.sub(
            r"\bmodule\s+verified_\w+",
            f"module {expected}",
            code,
            count=1,
        )
    return code


def bench_reference(
    problem: RTLLMProblem,
    liberty_file: str,
    opensta_binary: str | None,
    clk_period_ns: float,
    timeout_s: float,
    power_activity: float | None = None,
    power_duty: float = 0.5,
    power_clock_from_delay: bool = False,
    power_clock_margin: float = 1.25,
) -> dict[str, Any]:
    """Synthesize one reference design and return PPA metrics."""
    row: dict[str, Any] = {
        "problem_id": problem.problem_id,
        "category": problem.category,
        "area_um2": 0.0,
        "delay_ps": 0.0,
        "power_w": 0.0,
        "power_clock_ns": 0.0,
        "adp": 0.0,
        "adpp": 0.0,
        "timing_source": "",
        "success": False,
        "note": "",
    }

    ref_code = _read_reference(problem)
    if not ref_code:
        row["note"] = "no reference file"
        return row

    work_dir = tempfile.mkdtemp(prefix=f"bench_adp_{problem.problem_id}_")
    try:
        tb_params = extract_tb_param_overrides(problem.testbench_path, problem.design_name)
        synth = run_yosys_synth(
            verilog_code=ref_code,
            design_name=problem.design_name,
            work_dir=work_dir,
            timeout_s=timeout_s,
            liberty_file=liberty_file,
            param_overrides=tb_params or None,
        )
        if not synth["success"] or float(synth.get("area", 0.0)) <= 0.0:
            row["note"] = "yosys synth failed"
            return row

        area = float(synth["area"])
        delay = float(synth.get("delay", 0.0))
        power = 0.0
        timing_source = "abc" if delay > 0 else ""

        netlist_path = synth.get("netlist_path", "")
        if (
            opensta_binary
            and netlist_path
            and os.path.isfile(netlist_path)
            and liberty_file
            and os.path.isfile(liberty_file)
        ):
            sta = run_opensta_timing(
                netlist_path=netlist_path,
                top_module=synth.get("top_module", problem.design_name),
                liberty_file=liberty_file,
                timeout_s=timeout_s,
                opensta_binary=opensta_binary,
                clk_period_ns=clk_period_ns,
                power_activity=power_activity,
                power_duty=power_duty,
                power_clock_from_delay=power_clock_from_delay,
                power_clock_margin=power_clock_margin,
            )
            if sta["success"] and sta["delay_ps"] > 0:
                delay = float(sta["delay_ps"])
                power = float(sta.get("power_w", 0.0))
                row["power_clock_ns"] = float(sta.get("power_clock_ns", clk_period_ns))
                timing_source = "opensta"

        adp = area * delay if (area > 0 and delay > 0) else 0.0
        adpp = adp * power if (adp > 0 and power > 0) else 0.0

        row.update(
            {
                "area_um2": area,
                "delay_ps": delay,
                "power_w": power,
                "adp": adp,
                "adpp": adpp,
                "timing_source": timing_source,
                "success": True,
            }
        )
        return row
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _print_row(row: dict[str, Any]) -> None:
    if not row["success"]:
        print(f"[bench] {row['problem_id']:30s}  SKIP  ({row['note']})", flush=True)
        return
    print(
        f"[bench] {row['problem_id']:30s}  "
        f"area={row['area_um2']:10.1f} um^2  "
        f"delay={row['delay_ps']:8.1f} ps  "
        f"power={row['power_w'] * 1e6:8.3f} uW  "
        f"ADP={row['adp']:.3e}  ADPP={row['adpp']:.3e}  "
        f"[{row['timing_source']}]",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RTLLM reference ADP/ADPP benchmark.")
    parser.add_argument("--rtllm-dir", default="RTLLM", help="Path to RTLLM root.")
    parser.add_argument(
        "--liberty-file",
        default="recipe/ttt_rtl/pdk/Nangate45_typ.lib",
        help="Liberty (.lib) file for tech-mapped synthesis.",
    )
    parser.add_argument(
        "--opensta-binary",
        default="sta",
        help="OpenSTA executable (empty string disables OpenSTA).",
    )
    parser.add_argument("--clk-period-ns", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    # --- Paper-matching power-reporting knobs (defaults derived from a
    # counter_12 sweep; see benchmark_power_modes.py).
    parser.add_argument(
        "--power-activity",
        type=float,
        default=0.20,
        help="Global switching activity passed to OpenSTA's set_power_activity."
        " Use a negative value to leave the OpenSTA default unset.",
    )
    parser.add_argument("--power-duty", type=float, default=0.5)
    parser.add_argument(
        "--power-clock-from-delay",
        action="store_true",
        default=True,
        help="Report power at clk ≈ critical-path delay × margin (paper convention).",
    )
    parser.add_argument(
        "--no-power-clock-from-delay",
        dest="power_clock_from_delay",
        action="store_false",
        help="Report power at the fixed --clk-period-ns instead of delay-derived clock.",
    )
    parser.add_argument("--power-clock-margin", type=float, default=1.25)
    parser.add_argument(
        "--problems",
        default="",
        help="Comma-separated problem filter; empty = all RTLLM problems.",
    )
    parser.add_argument(
        "--output-csv",
        default="recipe/ttt_rtl/reference_adp.csv",
        help="Where to write the CSV table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (1 = serial).",
    )
    args = parser.parse_args()

    liberty_file = os.path.abspath(args.liberty_file) if args.liberty_file else ""
    if not liberty_file or not os.path.isfile(liberty_file):
        print(f"ERROR: liberty file not found: {args.liberty_file}", file=sys.stderr)
        return 2

    opensta_binary = args.opensta_binary.strip() or None
    if opensta_binary and not os.path.isfile(opensta_binary):
        print(
            f"WARNING: OpenSTA binary not found: {opensta_binary}  "
            f"— falling back to ABC stime and no power.",
            file=sys.stderr,
        )
        opensta_binary = None

    problem_filter = [p.strip() for p in args.problems.split(",") if p.strip()] or None
    problems = load_rtllm_problems(args.rtllm_dir, problem_filter)
    if not problems:
        print(f"ERROR: no problems loaded from {args.rtllm_dir}", file=sys.stderr)
        return 3

    power_activity = args.power_activity if args.power_activity >= 0 else None
    bench_kwargs = {
        "power_activity": power_activity,
        "power_duty": args.power_duty,
        "power_clock_from_delay": args.power_clock_from_delay,
        "power_clock_margin": args.power_clock_margin,
    }

    print(f"[bench] Loaded {len(problems)} RTLLM problems from {args.rtllm_dir}")
    print(f"[bench] Liberty : {liberty_file}")
    print(f"[bench] OpenSTA : {opensta_binary or '(disabled)'}")
    print(f"[bench] Clock   : {args.clk_period_ns} ns (timing)")
    if args.power_clock_from_delay:
        print(
            f"[bench] Power clk: delay × {args.power_clock_margin} "
            f"(paper-style sign-off clock)"
        )
    else:
        print(f"[bench] Power clk: {args.clk_period_ns} ns (fixed)")
    act_tag = "OpenSTA default" if power_activity is None else f"{power_activity:.2f}"
    print(f"[bench] Activity: {act_tag}, duty={args.power_duty}")
    print(f"[bench] Workers : {args.workers}")
    print()

    rows: list[dict[str, Any]] = []
    items = sorted(problems.items())

    if args.workers <= 1:
        for _pid, prob in items:
            row = bench_reference(
                prob, liberty_file, opensta_binary, args.clk_period_ns, args.timeout_s,
                **bench_kwargs,
            )
            rows.append(row)
            _print_row(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    bench_reference,
                    prob,
                    liberty_file,
                    opensta_binary,
                    args.clk_period_ns,
                    args.timeout_s,
                    **bench_kwargs,
                ): prob.problem_id
                for _pid, prob in items
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "problem_id": pid,
                        "category": "",
                        "area_um2": 0.0,
                        "delay_ps": 0.0,
                        "power_w": 0.0,
                        "power_clock_ns": 0.0,
                        "adp": 0.0,
                        "adpp": 0.0,
                        "timing_source": "",
                        "success": False,
                        "note": f"exception: {exc}",
                    }
                rows.append(row)
                _print_row(row)

    # Stable ordering for the final CSV/table: category then problem_id
    rows.sort(key=lambda r: (r["category"], r["problem_id"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    fieldnames = [
        "problem_id", "category", "area_um2", "delay_ps", "power_w",
        "power_clock_ns", "adp", "adpp", "timing_source", "success", "note",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print()
    print("=" * 110)
    print(
        f"{'problem':30s} {'category':24s} "
        f"{'area(um^2)':>12s} {'delay(ps)':>10s} {'power(uW)':>10s} "
        f"{'ADP':>12s} {'ADPP':>12s}"
    )
    print("-" * 110)
    ok = [r for r in rows if r["success"]]
    for r in rows:
        if not r["success"]:
            continue
        print(
            f"{r['problem_id']:30s} {r['category']:24s} "
            f"{r['area_um2']:12.1f} {r['delay_ps']:10.1f} "
            f"{r['power_w'] * 1e6:10.3f} {r['adp']:12.3e} {r['adpp']:12.3e}"
        )
    skipped = [r for r in rows if not r["success"]]
    if skipped:
        print("-" * 110)
        for r in skipped:
            print(f"{r['problem_id']:30s} SKIP  ({r['note']})")
    print("=" * 110)
    print(f"[bench] {len(ok)}/{len(rows)} succeeded — wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
