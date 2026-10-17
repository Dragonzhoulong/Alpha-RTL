"""Sweep OpenSTA power-measurement modes to reconcile with a paper's numbers.

Runs Yosys (Nangate45) on a single RTLLM reference design (default
``counter_12``), then re-runs OpenSTA with several ``(clock period,
switching activity)`` combinations and prints the resulting area/delay/power.

Motivation: activity-less ``report_power`` depends strongly on assumed input
activity and clock period. When a paper quotes e.g. ``counter_12 = 34.85 μm²,
0.24 ns, 98.6 μW`` but doesn't say how it measured power, this script lets us
probe which mode reproduces the quoted figure.

Usage (from repo root, inside the ``verl`` env):

    python -m recipe.ttt_rtl.benchmark_power_modes \\
        --problem counter_12 --rtllm-dir RTLLM
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from recipe.ttt_rtl.rtl_utils import (
    extract_tb_param_overrides,
    load_rtllm_problems,
    run_yosys_synth,
)


POWER_MODES: list[dict[str, Any]] = [
    # OpenSTA default: no set_power_activity — relies on Liberty's
    # default_input_switch_activity (most liberties leave this tiny or 0).
    {"name": "default (no set_power_activity)", "activity": None, "duty": None},
    {"name": "global activity=0.05 duty=0.5", "activity": 0.05, "duty": 0.5},
    {"name": "global activity=0.10 duty=0.5", "activity": 0.10, "duty": 0.5},
    {"name": "global activity=0.20 duty=0.5", "activity": 0.20, "duty": 0.5},
    {"name": "global activity=0.50 duty=0.5", "activity": 0.50, "duty": 0.5},
    {"name": "global activity=1.00 duty=0.5", "activity": 1.00, "duty": 0.5},
]

CLOCK_PERIODS_NS: list[float] = [10.0, 1.0, 0.5, 0.3]


def _write_sta_script(
    path: str,
    liberty_abs: str,
    netlist_abs: str,
    top_module: str,
    clk_period_ns: float,
    activity: float | None,
    duty: float | None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"read_liberty {liberty_abs}\n")
        f.write(f"read_verilog {netlist_abs}\n")
        f.write(f"link_design {top_module}\n")
        f.write("set has_clk 0\n")
        f.write("foreach clk_name {clk clock CLK clk_i sys_clk clk_in} {\n")
        f.write("    set p [get_ports $clk_name -quiet]\n")
        f.write("    if {[llength $p] > 0} {\n")
        f.write(f"        create_clock -name clk -period {clk_period_ns} $p\n")
        f.write("        set has_clk 1\n")
        f.write("        break\n")
        f.write("    }\n")
        f.write("}\n")
        f.write("if {$has_clk} {\n")
        f.write("    set_input_delay 0.0 -clock clk [all_inputs]\n")
        f.write("    set_output_delay 0.0 -clock clk [all_outputs]\n")
        f.write("    report_checks -path_delay max -digits 4\n")
        f.write("    report_worst_slack -digits 4\n")
        f.write("} else {\n")
        f.write("    set_input_delay 0.0 [all_inputs]\n")
        f.write("    set_output_delay 0.0 [all_outputs]\n")
        f.write("    report_checks -path_delay max -unconstrained -digits 4\n")
        f.write("    report_worst_slack -digits 4\n")
        f.write("}\n")
        if activity is not None and duty is not None:
            f.write(
                "set_power_activity -global "
                f"-activity {activity} -duty {duty}\n"
            )
        f.write('puts "=== POWER ==="\n')
        f.write("report_power -digits 6\n")
        f.write("exit\n")


def _parse_delay_ns(log: str) -> float:
    delay_ns = 0.0
    for line in log.splitlines():
        m = re.search(r"([\d.]+)\s+data arrival time", line)
        if m:
            v = float(m.group(1))
            if v > delay_ns:
                delay_ns = v
    return delay_ns


def _parse_power_w(log: str) -> tuple[float, float, float, float]:
    """Returns (internal, switching, leakage, total) in W."""
    for line in log.splitlines():
        m = re.match(
            r"^\s*Total\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
            line,
        )
        if m:
            try:
                return (
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
            except ValueError:
                pass
    return (0.0, 0.0, 0.0, 0.0)


def run_opensta_mode(
    netlist_abs: str,
    liberty_abs: str,
    top_module: str,
    opensta_binary: str,
    clk_period_ns: float,
    activity: float | None,
    duty: float | None,
    timeout_s: float,
    work_dir: str,
) -> dict[str, Any]:
    act_tag = "def" if activity is None else f"{activity:.2f}"
    script = os.path.join(work_dir, f"sta_clk{clk_period_ns}_act{act_tag}.tcl")
    _write_sta_script(
        script, liberty_abs, netlist_abs, top_module,
        clk_period_ns, activity, duty,
    )
    try:
        result = subprocess.run(
            [opensta_binary, "-no_init", "-exit", script],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = result.stdout + result.stderr
    except Exception as e:
        return {"ok": False, "err": str(e)}

    delay_ns = _parse_delay_ns(log)
    internal_w, switching_w, leakage_w, total_w = _parse_power_w(log)
    return {
        "ok": True,
        "delay_ns": delay_ns,
        "internal_w": internal_w,
        "switching_w": switching_w,
        "leakage_w": leakage_w,
        "total_w": total_w,
        "log": log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan OpenSTA power modes on one problem.")
    parser.add_argument("--problem", default="counter_12")
    parser.add_argument("--rtllm-dir", default="RTLLM")
    parser.add_argument(
        "--liberty-file",
        default="recipe/ttt_rtl/pdk/Nangate45_typ.lib",
    )
    parser.add_argument(
        "--opensta-binary",
        default="sta",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--paper-area-um2", type=float, default=34.85,
        help="Paper-reported area for the problem (used to flag a match).",
    )
    parser.add_argument(
        "--paper-delay-ns", type=float, default=0.24,
        help="Paper-reported critical-path delay.",
    )
    parser.add_argument(
        "--paper-power-uw", type=float, default=98.6,
        help="Paper-reported total power.",
    )
    parser.add_argument(
        "--keep-workdir", action="store_true",
        help="Keep the temp directory (containing netlist, TCL scripts, logs).",
    )
    args = parser.parse_args()

    liberty_abs = os.path.abspath(args.liberty_file)
    if not os.path.isfile(liberty_abs):
        print(f"ERROR: liberty not found: {liberty_abs}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.opensta_binary):
        print(f"ERROR: opensta not found: {args.opensta_binary}", file=sys.stderr)
        return 2

    problems = load_rtllm_problems(args.rtllm_dir, [args.problem])
    if args.problem not in problems:
        print(f"ERROR: problem {args.problem} not found under {args.rtllm_dir}", file=sys.stderr)
        return 3
    prob = problems[args.problem]

    ref_path = prob.reference_path
    if not ref_path or not os.path.isfile(ref_path):
        print(f"ERROR: reference file missing: {ref_path}", file=sys.stderr)
        return 3
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_code = f.read()
    # Rename `verified_xxx` to the design name so Yosys's top matches.
    ref_code = re.sub(
        r"\bmodule\s+verified_\w+",
        f"module {prob.design_name}",
        ref_code,
        count=1,
    )

    work_dir = tempfile.mkdtemp(prefix=f"ttt_pwr_{args.problem}_")
    print(f"[bench] problem      : {args.problem}")
    print(f"[bench] design_name  : {prob.design_name}")
    print(f"[bench] reference    : {ref_path}")
    print(f"[bench] liberty      : {liberty_abs}")
    print(f"[bench] opensta      : {args.opensta_binary}")
    print(f"[bench] work_dir     : {work_dir}")
    print()

    try:
        tb_params = extract_tb_param_overrides(prob.testbench_path, prob.design_name)
        synth = run_yosys_synth(
            verilog_code=ref_code,
            design_name=prob.design_name,
            work_dir=work_dir,
            timeout_s=args.timeout_s,
            liberty_file=liberty_abs,
            param_overrides=tb_params or None,
        )
        if not synth["success"]:
            print("[bench] Yosys synthesis FAILED:")
            print(synth.get("log", "")[-2000:])
            return 4

        area = float(synth["area"])
        abc_delay_ps = float(synth.get("delay", 0.0))
        netlist_path = synth.get("netlist_path", "")
        top_module = synth.get("top_module", prob.design_name)
        if not netlist_path or not os.path.isfile(netlist_path):
            print(f"[bench] netlist missing: {netlist_path}", file=sys.stderr)
            return 4

        print(f"[yosys]  area        : {area:10.3f}  um^2  (paper: {args.paper_area_um2})")
        print(f"[yosys]  abc delay   : {abc_delay_ps:10.3f}  ps")
        print(f"[yosys]  top module  : {top_module}")
        print(f"[yosys]  netlist     : {netlist_path}")
        print()

        # Header
        hdr = (
            f"  {'clk(ns)':>8s}  {'activity':>12s}  "
            f"{'delay(ns)':>10s}  {'internal(uW)':>13s}  "
            f"{'switch(uW)':>11s}  {'leak(uW)':>10s}  "
            f"{'total(uW)':>11s}  {'Δvs paper':>11s}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        best: tuple[float, dict[str, Any]] | None = None

        for clk_ns in CLOCK_PERIODS_NS:
            for mode in POWER_MODES:
                res = run_opensta_mode(
                    netlist_abs=os.path.abspath(netlist_path),
                    liberty_abs=liberty_abs,
                    top_module=top_module,
                    opensta_binary=args.opensta_binary,
                    clk_period_ns=clk_ns,
                    activity=mode["activity"],
                    duty=mode["duty"],
                    timeout_s=args.timeout_s,
                    work_dir=work_dir,
                )
                if not res["ok"]:
                    print(f"  clk={clk_ns:<6} {mode['name']:>28}  ERROR: {res.get('err')}")
                    continue
                total_uw = res["total_w"] * 1e6
                diff = total_uw - args.paper_power_uw
                marker = ""
                if abs(diff) < 5.0:
                    marker = "  <-- match"
                    if best is None or abs(diff) < abs(best[0]):
                        best = (diff, {"clk_ns": clk_ns, **mode, **res})
                act_label = "default" if mode["activity"] is None else f"{mode['activity']:.2f}/{mode['duty']:.1f}"
                print(
                    f"  {clk_ns:>8.3f}  {act_label:>12s}  "
                    f"{res['delay_ns']:>10.4f}  "
                    f"{res['internal_w'] * 1e6:>13.3f}  "
                    f"{res['switching_w'] * 1e6:>11.3f}  "
                    f"{res['leakage_w'] * 1e6:>10.3f}  "
                    f"{total_uw:>11.3f}  "
                    f"{diff:>+11.3f}{marker}"
                )

        print()
        print("=" * 80)
        print(
            f"  Paper targets  area={args.paper_area_um2} um^2, "
            f"delay={args.paper_delay_ns} ns, power={args.paper_power_uw} uW"
        )
        if best is not None:
            b = best[1]
            act = "default" if b["activity"] is None else f"{b['activity']:.2f}"
            print(
                f"  Closest mode : clk={b['clk_ns']} ns, activity={act}  "
                f"-> total={b['total_w'] * 1e6:.3f} uW "
                f"(Δ={best[0]:+.3f} uW)"
            )
        else:
            print("  No configuration reproduced the paper's power within 5 uW.")
        print("=" * 80)
        return 0
    finally:
        if args.keep_workdir:
            print(f"[bench] keeping work_dir={work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
