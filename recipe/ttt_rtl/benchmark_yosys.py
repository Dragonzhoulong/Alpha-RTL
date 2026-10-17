#!/usr/bin/env python3
"""Benchmark Yosys synthesis time on RTLLM reference designs."""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add recipe to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recipe.ttt_rtl.rtl_utils import load_rtllm_problems, run_yosys_synth


def benchmark_yosys(rtllm_dir: str, problems: list[str], liberty_file: str | None = None):
    """Benchmark Yosys synthesis time on reference designs."""

    print("=" * 80)
    print("Yosys Synthesis Time Benchmark")
    if liberty_file and os.path.isfile(liberty_file):
        print(f"  Liberty: {liberty_file}")
        print("  Mode: Technology-mapped (area in \u03bcm\u00b2, delay in ps)")
    else:
        print("  Mode: Generic (abstract cell count)")
        liberty_file = None
    print("=" * 80)
    print()

    # Load problems
    print(f"Loading problems from {rtllm_dir}...")
    problem_dict = load_rtllm_problems(rtllm_dir, problems=problems)

    if not problem_dict:
        print("ERROR: No problems found!")
        return

    print(f"Found {len(problem_dict)} problems: {list(problem_dict.keys())}")
    print()

    results = []

    for problem_id, problem in problem_dict.items():
        print(f"Testing: {problem_id}")
        print(f"  Category: {problem.category}")
        print(f"  Reference: {problem.reference_path}")

        # Read reference Verilog
        if not os.path.isfile(problem.reference_path):
            print(f"  Reference file not found, skipping")
            print()
            continue

        with open(problem.reference_path, 'r') as f:
            verilog_code = f.read()

        # Extract actual module name from Verilog (may be "verified_xxx")
        import re
        module_match = re.search(r'^\s*module\s+(\w+)', verilog_code, re.MULTILINE)
        if module_match:
            actual_module_name = module_match.group(1)
        else:
            actual_module_name = problem.design_name

        # Create temp work dir
        work_dir = tempfile.mkdtemp(prefix=f"yosys_bench_{problem_id}_")

        # Run synthesis with timing
        start_time = time.time()
        try:
            synth_result = run_yosys_synth(
                verilog_code=verilog_code,
                design_name=actual_module_name,
                work_dir=work_dir,
                timeout_s=120.0,
                liberty_file=liberty_file,
            )
            elapsed = time.time() - start_time

            if synth_result["success"]:
                area = synth_result["area"]
                delay = synth_result.get("delay", 0.0)
                adp = synth_result.get("adp", 0.0)
                wire_count = synth_result.get("wire_count", 0)
                print(f"  Success in {elapsed:.2f}s")
                if delay > 0:
                    print(f"    Area: {area:.1f} \u03bcm\u00b2, Delay: {delay:.1f} ps, ADP: {adp:.1f}")
                else:
                    print(f"    Area: {area} cells, {wire_count} wires")

                results.append({
                    "problem": problem_id,
                    "time": elapsed,
                    "area": area,
                    "delay": delay,
                    "adp": adp,
                    "wires": wire_count,
                    "success": True,
                })
            else:
                print(f"  Failed in {elapsed:.2f}s")
                print(f"    Error: {synth_result['log'][:200]}")

                results.append({
                    "problem": problem_id,
                    "time": elapsed,
                    "success": False,
                })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Exception in {elapsed:.2f}s: {e}")
            results.append({
                "problem": problem_id,
                "time": elapsed,
                "success": False,
            })

        # Cleanup
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

        print()

    # Summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total problems: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        times = [r["time"] for r in successful]
        areas = [r["area"] for r in successful]

        print("Synthesis Time Statistics:")
        print(f"  Min:    {min(times):.2f}s")
        print(f"  Max:    {max(times):.2f}s")
        print(f"  Mean:   {sum(times)/len(times):.2f}s")
        print(f"  Median: {sorted(times)[len(times)//2]:.2f}s")
        print()

        # Check if we have tech-mapped results
        has_delay = any(r.get("delay", 0) > 0 for r in successful)

        if has_delay:
            print("Area Statistics (\u03bcm\u00b2):")
            print(f"  Min:    {min(areas):.1f}")
            print(f"  Max:    {max(areas):.1f}")
            print(f"  Mean:   {sum(areas)/len(areas):.1f}")
            print()

            delays = [r["delay"] for r in successful if r.get("delay", 0) > 0]
            if delays:
                print("Delay Statistics (ps):")
                print(f"  Min:    {min(delays):.1f}")
                print(f"  Max:    {max(delays):.1f}")
                print(f"  Mean:   {sum(delays)/len(delays):.1f}")
                print()

            adps = [r["adp"] for r in successful if r.get("adp", 0) > 0]
            if adps:
                print("ADP Statistics (Area x Delay):")
                print(f"  Min:    {min(adps):.1f}")
                print(f"  Max:    {max(adps):.1f}")
                print(f"  Mean:   {sum(adps)/len(adps):.1f}")
                print()
        else:
            print("Area Statistics (cells):")
            print(f"  Min:    {min(areas)} cells")
            print(f"  Max:    {max(areas)} cells")
            print(f"  Mean:   {sum(areas)/len(areas):.1f} cells")
            print(f"  Median: {sorted(areas)[len(areas)//2]} cells")
            print()

        # Percentile recommendations
        p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else max(times)
        p99 = sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else max(times)

        print("Recommended timeout_s values:")
        print(f"  Conservative (99th percentile): {p99 * 1.5:.1f}s")
        print(f"  Balanced (95th percentile):     {p95 * 1.5:.1f}s")
        mean_t = sum(times) / len(times)
        std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        print(f"  Aggressive (mean + 2\u03c3):         {mean_t + 2 * std_t:.1f}s")
        print()

        # Detailed results
        print("Detailed Results:")
        if has_delay:
            print(f"{'Problem':<20} {'Time (s)':<10} {'Area (\u03bcm\u00b2)':<14} {'Delay (ps)':<12} {'ADP':<14}")
            print("-" * 70)
            for r in successful:
                print(f"{r['problem']:<20} {r['time']:<10.2f} {r['area']:<14.1f} {r.get('delay', 0):<12.1f} {r.get('adp', 0):<14.1f}")
        else:
            print(f"{'Problem':<20} {'Time (s)':<10} {'Area':<10} {'Wires':<10}")
            print("-" * 50)
            for r in successful:
                print(f"{r['problem']:<20} {r['time']:<10.2f} {r['area']:<10} {r.get('wires', 'N/A'):<10}")

    if failed:
        print()
        print("Failed Problems:")
        for r in failed:
            print(f"  - {r['problem']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Yosys synthesis time")
    parser.add_argument("--rtllm-dir", default="RTLLM", help="RTLLM directory")
    parser.add_argument("--problems", default="adder_8bit,adder_16bit,counter_12,fsm,edge_detect",
                        help="Comma-separated problem list")
    parser.add_argument("--liberty-file", default="recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib",
                        help="Path to Liberty timing library for tech-mapped synthesis")

    args = parser.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    lib_file = args.liberty_file if os.path.isfile(args.liberty_file) else None

    benchmark_yosys(args.rtllm_dir, problems, liberty_file=lib_file)
