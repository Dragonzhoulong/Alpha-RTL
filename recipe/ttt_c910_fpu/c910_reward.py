"""Hierarchical reward function for C910 FPU component optimization.

Reward levels:
  Level 1 — Syntax: compiles with iverilog (candidate + reference + testbench)
  Level 2 — Functional: passes golden-model comparison testbench
  Level 3 — PPA: Yosys synthesis area estimate (candidate only)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from recipe.ttt_c910_fpu.c910_utils import (
    extract_verilog_code,
    parse_compile_error_score,
    parse_sim_pass_ratio,
    run_iverilog_compile,
    run_iverilog_sim,
    run_opensta_timing,
    run_yosys_synth,
    C910Problem,
    load_c910_problems,
)


# Cache loaded problems to avoid re-reading YAML on every reward call
_problems_cache: dict[str, C910Problem] | None = None


def _get_problems(problems_dir: str) -> dict[str, C910Problem]:
    global _problems_cache
    if _problems_cache is None:
        _problems_cache = load_c910_problems(problems_dir)
    return _problems_cache


def _failure_payload(parent_state_id: str, code: str, msg: str, failure_reward: float) -> dict[str, Any]:
    return {
        "score": float(failure_reward),
        "ttt_reward": float(failure_reward),
        "ttt_raw_score": None,
        "ttt_result_construction": "",
        "ttt_parent_state_id": parent_state_id,
        "ttt_code": code,
        "ttt_msg": msg,
        "ttt_success": False,
        "ttt_syntax_ok": False,
        "ttt_func_ok": False,
        "ttt_ppa_score": 0.0,
        "ttt_ppa_area": 0.0,
        "ttt_ppa_delay": 0.0,
    }


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,  # noqa: ARG001
    extra_info: dict | None = None,
    timeout_s: float = 30.0,
    sim_timeout_s: float = 120.0,
    w_syntax: float = 0.2,
    w_func: float = 1.0,
    w_ppa: float = 10.0,
    ppa_baseline: float = 0.0,
    failure_reward: float = 0.0,
    liberty_file: str | None = None,
    opensta_binary: str | None = None,
    opensta_timeout_s: float = 30.0,
    clk_period_ns: float = 10.0,
    problems_dir: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Compute hierarchical reward for C910 FPU component optimization.

    Args:
        solution_str: Model's full response string.
        extra_info: Dict with ttt_problem_id, ttt_parent_state_id, etc.
        timeout_s: Timeout for compilation/synthesis.
        sim_timeout_s: Timeout for simulation (longer because golden comparison
            runs many test vectors).
        w_syntax: Weight for syntax correctness reward.
        w_func: Weight for functional correctness reward.
        w_ppa: Weight for PPA quality reward.
        ppa_baseline: Reference ADP. When > 0, PPA reward = baseline/actual.
        failure_reward: Reward for complete failure.
        liberty_file: Path to Liberty timing library for tech-mapped synthesis.
        opensta_binary: Path to OpenSTA executable.
        problems_dir: Path to the problems/ directory.

    Returns:
        Dict with score, ttt_* fields for state pool integration.
    """
    extra_info = extra_info or {}
    parent_state_id = str(extra_info.get("ttt_parent_state_id", ""))
    problem_id = str(extra_info.get("ttt_problem_id", ""))
    per_problem_baseline = float(extra_info.get("ttt_ppa_baseline", 0))
    if per_problem_baseline > 0:
        ppa_baseline = per_problem_baseline

    # Resolve problems directory
    if not problems_dir:
        problems_dir = os.path.join(os.path.dirname(__file__), "problems")
    problems = _get_problems(problems_dir)
    problem = problems.get(problem_id)
    if problem is None:
        return _failure_payload(parent_state_id, "", f"Unknown problem: {problem_id}", failure_reward)

    # Extract Verilog code from model response
    verilog_code = extract_verilog_code(solution_str)
    if not verilog_code.strip():
        return _failure_payload(parent_state_id, verilog_code, "No Verilog code found.", failure_reward)

    work_dir = tempfile.mkdtemp(prefix=f"ttt_c910_{problem_id}_")
    feedback_parts: list[str] = []

    try:
        # Level 1: Syntax — compile candidate + reference + testbench
        syntax_ok, compile_log = run_iverilog_compile(
            verilog_code=verilog_code,
            problem=problem,
            work_dir=work_dir,
            timeout_s=timeout_s,
        )
        if syntax_ok:
            r_syntax = 1.0
        else:
            r_syntax = parse_compile_error_score(compile_log)

        if not syntax_ok:
            feedback_parts.append("Compilation FAILED.")
            if compile_log:
                log_text = compile_log if len(compile_log) <= 1000 else compile_log[:1000] + "\n...(truncated)"
                feedback_parts.append(f"Compiler output:\n{log_text}")

            reward = w_syntax * r_syntax
            return {
                "score": reward,
                "ttt_reward": reward,
                "ttt_raw_score": reward,
                "ttt_result_construction": verilog_code,
                "ttt_parent_state_id": parent_state_id,
                "ttt_code": verilog_code,
                "ttt_msg": "\n".join(feedback_parts),
                "ttt_success": True,
                "ttt_syntax_ok": False,
                "ttt_func_ok": False,
                "ttt_ppa_score": 0.0,
                "ttt_ppa_area": 0.0,
                "ttt_ppa_delay": 0.0,
            }

        feedback_parts.append("Compilation PASSED.")

        # Level 2: Functional — run golden-model comparison testbench
        func_ok, sim_output = run_iverilog_sim(
            work_dir=work_dir,
            timeout_s=sim_timeout_s,
        )
        if func_ok:
            r_func = 1.0
        else:
            r_func = parse_sim_pass_ratio(sim_output)

        if func_ok:
            feedback_parts.append("Functional test PASSED (matches reference).")
        else:
            feedback_parts.append("Functional test FAILED (output mismatch with reference).")
            if sim_output:
                sim_text = sim_output if len(sim_output) <= 1000 else sim_output[:1000] + "\n...(truncated)"
                feedback_parts.append(f"Simulation output:\n{sim_text}")

        # Level 3: PPA — Yosys synthesis (candidate only, only if functional)
        r_ppa = 0.0
        ppa_score = 0.0
        ppa_area = 0.0
        ppa_delay = 0.0
        if func_ok and w_ppa > 0:
            synth_dir = os.path.join(work_dir, "synth")
            synth_result = run_yosys_synth(
                verilog_code=verilog_code,
                design_name=problem.design_name,
                work_dir=synth_dir,
                timeout_s=timeout_s,
                liberty_file=liberty_file,
                dep_files=problem.dep_files,
            )
            if synth_result["success"]:
                area = float(synth_result["area"])
                delay = float(synth_result.get("delay", 0.0))

                # OpenSTA for accurate timing
                if (
                    opensta_binary
                    and synth_result.get("netlist_path")
                    and os.path.isfile(synth_result.get("netlist_path", ""))
                    and liberty_file
                ):
                    sta = run_opensta_timing(
                        netlist_path=synth_result["netlist_path"],
                        top_module=synth_result.get("top_module", problem.design_name),
                        liberty_file=liberty_file,
                        timeout_s=opensta_timeout_s,
                        opensta_binary=opensta_binary,
                        clk_period_ns=clk_period_ns,
                    )
                    if sta["success"] and sta["delay_ps"] > 0:
                        delay = sta["delay_ps"]

                ppa_area = area
                ppa_delay = delay
                adp = area * delay if (area > 0 and delay > 0) else area
                ppa_score = adp if adp > 0 else area

                if ppa_baseline > 0 and ppa_score > 0:
                    r_ppa = ppa_baseline / ppa_score
                elif ppa_score > 0:
                    r_ppa = 1.0 / (1.0 + ppa_score / 100.0)

                if area > 0 and delay > 0:
                    feedback_parts.append(
                        f"Synthesis: area={area:.1f}\u03bcm\u00b2, delay={delay:.1f}ps, ADP={adp:.1f}"
                    )
                elif area > 0:
                    feedback_parts.append(f"Synthesis: {int(area)} cells")
            else:
                feedback_parts.append("Synthesis FAILED.")

        reward = w_syntax * r_syntax + w_func * r_func + w_ppa * r_ppa

        return {
            "score": reward,
            "ttt_reward": reward,
            "ttt_raw_score": reward,
            "ttt_result_construction": verilog_code,
            "ttt_parent_state_id": parent_state_id,
            "ttt_code": verilog_code,
            "ttt_msg": "\n".join(feedback_parts),
            "ttt_success": True,
            "ttt_syntax_ok": syntax_ok,
            "ttt_func_ok": func_ok,
            "ttt_ppa_score": ppa_score,
            "ttt_ppa_area": ppa_area,
            "ttt_ppa_delay": ppa_delay,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
