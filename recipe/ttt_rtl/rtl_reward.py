"""Hierarchical reward function for TTT-Discover RTL generation.

Reward levels:
  Level 1 — Syntax: compiles with iverilog
  Level 2 — Functional: passes testbench simulation
  Level 3 — PPA: Yosys synthesis area estimate (optional)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from recipe.ttt_rtl.rtl_utils import (
    extract_tb_param_overrides,
    extract_verilog_code,
    parse_compile_error_score,
    parse_sim_pass_ratio,
    run_iverilog_compile,
    run_iverilog_sim,
    run_opensta_timing,
    run_yosys_synth,
)


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
        "ttt_ppa_power": 0.0,
    }


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,  # noqa: ARG001
    extra_info: dict | None = None,
    timeout_s: float = 30.0,
    w_syntax: float = 0.2,
    w_func: float = 1.0,
    w_ppa: float = 0.0,
    ppa_baseline: float = 0.0,
    failure_reward: float = 0.0,
    liberty_file: str | None = None,
    opensta_binary: str | None = None,
    opensta_timeout_s: float = 30.0,
    clk_period_ns: float = 10.0,
    include_power: bool = False,
    power_activity: float | None = None,
    power_duty: float = 0.5,
    power_clock_from_delay: bool = False,
    power_clock_margin: float = 1.25,
    **kwargs,
) -> dict[str, Any]:
    """Compute hierarchical reward for RTL generation.

    Args:
        data_source: Data source identifier (unused, kept for verl interface).
        solution_str: Model's full response string.
        ground_truth: Ground truth (unused for RTL).
        extra_info: Dict with ttt_problem_id, ttt_testbench_path, ttt_design_name, etc.
        timeout_s: Timeout for each EDA tool invocation.
        w_syntax: Weight for syntax correctness reward.
        w_func: Weight for functional correctness reward.
        w_ppa: Weight for PPA quality reward.
        ppa_baseline: Reference PPA metric (ADPP when include_power is on, else
            ADP). When > 0, PPA reward is baseline/observed (ratio-based, unit-
            consistent). When 0, falls back to a scale-matched normalizer.
        failure_reward: Reward for complete failure (no valid Verilog).
        liberty_file: Optional path to a Liberty timing library for technology-mapped
            synthesis with real area (μm²) and delay (ps). When provided, PPA reward
            is based on ADP (Area × Delay Product).
        opensta_binary: Path to the OpenSTA executable (e.g.
            ``/path/to/OpenSTA22/bin/sta``). When set, OpenSTA replaces ABC's
            rough ``stime`` estimate with accurate static timing analysis.
            Requires *liberty_file* and *w_ppa* > 0.
        opensta_timeout_s: Timeout for OpenSTA STA run in seconds.
        clk_period_ns: Target clock period (ns) for OpenSTA setup-timing
            analysis. Sequential designs meeting this period receive WNS ≥ 0.
        include_power: When True, fold OpenSTA power into the PPA metric —
            ``ppa_score = area × delay × power_w`` (A·D·P). Requires
            *opensta_binary* and *liberty_file*; falls back to A×D if OpenSTA
            power reporting fails. *ppa_baseline* must be in the same A·D·P
            units for the ratio-based reward to be meaningful.

    Returns:
        Dict with score, ttt_* fields for state pool integration.
    """
    extra_info = extra_info or {}
    parent_state_id = str(extra_info.get("ttt_parent_state_id", ""))
    problem_id = str(extra_info.get("ttt_problem_id", ""))
    testbench_path = str(extra_info.get("ttt_testbench_path", ""))
    design_name = str(extra_info.get("ttt_design_name", ""))
    data_files = extra_info.get("ttt_data_files", [])
    # Per-problem baseline from dataset (auto-computed from reference); overrides config
    per_problem_baseline = float(extra_info.get("ttt_ppa_baseline", 0))
    if per_problem_baseline > 0:
        ppa_baseline = per_problem_baseline
    if isinstance(data_files, str):
        data_files = [f.strip() for f in data_files.split(",") if f.strip()]

    # Extract Verilog code from model response
    verilog_code = extract_verilog_code(solution_str)
    if not verilog_code.strip():
        return _failure_payload(
            parent_state_id=parent_state_id,
            code=verilog_code,
            msg="No Verilog code found in response.",
            failure_reward=failure_reward,
        )

    if not testbench_path or not os.path.isfile(testbench_path):
        return _failure_payload(
            parent_state_id=parent_state_id,
            code=verilog_code,
            msg=f"Testbench not found: {testbench_path}",
            failure_reward=failure_reward,
        )

    # Create temp working directory
    work_dir = tempfile.mkdtemp(prefix=f"ttt_rtl_{problem_id}_")
    feedback_parts: list[str] = []

    try:
        # Level 1: Syntax — compile with iverilog
        syntax_ok, compile_log = run_iverilog_compile(
            verilog_code=verilog_code,
            testbench_path=testbench_path,
            work_dir=work_dir,
            design_name=design_name,
            data_files=data_files,
            timeout_s=timeout_s,
        )
        if syntax_ok:
            r_syntax = 1.0
        else:
            r_syntax = parse_compile_error_score(compile_log)

        if not syntax_ok:
            feedback_parts.append("Compilation FAILED.")
            if compile_log:
                # Truncate long logs
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
                "ttt_success": True,  # syntax_ok=False but we still add to pool as stepping stone
                "ttt_syntax_ok": False,
                "ttt_func_ok": False,
                "ttt_ppa_score": 0.0,
                "ttt_ppa_area": 0.0,
                "ttt_ppa_delay": 0.0,
                "ttt_ppa_power": 0.0,
            }

        feedback_parts.append("Compilation PASSED.")

        # Level 2: Functional — simulate with vvp
        func_ok, sim_output = run_iverilog_sim(
            work_dir=work_dir,
            timeout_s=timeout_s,
        )
        if func_ok:
            r_func = 1.0
        else:
            r_func = parse_sim_pass_ratio(sim_output)

        if func_ok:
            feedback_parts.append("Functional test PASSED.")
        else:
            feedback_parts.append("Functional test FAILED.")
            if sim_output:
                sim_text = sim_output if len(sim_output) <= 1000 else sim_output[:1000] + "\n...(truncated)"
                feedback_parts.append(f"Simulation output:\n{sim_text}")

        # Level 3: PPA — Yosys synthesis (only if functional and w_ppa > 0)
        r_ppa = 0.0
        ppa_score = 0.0
        ppa_area = 0.0
        ppa_delay = 0.0
        ppa_power = 0.0
        if func_ok and w_ppa > 0:
            if not liberty_file or not os.path.isfile(liberty_file):
                raise ValueError(
                    f"w_ppa>0 requires a valid liberty_file for tech-mapped synthesis; "
                    f"got liberty_file={liberty_file!r}."
                )
            # Extract testbench parameter overrides so synthesis uses the same
            # parameterization as the functional simulation.
            tb_param_overrides = extract_tb_param_overrides(testbench_path, design_name)
            synth_result = run_yosys_synth(
                verilog_code=verilog_code,
                design_name=design_name,
                work_dir=work_dir,
                timeout_s=timeout_s,
                liberty_file=liberty_file,
                param_overrides=tb_param_overrides or None,
            )
            if synth_result["success"]:
                area = synth_result["area"]
                delay = synth_result.get("delay", 0.0)
                adp = synth_result.get("adp", 0.0)

                ppa_area = float(area)
                ppa_delay = float(delay)

                # --- OpenSTA: replace ABC stime with accurate STA ---
                use_opensta = (
                    opensta_binary
                    and synth_result.get("netlist_path")
                    and liberty_file
                    and os.path.isfile(liberty_file)
                )
                if use_opensta:
                    sta_result = run_opensta_timing(
                        netlist_path=synth_result["netlist_path"],
                        top_module=synth_result.get("top_module", design_name),
                        liberty_file=liberty_file,
                        timeout_s=opensta_timeout_s,
                        opensta_binary=opensta_binary,
                        clk_period_ns=clk_period_ns,
                        power_activity=power_activity,
                        power_duty=power_duty,
                        power_clock_from_delay=power_clock_from_delay,
                        power_clock_margin=power_clock_margin,
                    )
                    if sta_result["success"] and sta_result["delay_ps"] > 0:
                        sta_delay = sta_result["delay_ps"]
                        adp = float(area) * sta_delay
                        ppa_delay = sta_delay
                        wns = sta_result["wns_ps"]
                        meets = sta_result["meets_timing"]
                        ppa_power = float(sta_result.get("power_w", 0.0))
                        feedback_parts.append(
                            f"OpenSTA: delay={sta_delay:.1f}ps, WNS={wns:.1f}ps"
                            f"{'  [TIMING MET]' if meets else '  [TIMING VIOLATED]'}"
                        )
                    else:
                        # OpenSTA failed — fall back to ABC delay already in adp
                        feedback_parts.append("OpenSTA failed, using ABC timing.")

                # Canonical PPA metric: ADPP (A*D*P) when include_power is on,
                # else ADP. If the canonical metric can't be computed (OpenSTA
                # failed or power report failed), r_ppa is capped at a small
                # area-monotonic fallback \u2014 strictly below any normal primary
                # reward so the policy doesn't prefer pipeline-failing designs
                # over correctly-measured ones.
                #
                # No-baseline normalizers are calibrated to Nangate45/RTLLM
                # typical values so r_ppa \u2248 0.5 for a typical design and
                # \u2248 0.9 for an excellent one.
                if include_power and adp > 0 and ppa_power > 0:
                    adpp = adp * ppa_power
                    ppa_score = adpp
                    if ppa_baseline > 0:
                        r_ppa = float(ppa_baseline) / adpp
                    else:
                        r_ppa = 1.0 / (1.0 + adpp / 0.1)
                    feedback_parts.append(
                        f"Synthesis: area={area:.1f}\u03bcm\u00b2, delay={ppa_delay:.1f}ps, "
                        f"power={ppa_power * 1e6:.3f}\u03bcW, ADPP={adpp:.3e}"
                    )
                elif (not include_power) and adp > 0:
                    ppa_score = adp
                    if ppa_baseline > 0:
                        r_ppa = float(ppa_baseline) / adp
                    else:
                        r_ppa = 1.0 / (1.0 + adp / 1e3)
                    feedback_parts.append(
                        f"Synthesis: area={area:.1f}\u03bcm\u00b2, delay={ppa_delay:.1f}ps, "
                        f"ADP={adp:.1f}"
                    )
                else:
                    # Pipeline-incomplete partial credit.
                    if area > 0:
                        ppa_score = float(area)
                        r_ppa = 0.05 / (1.0 + area / 20.0)
                    missing = "delay" if adp <= 0 else "power"
                    feedback_parts.append(
                        f"Synthesis: area={area:.1f}\u03bcm\u00b2 "
                        f"(PPA pipeline incomplete: missing {missing}; partial credit)."
                    )
            else:
                feedback_parts.append("Yosys synthesis failed.")

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
            "ttt_ppa_power": ppa_power,
        }

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
