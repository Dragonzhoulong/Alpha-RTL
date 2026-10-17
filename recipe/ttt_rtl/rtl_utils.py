"""Utilities for RTLLM problem loading, Verilog extraction, and EDA tool helpers."""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RTLLMProblem:
    """Represents a single RTLLM benchmark problem."""

    problem_id: str
    category: str
    design_description: str
    testbench_path: str
    reference_path: str
    design_name: str
    problem_dir: str
    data_files: list[str] = field(default_factory=list)


def load_rtllm_problems(
    rtllm_dir: str,
    problems: list[str] | None = None,
) -> dict[str, RTLLMProblem]:
    """Walk the RTLLM directory tree and load problem metadata.

    Args:
        rtllm_dir: Path to the RTLLM root directory.
        problems: Optional list of problem IDs to filter to. If None, load all.

    Returns:
        Dict mapping problem_id -> RTLLMProblem.
    """
    rtllm_dir = os.path.abspath(rtllm_dir)
    result: dict[str, RTLLMProblem] = {}

    # Walk 3-level structure: Category/Subcategory/problem_dir/
    for category_name in sorted(os.listdir(rtllm_dir)):
        category_path = os.path.join(rtllm_dir, category_name)
        if not os.path.isdir(category_path) or category_name.startswith("_"):
            continue

        for subcategory_name in sorted(os.listdir(category_path)):
            subcategory_path = os.path.join(category_path, subcategory_name)
            if not os.path.isdir(subcategory_path):
                continue

            for problem_name in sorted(os.listdir(subcategory_path)):
                problem_path = os.path.join(subcategory_path, problem_name)
                if not os.path.isdir(problem_path):
                    continue

                makefile_path = os.path.join(problem_path, "makefile")
                desc_path = os.path.join(problem_path, "design_description.txt")
                testbench_path = os.path.join(problem_path, "testbench.v")

                if not all(os.path.isfile(p) for p in [makefile_path, desc_path, testbench_path]):
                    continue

                design_name = _extract_design_name(makefile_path)
                if not design_name:
                    design_name = problem_name

                if problems is not None and design_name not in problems and problem_name not in problems:
                    continue

                with open(desc_path, "r", encoding="utf-8") as f:
                    design_description = f.read().strip()

                reference_path = os.path.join(problem_path, f"verified_{design_name}.v")
                if not os.path.isfile(reference_path):
                    # Try to find any verified_*.v file
                    verified_files = glob.glob(os.path.join(problem_path, "verified_*.v"))
                    reference_path = verified_files[0] if verified_files else ""

                # Collect data files (.txt, .dat, .hex) that testbenches may need via $readmemh
                data_files: list[str] = []
                for fname in sorted(os.listdir(problem_path)):
                    fpath = os.path.join(problem_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if fname == "design_description.txt":
                        continue
                    if fname.endswith((".txt", ".dat", ".hex")):
                        data_files.append(fpath)

                problem_id = design_name
                result[problem_id] = RTLLMProblem(
                    problem_id=problem_id,
                    category=f"{category_name}/{subcategory_name}",
                    design_description=design_description,
                    testbench_path=testbench_path,
                    reference_path=reference_path,
                    design_name=design_name,
                    problem_dir=problem_path,
                    data_files=data_files,
                )

    return result


def _extract_design_name(makefile_path: str) -> str:
    """Extract TEST_DESIGN variable from makefile."""
    try:
        with open(makefile_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^\s*TEST_DESIGN\s*=\s*(\S+)", line)
                if match:
                    return match.group(1).strip()
    except Exception:
        pass
    return ""


def extract_verilog_code(solution_str: str) -> str:
    """Extract Verilog code from model response.

    Looks for markdown fences (```verilog, ```systemverilog, ```v, ```)
    or detects raw Verilog by module keyword.
    """
    # Try markdown fences first
    fence_patterns = [
        r"```(?:verilog|systemverilog|sv|v)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in fence_patterns:
        matches = re.findall(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the longest match (most likely the full module)
            return max(matches, key=len).strip()

    # Fallback: detect raw Verilog by module keyword
    if re.search(r"\bmodule\b", solution_str):
        # Try to extract from first 'module' to last 'endmodule'
        match = re.search(r"(module\b.*\bendmodule\b)", solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    return solution_str.strip()


def _extract_balanced_parens(text: str, start: int) -> str:
    """Return content inside balanced parentheses whose opening '(' is at *start*."""
    if start >= len(text) or text[start] != "(":
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return ""


def extract_tb_param_overrides(testbench_path: str, design_name: str) -> dict[str, str]:
    """Parse named parameter overrides from a testbench's DUT instantiation.

    Looks for ``design_name #(.P1(v1), .P2(v2)) inst_name (`` and returns a
    ``{param_name: value_str}`` dict.  Returns ``{}`` when no parameterized
    instantiation is found or the file cannot be read.

    Args:
        testbench_path: Absolute path to the testbench Verilog file.
        design_name: The DUT module name to search for.

    Returns:
        Dict mapping parameter name to its value string (e.g. ``{"N": "8"}``).
    """
    try:
        with open(testbench_path, "r", encoding="utf-8") as f:
            tb_code = f.read()
        # Strip comments so they don't interfere with the regex
        tb_code = re.sub(r"//.*$", "", tb_code, flags=re.MULTILINE)
        tb_code = re.sub(r"/\*.*?\*/", "", tb_code, flags=re.DOTALL)

        # Find: design_name #(
        m = re.search(r"\b" + re.escape(design_name) + r"\s*#\s*\(", tb_code)
        if not m:
            return {}

        # Position of the opening '(' of the #(...) block
        paren_start = m.end() - 1
        params_str = _extract_balanced_parens(tb_code, paren_start)
        if not params_str:
            return {}

        overrides: dict[str, str] = {}
        for pm in re.finditer(r"\.(\w+)\s*\(([^)]+)\)", params_str):
            overrides[pm.group(1)] = pm.group(2).strip()

        # Resolve TB-level parameter references: when a value is a bare identifier
        # (e.g. .DATA_WIDTH(DATA_WIDTH)) rather than a numeric literal, look it up
        # in the testbench's own `parameter X = <literal>;` declarations.
        tb_params: dict[str, str] = {}
        for pm in re.finditer(r"\bparameter\b(?:\s+\w+)?\s+(\w+)\s*=\s*([^;,\n]+)", tb_code):
            tb_params[pm.group(1).strip()] = pm.group(2).strip()
        for k, v in list(overrides.items()):
            if re.fullmatch(r"[A-Za-z_]\w*", v) and v in tb_params:
                overrides[k] = tb_params[v]

        return overrides
    except Exception:
        return {}


def normalize_verilog_for_dedup(code: str) -> str:
    """Normalize Verilog code for deduplication.

    Strips comments, collapses whitespace.
    """
    # Remove single-line comments
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    # Remove block comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Collapse whitespace
    code = re.sub(r"\s+", " ", code).strip()
    return code


def verilog_dedup_key(code: str) -> str:
    """Return a hash key for deduplication of Verilog code."""
    normalized = normalize_verilog_for_dedup(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_iverilog_compile(
    verilog_code: str,
    testbench_path: str,
    work_dir: str,
    design_name: str,
    data_files: list[str] | None = None,
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    """Compile Verilog design + testbench with Icarus Verilog.

    Args:
        verilog_code: The Verilog source code to compile.
        testbench_path: Path to the testbench file.
        work_dir: Working directory for compilation.
        design_name: Name of the design module.
        data_files: List of data file paths to copy into work_dir.
        timeout_s: Compilation timeout in seconds.

    Returns:
        (success, compile_log) tuple.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Write design file
    design_path = os.path.join(work_dir, f"{design_name}.v")
    with open(design_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)

    # Copy testbench
    tb_dest = os.path.join(work_dir, "testbench.v")
    shutil.copy2(testbench_path, tb_dest)

    # Copy data files
    if data_files:
        for data_file in data_files:
            if os.path.isfile(data_file):
                shutil.copy2(data_file, os.path.join(work_dir, os.path.basename(data_file)))

    simv_path = os.path.join(work_dir, "simv")
    cmd = ["iverilog", "-o", "simv", "-g2005-sv", f"{design_name}.v", "testbench.v"]

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        compile_log = result.stdout + result.stderr
        success = os.path.isfile(simv_path) and result.returncode == 0
        return success, compile_log.strip()
    except subprocess.TimeoutExpired:
        return False, f"Compilation timed out after {timeout_s}s."
    except FileNotFoundError:
        return False, "iverilog not found. Please install Icarus Verilog."
    except Exception as e:
        return False, f"Compilation error: {e}"


def run_iverilog_sim(
    work_dir: str,
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    """Run simulation with vvp.

    Args:
        work_dir: Working directory containing the compiled 'simv'.
        timeout_s: Simulation timeout in seconds.

    Returns:
        (passed, sim_output) tuple. passed is True if output contains
        "Your Design Passed".  When the simulation fails, the output string
        is enriched with metadata tags so that ``parse_sim_pass_ratio`` can
        distinguish different failure modes (timeout, crash, silent exit).
    """
    simv_path = os.path.join(work_dir, "simv")
    if not os.path.isfile(simv_path):
        return False, "No compiled simulation binary (simv) found."

    try:
        result = subprocess.run(
            ["vvp", "simv"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        sim_output = result.stdout + result.stderr
        # RTLLM testbenches print "Your Design Passed" on success
        passed = "your design passed" in sim_output.lower()
        if not passed and not sim_output.strip():
            # Silent exit — enrich with returncode so parse_sim_pass_ratio
            # can distinguish this from a crash or normal error output.
            sim_output = f"[SIM_SILENT_EXIT returncode={result.returncode}]"
        return passed, sim_output.strip()
    except subprocess.TimeoutExpired:
        return False, f"[SIM_TIMEOUT] Simulation timed out after {timeout_s}s."
    except FileNotFoundError:
        return False, "vvp not found. Please install Icarus Verilog."
    except Exception as e:
        return False, f"Simulation error: {e}"


def run_yosys_synth(
    verilog_code: str,
    design_name: str,
    work_dir: str,
    timeout_s: float = 60.0,
    liberty_file: str | None = None,
    param_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run Yosys synthesis for area/delay estimation.

    When *liberty_file* points to a valid Liberty (.lib) file the synthesis is
    **technology-mapped** against that standard-cell library, producing real
    physical area (μm²) and critical-path delay (ps).  Without it the function
    falls back to the original generic (technology-independent) synthesis that
    only reports abstract cell/wire counts.

    Args:
        verilog_code: The Verilog source code.
        design_name: Top module name (used as fallback filename).
        work_dir: Working directory.
        timeout_s: Synthesis timeout in seconds.
        liberty_file: Optional path to a Liberty timing library (e.g.
            ``sky130_fd_sc_hd__tt_025C_1v80.lib``).

    Returns:
        Dict with keys: area, wire_count, delay, adp, success, log.
        When technology-mapped, *area* is physical area in μm² (float),
        *delay* is critical-path delay in ps (float), and *adp* is their
        product.  In generic mode *area* is the abstract cell count (int),
        *delay* and *adp* are 0.0.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Determine top module name. For multi-module files (e.g. merged reference
    # implementations), prefer design_name if it appears as a module declaration.
    # Fall back to the first module found in the file for single-module LLM outputs.
    top_module = design_name
    all_modules = re.findall(r"\bmodule\s+(\w+)", verilog_code)
    if all_modules:
        if design_name not in all_modules:
            # design_name not present — use first module (single-module LLM output)
            top_module = all_modules[0]
        # else: design_name found → keep top_module = design_name

    design_path = os.path.join(work_dir, f"{design_name}.v")
    with open(design_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)

    # Decide whether to use technology-mapped or generic synthesis
    use_liberty = bool(liberty_file) and os.path.isfile(liberty_file)
    if use_liberty:
        liberty_file = os.path.abspath(liberty_file)

    # Write Yosys TCL script
    script_path = os.path.join(work_dir, "synth.tcl")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(f"read_verilog -sv {design_name}.v\n")
        if param_overrides:
            for pname, pval in param_overrides.items():
                f.write(f"chparam -set {pname} {pval}\n")
        f.write(f"synth -top {top_module}\n")
        if use_liberty:
            f.write(f"dfflibmap -liberty {liberty_file}\n")
            # Map with ABC (no stime -p: it fails on large designs and OpenSTA
            # provides accurate timing via run_opensta_timing anyway).
            f.write(f"abc -liberty {liberty_file}\n")

            # Write gate-level netlist for downstream STA (e.g. OpenSTA)
            f.write("write_verilog -noattr gate_netlist.v\n")
            f.write(f"stat -liberty {liberty_file}\n")
        else:
            f.write("stat\n")

    try:
        result = subprocess.run(
            ["yosys", "-s", "synth.tcl"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = result.stdout + result.stderr

        cell_count = 0
        wire_count = 0
        chip_area = 0.0
        delay_ps = 0.0

        for line in log.splitlines():
            if use_liberty:
                # Technology-mapped: parse physical area from stat -liberty
                # Pattern: "Chip area for module '\fp_mul': 19666.361600"
                # Match both "Chip area for module" and "Chip area for top module".
                # Prefer the top-module value (includes hierarchy); take the maximum
                # so that multi-module merged files report the full flattened area.
                area_match = re.search(r"Chip area for (?:top )?module\b.*?:\s*([\d.]+)", line)
                if area_match:
                    candidate = float(area_match.group(1))
                    if candidate > chip_area:
                        chip_area = candidate

                # Parse delay from ABC stime -p output
                # Pattern: "Delay = 5125.58 ps"
                delay_match = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", line)
                if delay_match:
                    delay_ps = float(delay_match.group(1))

            # Always try to parse generic cell/wire counts (useful even in
            # tech-mapped mode for logging)
            cell_match = re.match(r"\s*Number of cells:\s+(\d+)", line)
            if cell_match:
                cell_count = int(cell_match.group(1))
            cell_match2 = re.match(r"\s*(\d+)\s+cells\s*$", line)
            if cell_match2:
                cell_count = int(cell_match2.group(1))

            wire_match = re.match(r"\s*Number of wires:\s+(\d+)", line)
            if wire_match:
                wire_count = int(wire_match.group(1))
            wire_match2 = re.match(r"\s*(\d+)\s+wires\s*$", line)
            if wire_match2:
                wire_count = int(wire_match2.group(1))

        # Detect critical synthesis warnings that invalidate results
        has_conflicting_drivers = "multiple conflicting drivers" in log

        if use_liberty:
            synth_valid = result.returncode == 0 and chip_area > 0 and not has_conflicting_drivers
            area_val: int | float = chip_area
        else:
            synth_valid = result.returncode == 0 and cell_count > 0 and not has_conflicting_drivers
            area_val = cell_count

        adp = chip_area * delay_ps if (chip_area > 0 and delay_ps > 0) else 0.0
        netlist_path = os.path.join(work_dir, "gate_netlist.v") if use_liberty else ""

        return {
            "area": area_val,
            "wire_count": wire_count,
            "delay": delay_ps,
            "adp": adp,
            "success": synth_valid,
            "log": log.strip(),
            "top_module": top_module,
            "netlist_path": netlist_path if os.path.isfile(netlist_path) else "",
        }
    except subprocess.TimeoutExpired:
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False, "log": f"Yosys timed out after {timeout_s}s.", "top_module": top_module, "netlist_path": ""}
    except FileNotFoundError:
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False, "log": "yosys not found.", "top_module": top_module, "netlist_path": ""}
    except Exception as e:
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False, "log": f"Yosys error: {e}", "top_module": top_module, "netlist_path": ""}


def parse_compile_error_score(compile_log: str) -> float:
    """Parse iverilog compile log and return a continuous score in (0, 1).

    Fewer errors → higher score. Port/module-level errors (indicating the
    generated interface is completely wrong) receive an extra penalty.

    Args:
        compile_log: Raw stdout+stderr from iverilog.

    Returns:
        Float in (0, 1). Only called when compilation failed.
    """
    lines = compile_log.strip().splitlines()
    n_errors = 0
    has_port_error = False
    for line in lines:
        lower = line.lower()
        if "error" in lower:
            n_errors += 1
            if any(kw in lower for kw in ("port", "unknown module", "not a module")):
                has_port_error = True
    if n_errors == 0:
        return 0.5  # log present but unparseable — neutral mid-range value
    score = 1.0 / (1.0 + n_errors)
    if has_port_error:
        score *= 0.3
    return score


def parse_sim_pass_ratio(sim_output: str) -> float:
    """Extract partial pass ratio from RTLLM testbench simulation output.

    Supports two common RTLLM patterns:
      - ``"Test completed with X / Y failures"``  → ``1 - X/Y``
      - ``"error = X"`` (or ``"error=X"``)         → ``1/(1+X)``

    Also handles metadata tags injected by ``run_iverilog_sim``:
      - ``[SIM_TIMEOUT]``      → 0.05 (ran long enough to maybe be doing work)
      - ``[SIM_SILENT_EXIT]``  → 0.02 (compiled but produced nothing)

    Args:
        sim_output: Raw stdout+stderr from vvp simulation.

    Returns:
        Float in [0, 1]. 1.0 means all tests passed; 0.0 means complete
        failure or unparseable output.
    """
    if "your design passed" in sim_output.lower():
        return 1.0
    # Pattern 1: "X / Y failures"
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*failures", sim_output)
    if m:
        failures, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return max(0.0, 1.0 - failures / total)
    # Pattern 2: "error = X" or "error=X" (integer error counter)
    m = re.search(r"error\s*[=:]\s*(\d+)", sim_output, re.IGNORECASE)
    if m:
        errors = int(m.group(1))
        return 1.0 / (1.0 + errors) if errors > 0 else 1.0
    # Pattern 3: metadata tags from run_iverilog_sim
    # Timeout means the design at least did *something* (better than silent exit)
    if "[SIM_TIMEOUT]" in sim_output:
        return 0.05
    # Silent exit means compiled OK but sim produced nothing — slightly above 0
    if "[SIM_SILENT_EXIT" in sim_output:
        return 0.02
    return 0.0


def run_opensta_timing(
    netlist_path: str,
    top_module: str,
    liberty_file: str,
    timeout_s: float = 30.0,
    opensta_binary: str = "sta",
    clk_period_ns: float = 10.0,
    power_activity: float | None = None,
    power_duty: float = 0.5,
    power_clock_from_delay: bool = False,
    power_clock_margin: float = 1.25,
    power_clock_min_ns: float = 0.1,
) -> dict[str, Any]:
    """Run OpenSTA static timing analysis on a gate-level netlist.

    Produces accurate critical-path delay and timing slack, replacing the
    rougher estimate from ABC ``stime -p``.  Works for both combinational
    circuits (no clock port) and sequential circuits (detects common clock
    port names and creates a ``create_clock`` constraint at *clk_period_ns*).

    Args:
        netlist_path: Absolute path to the gate-level Verilog netlist written
            by Yosys (``write_verilog -noattr gate_netlist.v``).
        top_module: Top-level module name.
        liberty_file: Absolute path to the Liberty (.lib) timing library.
        timeout_s: STA timeout in seconds.
        opensta_binary: Path or name of the OpenSTA executable (``sta``).
        clk_period_ns: Clock period used for setup timing analysis (ns).
            Designs that meet timing at this period receive WNS ≥ 0.
        power_activity: When set, inject
            ``set_power_activity -global -activity <a> -duty <d>`` before
            ``report_power`` to pin switching activity. ``None`` preserves
            OpenSTA's default (liberty-driven) behavior.
        power_duty: Duty cycle paired with *power_activity*.
        power_clock_from_delay: When True, run STA in two passes — the first
            measures ``delay_ps`` at *clk_period_ns*, the second re-runs
            ``report_power`` with the clock tightened to
            ``max(delay_ns * power_clock_margin, power_clock_min_ns)``.
            Matches the "clock ≈ critical-path delay" sign-off convention that
            several RTL papers use when reporting μW numbers. Timing/WNS are
            always reported from the first (fixed-clock) pass.
        power_clock_margin: Multiplier on measured delay for the second-pass
            clock period (e.g. 1.25 = 25 % margin).
        power_clock_min_ns: Floor on the second-pass clock period (ns) so we
            never create a 0 ns clock for trivial designs.

    Returns:
        Dict with keys:
          - ``delay_ps``: critical-path arrival time in ps (float).
          - ``wns_ps``: worst negative slack in ps; 0 means timing met (float).
          - ``meets_timing``: True when WNS ≥ 0 (bool).
          - ``success``: False if STA failed or could not parse results (bool).
          - ``log``: raw OpenSTA stdout/stderr (str).
          - ``power_w``: total power in watts. Switching estimate reflects
            *power_activity* when provided, otherwise OpenSTA defaults.
          - ``internal_power_w``, ``switching_power_w``, ``leakage_power_w``:
            per-category breakdown of total power (W).
          - ``power_clock_ns``: the clock period actually used for the power
            report (equals *clk_period_ns* when *power_clock_from_delay* is
            False, else the tightened period).
    """
    _fail: dict[str, Any] = {
        "delay_ps": 0.0,
        "wns_ps": 0.0,
        "meets_timing": False,
        "success": False,
        "log": "",
        "power_w": 0.0,
        "internal_power_w": 0.0,
        "switching_power_w": 0.0,
        "leakage_power_w": 0.0,
        "power_clock_ns": clk_period_ns,
    }

    if not netlist_path or not os.path.isfile(netlist_path):
        return {**_fail, "log": f"Netlist not found: {netlist_path}"}
    if not liberty_file or not os.path.isfile(liberty_file):
        return {**_fail, "log": f"Liberty file not found: {liberty_file}"}

    work_dir = os.path.dirname(netlist_path)
    netlist_abs = os.path.abspath(netlist_path)
    liberty_abs = os.path.abspath(liberty_file)

    def _write_script(script_path: str, clk_ns: float, include_power: bool) -> None:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"read_liberty {liberty_abs}\n")
            f.write(f"read_verilog {netlist_abs}\n")
            f.write(f"link_design {top_module}\n")
            f.write("set has_clk 0\n")
            f.write("foreach clk_name {clk clock CLK clk_i sys_clk clk_in} {\n")
            f.write("    set p [get_ports $clk_name -quiet]\n")
            f.write("    if {[llength $p] > 0} {\n")
            f.write(f"        create_clock -name clk -period {clk_ns} $p\n")
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
            if include_power:
                if power_activity is not None:
                    f.write(
                        f"set_power_activity -global -activity {power_activity}"
                        f" -duty {power_duty}\n"
                    )
                f.write('puts "=== POWER ==="\n')
                f.write("report_power -digits 6\n")
            f.write("exit\n")

    def _invoke(script_path: str) -> tuple[bool, str, dict[str, Any] | None]:
        try:
            result = subprocess.run(
                [opensta_binary, "-no_init", "-exit", script_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return True, result.stdout + result.stderr, None
        except subprocess.TimeoutExpired:
            return False, "", {**_fail, "log": f"OpenSTA timed out after {timeout_s}s."}
        except FileNotFoundError:
            return False, "", {**_fail, "log": f"OpenSTA binary not found: {opensta_binary}"}
        except Exception as e:
            return False, "", {**_fail, "log": f"OpenSTA error: {e}"}

    def _parse_power(lines: list[str]) -> tuple[float, float, float, float]:
        for line in lines:
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
                    return 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0

    # --- Pass 1: timing (and power if single-pass) ---
    single_pass = not power_clock_from_delay
    sta_script = os.path.join(work_dir, "sta_timing.tcl")
    _write_script(sta_script, clk_period_ns, include_power=single_pass)
    ok, log, err = _invoke(sta_script)
    if not ok:
        return err  # type: ignore[return-value]

    # Parse timing from pass-1 log.
    delay_ns = 0.0
    for line in log.splitlines():
        m = re.search(r"([\d.]+)\s+data arrival time", line)
        if m:
            candidate = float(m.group(1))
            if candidate > delay_ns:
                delay_ns = candidate

    wns_ns: float | None = None
    for line in log.splitlines():
        m = re.search(r"[Ww]orst slack\s+([-\d.]+|INF)", line)
        if m:
            raw = m.group(1)
            wns_ns = float("inf") if raw == "INF" else float(raw)
            break

    if delay_ns == 0.0 and wns_ns is None:
        return {**_fail, "log": log.strip()}

    delay_ps = delay_ns * 1000.0

    if wns_ns is None:
        wns_ps = (clk_period_ns - delay_ns) * 1000.0
    elif wns_ns == float("inf"):
        wns_ps = (clk_period_ns - delay_ns) * 1000.0
    else:
        wns_ps = wns_ns * 1000.0

    meets_timing = wns_ps >= 0.0

    # --- Pass 2: re-run power at tightened clock period, if requested ---
    power_log = log
    power_clock_ns = clk_period_ns
    if power_clock_from_delay:
        power_clock_ns = max(delay_ns * power_clock_margin, power_clock_min_ns)
        power_script = os.path.join(work_dir, "sta_power.tcl")
        _write_script(power_script, power_clock_ns, include_power=True)
        ok2, log2, err2 = _invoke(power_script)
        if not ok2:
            return err2  # type: ignore[return-value]
        power_log = log2
        log = log + "\n" + log2

    internal_w, switching_w, leakage_w, total_w = _parse_power(power_log.splitlines())

    return {
        "delay_ps": delay_ps,
        "wns_ps": wns_ps,
        "meets_timing": meets_timing,
        "success": True,
        "log": log.strip(),
        "power_w": total_w,
        "internal_power_w": internal_w,
        "switching_power_w": switching_w,
        "leakage_power_w": leakage_w,
        "power_clock_ns": power_clock_ns,
    }
