"""Utilities for C910 FPU component optimization problems."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class C910Problem:
    """Represents a single C910 FPU optimization target."""

    problem_id: str
    design_name: str
    description: str
    reference_path: str
    reference_renamed_path: str
    testbench_path: str
    dep_files: list[str] = field(default_factory=list)
    problem_dir: str = ""
    prompt_template: str = ""


def load_c910_problems(
    problems_dir: str,
    problems: list[str] | None = None,
) -> dict[str, C910Problem]:
    """Load C910 FPU optimization problems from the problems/ directory.

    Each subdirectory under problems_dir should contain:
      - spec.yaml: problem metadata
      - reference.v: original implementation
      - reference_renamed.v: golden model with renamed module
      - testbench.v: verification testbench
      - deps/: dependency Verilog files
      - prompt_template.txt: (optional) base prompt template
    """
    problems_dir = os.path.abspath(problems_dir)
    result: dict[str, C910Problem] = {}

    if not os.path.isdir(problems_dir):
        return result

    for entry in sorted(os.listdir(problems_dir)):
        entry_path = os.path.join(problems_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        spec_path = os.path.join(entry_path, "spec.yaml")
        if not os.path.isfile(spec_path):
            continue

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        problem_id = str(spec.get("problem_id", entry))
        if problems is not None and problem_id not in problems:
            continue

        design_name = str(spec.get("design_name", problem_id))
        description = str(spec.get("description", ""))

        reference_path = os.path.join(entry_path, "reference.v")
        reference_renamed_path = os.path.join(entry_path, "reference_renamed.v")
        testbench_path = os.path.join(entry_path, "testbench.v")

        # Collect dependency files
        dep_files: list[str] = []
        deps_spec = spec.get("deps", [])
        for dep_rel in deps_spec:
            dep_path = os.path.join(entry_path, dep_rel)
            if os.path.isfile(dep_path):
                dep_files.append(dep_path)

        # If no deps listed in spec, scan deps/ directory
        if not dep_files:
            deps_dir = os.path.join(entry_path, "deps")
            if os.path.isdir(deps_dir):
                for fname in sorted(os.listdir(deps_dir)):
                    if fname.endswith(".v"):
                        dep_files.append(os.path.join(deps_dir, fname))

        # Load prompt template
        prompt_path = os.path.join(entry_path, "prompt_template.txt")
        prompt_template = ""
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()

        result[problem_id] = C910Problem(
            problem_id=problem_id,
            design_name=design_name,
            description=description,
            reference_path=reference_path,
            reference_renamed_path=reference_renamed_path,
            testbench_path=testbench_path,
            dep_files=dep_files,
            problem_dir=entry_path,
            prompt_template=prompt_template,
        )

    return result


def extract_verilog_code(solution_str: str) -> str:
    """Extract Verilog code from model response.

    Strips ``<think>...</think>`` blocks first so that reasoning tokens
    (which may contain words like "module") are not mistaken for Verilog.
    If the response is an incomplete thinking block with no code, returns
    empty string rather than passing the raw thinking text as Verilog.
    """
    # Strip complete <think>...</think> blocks
    stripped = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    # If only an incomplete <think> block remains (model ran out of tokens mid-thought),
    # remove it and work with whatever follows (usually nothing).
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL).strip()

    fence_patterns = [
        r"```(?:verilog|systemverilog|sv|v)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in fence_patterns:
        matches = re.findall(pattern, stripped, re.DOTALL | re.IGNORECASE)
        if matches:
            return max(matches, key=len).strip()

    if re.search(r"\bmodule\b", stripped):
        match = re.search(r"(module\b.*\bendmodule\b)", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()

    # No code found — return empty instead of raw text
    return ""


def normalize_verilog_for_dedup(code: str) -> str:
    """Normalize Verilog code for deduplication."""
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", " ", code).strip()
    return code


def verilog_dedup_key(code: str) -> str:
    """Return a hash key for deduplication of Verilog code."""
    normalized = normalize_verilog_for_dedup(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_iverilog_compile(
    verilog_code: str,
    problem: C910Problem,
    work_dir: str,
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    """Compile candidate design + reference + testbench with iverilog.

    This compiles both the candidate (as ct_vfmau_mult_compressor) and the
    reference (as ct_vfmau_mult_compressor_ref) together with the testbench.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Write candidate design
    design_path = os.path.join(work_dir, f"{problem.design_name}.v")
    with open(design_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)

    # Copy reference (renamed), testbench, and deps
    shutil.copy2(problem.reference_renamed_path, os.path.join(work_dir, "reference_renamed.v"))
    shutil.copy2(problem.testbench_path, os.path.join(work_dir, "testbench.v"))
    for dep_path in problem.dep_files:
        shutil.copy2(dep_path, os.path.join(work_dir, os.path.basename(dep_path)))

    # Build iverilog command — include all .v files
    verilog_files = [f"{problem.design_name}.v", "reference_renamed.v", "testbench.v"]
    for dep_path in problem.dep_files:
        verilog_files.append(os.path.basename(dep_path))

    cmd = ["iverilog", "-o", "simv", "-g2005-sv"] + verilog_files

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        compile_log = result.stdout + result.stderr
        simv_path = os.path.join(work_dir, "simv")
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
    timeout_s: float = 60.0,
) -> tuple[bool, str]:
    """Run simulation with vvp and check for 'Your Design Passed'."""
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
        passed = "your design passed" in sim_output.lower()
        if not passed and not sim_output.strip():
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
    dep_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run Yosys synthesis for area/delay estimation.

    Same as rtl_utils.run_yosys_synth but without param_overrides (C910
    modules don't use testbench-driven parameterization).

    Args:
        dep_files: Paths to dependency Verilog files (submodules) that must
            be included in synthesis for the design to elaborate correctly.
    """
    os.makedirs(work_dir, exist_ok=True)

    top_module = design_name
    all_modules = re.findall(r"\bmodule\s+(\w+)", verilog_code)
    if all_modules and design_name not in all_modules:
        top_module = all_modules[0]

    design_path = os.path.join(work_dir, f"{design_name}.v")
    with open(design_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)

    # Copy dependency files into work_dir
    dep_basenames: list[str] = []
    for dep_path in dep_files or []:
        if os.path.isfile(dep_path):
            basename = os.path.basename(dep_path)
            shutil.copy2(dep_path, os.path.join(work_dir, basename))
            dep_basenames.append(basename)

    use_liberty = bool(liberty_file) and os.path.isfile(liberty_file)
    if use_liberty:
        liberty_file = os.path.abspath(liberty_file)

    script_path = os.path.join(work_dir, "synth.tcl")
    with open(script_path, "w", encoding="utf-8") as f:
        for dep_name in dep_basenames:
            f.write(f"read_verilog {dep_name}\n")
        f.write(f"read_verilog {design_name}.v\n")
        f.write(f"synth -top {top_module}\n")
        if use_liberty:
            f.write(f"dfflibmap -liberty {liberty_file}\n")
            f.write(f"abc -liberty {liberty_file}\n")
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
                area_match = re.search(r"Chip area for (?:top )?module\b.*?:\s*([\d.]+)", line)
                if area_match:
                    candidate = float(area_match.group(1))
                    if candidate > chip_area:
                        chip_area = candidate
                delay_match = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", line)
                if delay_match:
                    delay_ps = float(delay_match.group(1))

            cell_match = re.match(r"\s*Number of cells:\s+(\d+)", line)
            if cell_match:
                cell_count = int(cell_match.group(1))
            wire_match = re.match(r"\s*Number of wires:\s+(\d+)", line)
            if wire_match:
                wire_count = int(wire_match.group(1))

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
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False,
                "log": f"Yosys timed out after {timeout_s}s.", "top_module": top_module, "netlist_path": ""}
    except FileNotFoundError:
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False,
                "log": "yosys not found.", "top_module": top_module, "netlist_path": ""}
    except Exception as e:
        return {"area": 0, "wire_count": 0, "delay": 0.0, "adp": 0.0, "success": False,
                "log": f"Yosys error: {e}", "top_module": top_module, "netlist_path": ""}


def parse_compile_error_score(compile_log: str) -> float:
    """Parse iverilog compile log and return a continuous score in (0, 1)."""
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
        return 0.5
    score = 1.0 / (1.0 + n_errors)
    if has_port_error:
        score *= 0.3
    return score


def parse_sim_pass_ratio(sim_output: str) -> float:
    """Extract partial pass ratio from testbench simulation output.

    Supports:
      - "X / Y passed" pattern from our testbench
      - "X / Y failures" pattern
      - "error = X" pattern
    """
    if "your design passed" in sim_output.lower():
        return 1.0

    # Our testbench outputs: "Test completed: X / Y passed"
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*passed", sim_output)
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return passed / total

    # Pattern: "X / Y failures"
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*failures", sim_output)
    if m:
        failures, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return max(0.0, 1.0 - failures / total)

    # Pattern: "error = X"
    m = re.search(r"error\s*[=:]\s*(\d+)", sim_output, re.IGNORECASE)
    if m:
        errors = int(m.group(1))
        return 1.0 / (1.0 + errors) if errors > 0 else 1.0

    if "[SIM_TIMEOUT]" in sim_output:
        return 0.05
    if "[SIM_SILENT_EXIT" in sim_output:
        return 0.02
    return 0.0


# Re-export run_opensta_timing from rtl_utils to avoid duplication
def run_opensta_timing(
    netlist_path: str,
    top_module: str,
    liberty_file: str,
    timeout_s: float = 30.0,
    opensta_binary: str = "sta",
    clk_period_ns: float = 10.0,
) -> dict[str, Any]:
    """Run OpenSTA static timing analysis. Delegates to rtl_utils."""
    from recipe.ttt_rtl.rtl_utils import run_opensta_timing as _run_opensta_timing

    return _run_opensta_timing(
        netlist_path=netlist_path,
        top_module=top_module,
        liberty_file=liberty_file,
        timeout_s=timeout_s,
        opensta_binary=opensta_binary,
        clk_period_ns=clk_period_ns,
    )
