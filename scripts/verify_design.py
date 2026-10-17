"""Verify a C910 candidate design: simulation, re-synthesis, and formal equivalence.

Run from the verl root (so ``recipe.*`` imports resolve) with iverilog, yosys and
OpenSTA on PATH:

    python scripts/verify_design.py --problem lza_simd_half \
        --design results/c910/lza_sky130/discovered.v \
        --liberty recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib

Formal equivalence builds a miter of the candidate and ``reference_renamed.v`` in
Yosys and proves with SAT that the outputs never differ. This catches designs that
pass simulation but synthesize to a different circuit (for example, hierarchical
references that Icarus resolves but Yosys leaves undriven).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())  # run from the verl root
from recipe.ttt_c910_fpu.c910_reward import compute_score  # noqa: E402

PROBLEMS_DIR = "recipe/ttt_c910_fpu/problems"

# Problems whose spec has don't-care outputs need a hand-written miter; ``bad`` must be provably 0.
CUSTOM_MITERS = {
    # ff1_pred / ff1_pred_onehot are don't-care when close_eq = 1 (see spec.yaml).
    "fadd_close_s0_d": """
module chk(input [52:0] a, input [52:0] b, output bad);
  wire e0, e1, c0, c1; wire [52:0] ab0, ab1, ba0, ba1, oh0, oh1; wire [5:0] p0, p1;
  ct_fadd_close_s0_d_ref u0(.close_adder0(a), .close_adder1(b), .close_eq(e0), .close_op_chg(c0),
    .close_sum_a_b(ab0), .close_sum_b_a(ba0), .ff1_pred(p0), .ff1_pred_onehot(oh0));
  ct_fadd_close_s0_d     u1(.close_adder0(a), .close_adder1(b), .close_eq(e1), .close_op_chg(c1),
    .close_sum_a_b(ab1), .close_sum_b_a(ba1), .ff1_pred(p1), .ff1_pred_onehot(oh1));
  assign bad = (e0 != e1) | (c0 != c1) | (ab0 != ab1) | (ba0 != ba1) | (~e0 & ((p0 != p1) | (oh0 != oh1)));
endmodule
""",
}


def design_name(problem):
    spec = open(os.path.join(PROBLEMS_DIR, problem, "spec.yaml")).read()
    return re.search(r"^design_name:\s*(\S+)", spec, re.M).group(1)


def formal_equivalence(problem, design_path, timeout_s=3600):
    """Return True if Yosys proves the design equivalent to the reference, False if not, None on timeout."""
    deps = sorted(glob.glob(os.path.join(PROBLEMS_DIR, problem, "deps", "*.v")))
    ref = os.path.join(PROBLEMS_DIR, problem, "reference_renamed.v")
    top = design_name(problem)
    with tempfile.TemporaryDirectory(prefix="alpha_rtl_equiv_") as work:
        if problem in CUSTOM_MITERS:
            chk = os.path.join(work, "chk.v")
            open(chk, "w").write(CUSTOM_MITERS[problem])
            files = " ".join(deps + [ref, os.path.abspath(design_path), chk])
            body = "hierarchy -top chk\nproc; flatten; opt_clean\nsat -verify -prove bad 0 chk\n"
        else:
            files = " ".join(deps + [ref, os.path.abspath(design_path)])
            body = (f"proc; flatten {top}_ref; flatten {top}; opt_clean\n"
                    f"miter -equiv -flatten -make_outputs {top}_ref {top} miter\n"
                    "hierarchy -top miter\nsat -verify -prove trigger 0 miter\n")
        ys = os.path.join(work, "equiv.ys")
        open(ys, "w").write(f"read_verilog -sv {files}\n{body}")
        try:
            p = subprocess.run(["yosys", "-q", "-s", ys], capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None
        return p.returncode == 0


def measure(problem, code, liberty, opensta="sta"):
    """Simulate and synthesize ``code`` with the training reward pipeline."""
    r = compute_score("c910", f"```verilog\n{code}\n```", "", extra_info={"ttt_problem_id": problem},
                      liberty_file=liberty, opensta_binary=opensta, problems_dir=PROBLEMS_DIR,
                      timeout_s=120, sim_timeout_s=120)
    return {"compiles": r["ttt_syntax_ok"], "simulation": r["ttt_func_ok"],
            "area_um2": r["ttt_ppa_area"], "delay_ps": r["ttt_ppa_delay"],
            "adp": r["ttt_ppa_area"] * r["ttt_ppa_delay"]}


def verify(problem, design_path, liberty, opensta="sta"):
    ref = measure(problem, open(os.path.join(PROBLEMS_DIR, problem, "reference.v")).read(), liberty, opensta)
    cand = measure(problem, open(design_path).read(), liberty, opensta)
    cand["formal_equivalence"] = formal_equivalence(problem, design_path)
    cand["adp_reduction"] = 1 - cand["adp"] / ref["adp"] if ref["adp"] and cand["adp"] else None
    return {"problem": problem, "liberty": os.path.basename(liberty), "reference": ref, "candidate": cand}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--liberty", required=True)
    ap.add_argument("--opensta", default=os.environ.get("OPENSTA_BINARY", "sta"))
    args = ap.parse_args()
    print(json.dumps(verify(args.problem, args.design, args.liberty, args.opensta), indent=2))


if __name__ == "__main__":
    main()
