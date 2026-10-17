"""Export a finished TTT-C910 run: best verified design, per-step rollouts, and lineage.

    python scripts/export_run.py --run-dir checkpoints/verl-ttt-c910/<exp> \
        --problem lza_simd_half --liberty recipe/ttt_rtl/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib \
        --out results/c910/my_run

Candidates are tried in ADP order; the first one that Yosys proves equivalent to the
reference is reported. Writes reference.v, discovered.v, verify.json, trajectory.json
(ADP / ADP_ref of every functional rollout per step) and lineage.json (ancestors that
set a new best ADP, with their code) into --out.
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_design import PROBLEMS_DIR, formal_equivalence, measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--liberty", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opensta", default=os.environ.get("OPENSTA_BINARY", "sta"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pool = json.load(open(os.path.join(args.run_dir, "ttt_c910_state_pool", "latest.json")))["states"]
    by_id = {s["state_id"]: s for s in pool}
    ok = sorted([s for s in pool if s["func_ok"] and s["ppa_area"] > 0 and s["ppa_delay"] > 0],
                key=lambda s: s["ppa_area"] * s["ppa_delay"])

    disc_path = os.path.join(args.out, "discovered.v")
    best = None
    for s in ok:
        open(disc_path, "w").write(s["verilog_code"])
        if formal_equivalence(args.problem, disc_path):
            best = s
            break
        print(f"rejected step {s['timestep']}: not formally equivalent", file=sys.stderr)
    if best is None:
        sys.exit("no formally equivalent design in this run")

    shutil.copy(os.path.join(PROBLEMS_DIR, args.problem, "reference.v"), os.path.join(args.out, "reference.v"))
    ref = measure(args.problem, open(os.path.join(args.out, "reference.v")).read(), args.liberty, args.opensta)
    cand = measure(args.problem, best["verilog_code"], args.liberty, args.opensta)

    chain = [by_id[p] for p in reversed(best["parent_ids"]) if p in by_id] + [best]
    lineage, cur = [], float("inf")
    for s in chain:
        adp = s["ppa_area"] * s["ppa_delay"] if s["func_ok"] and s["ppa_area"] > 0 else None
        improved = adp is not None and adp < cur
        cur = adp if improved else cur
        lineage.append({"step": s["timestep"], "func_ok": s["func_ok"], "syntax_ok": s["syntax_ok"],
                        "area": s["ppa_area"], "delay": s["ppa_delay"], "adp": adp,
                        "improved": improved, "code": s["verilog_code"] if improved else None})

    traj = []
    files = glob.glob(os.path.join(args.run_dir, "rollout_samples", "*.jsonl"))
    for f in sorted(files, key=lambda x: int(os.path.basename(x)[:-6])):
        ratios, n_syn, n_fail = [], 0, 0
        for line in open(f):
            r = json.loads(line)
            if r.get("ttt_func_ok") and r.get("ttt_ppa_area", 0) > 0 and r.get("ttt_ppa_delay", 0) > 0:
                ratios.append(round(r["ttt_ppa_area"] * r["ttt_ppa_delay"] / ref["adp"], 4))
            elif r.get("ttt_syntax_ok"):
                n_syn += 1
            else:
                n_fail += 1
        traj.append({"step": int(os.path.basename(f)[:-6]), "adp_ratio": sorted(ratios),
                     "n_syntax_only": n_syn, "n_fail": n_fail})

    verify = {"problem": args.problem, "liberty": os.path.basename(args.liberty),
              "reference": {k: ref[k] for k in ("area_um2", "delay_ps", "adp")},
              "discovered": {**{k: cand[k] for k in ("area_um2", "delay_ps", "adp")}, "found_at_step": best["timestep"]},
              "adp_reduction": 1 - cand["adp"] / ref["adp"],
              "checks": {"simulation": cand["simulation"], "formal_equivalence": True}}
    json.dump(verify, open(os.path.join(args.out, "verify.json"), "w"), indent=2)
    json.dump(lineage, open(os.path.join(args.out, "lineage.json"), "w"))
    json.dump(traj, open(os.path.join(args.out, "trajectory.json"), "w"))
    print(f"{args.out}: ADP {ref['adp']:.0f} -> {cand['adp']:.0f} ({100 * verify['adp_reduction']:.1f}% lower)")


if __name__ == "__main__":
    main()
