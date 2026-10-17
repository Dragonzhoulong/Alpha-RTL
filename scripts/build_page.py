"""Build the Alpha-RTL project page from verified run data.

Usage: python scripts/build_page.py results/c910 site/template.html index.html [--standalone]

results/c910/<task>/ holds verify.json, trajectory.json, lineage.json, reference.v and
discovered.v, as written by scripts/export_run.py. --standalone wraps the page in a full
HTML document for GitHub Pages.
"""
import json
import math
import os
import sys

TASKS = [
    {
        "key": "lza_sky130", "module": "ct_vfmau_lza_simd_half", "short": "LZA", "pdk": "Sky130",
        "label": "24-bit leading-zero anticipator, SIMD FP16 FMA",
        "desc": "The LZA predicts where the leading one of an FMA sum lands before the add finishes, so "
                "normalization can start early. The spec fixes the pre-encoding logic; the model may only "
                "restructure the 24-bit priority encoder that follows it.",
        "insight": "C910 decodes lza_precod with one flat 24-way casez. Alpha-RTL splits it into six 4-bit "
                   "groups, reuses C910's own ct_vfmau_lza_42 cell for each group's valid bit and position, "
                   "and selects the first valid group.",
    },
    {
        "key": "lza_nangate45", "module": "ct_vfmau_lza_simd_half", "short": "LZA", "pdk": "Nangate45",
        "label": "Same LZA, Nangate 45 nm library",
        "desc": "The same LZA optimized in a separate run against the Nangate 45 nm open cell library. "
                "Reward comes from that library's area and timing, so the run tunes to it from scratch.",
        "insight": "This run reaches the same grouped idea by a different route: six 4-bit groups whose "
                   "valid bits are plain OR-reductions, each with a small 4-way priority mux.",
    },
    {
        "key": "lza_gf180", "module": "ct_vfmau_lza_simd_half", "short": "LZA", "pdk": "GF180",
        "label": "Same LZA, GF180MCU 180 nm library",
        "desc": "The same LZA optimized against GlobalFoundries' 180 nm GF180MCU library, where cells are "
                "slower and larger and the area–delay trade-off shifts.",
        "insight": "On GF180 the gain is smaller but still large: the grouped encoder pays off even when "
                   "every gate costs more delay.",
    },
    {
        "key": "fadd_sky130", "module": "ct_fadd_close_s0_d", "short": "FADD close", "pdk": "Sky130",
        "label": "FP64 adder close path, stage 0",
        "desc": "When two FP64 exponents differ by at most one, this stage computes A−B and B−A on 53-bit "
                "mantissas and predicts the leading-one position of the result. The spec fixes the "
                "subtractors and pre-encoding; the model may restructure the find-first-one encoder.",
        "insight": "C910 finds the first one with a single 51-way casez that writes both the binary and the "
                   "one-hot outputs. Alpha-RTL uses a two-level encoder over 16-bit groups and derives the "
                   "one-hot output by shifting the binary index.",
    },
]


def delta(a, b):
    return "%+.1f%%" % (100 * (a / b - 1))


def load(results, spec):
    d = os.path.join(results, spec["key"])
    v = json.load(open(os.path.join(d, "verify.json")))
    assert v["checks"]["formal_equivalence"], f"{spec['key']}: discovered design is not formally proven"
    ref = {"area": v["reference"]["area_um2"], "delay": v["reference"]["delay_ps"]}
    ref["adp"] = ref["area"] * ref["delay"]
    disc = {"area": v["discovered"]["area_um2"], "delay": v["discovered"]["delay_ps"]}
    disc["adp"] = disc["area"] * disc["delay"]

    traj = [{"r": t["adp_ratio"], "s": t["n_syntax_only"], "f": t["n_fail"]}
            for t in json.load(open(os.path.join(d, "trajectory.json")))]
    lineage = [{"step": m["step"], "area": m["area"], "delay": m["delay"], "adp": m["adp"], "code": m["code"]}
               for m in json.load(open(os.path.join(d, "lineage.json"))) if m["improved"]]
    # The page shows the re-measured final design, not the training-time numbers.
    lineage[-1].update(area=disc["area"], delay=disc["delay"], adp=disc["adp"],
                       code=open(os.path.join(d, "discovered.v")).read())

    lo = math.floor((disc["adp"] / ref["adp"] - 0.05) * 10) / 10
    hi = 1.6
    step = 0.2 if hi - lo > 0.7 else 0.1
    ticks = [round(lo + i * step, 2) for i in range(int(round((hi - lo) / step)) + 1)]

    insight = spec["insight"] + " Area %s, delay %s." % (delta(disc["area"], ref["area"]), delta(disc["delay"], ref["delay"]))
    return {**{k: spec[k] for k in ("key", "module", "short", "pdk", "label", "desc")}, "insight": insight,
            "ref": ref, "disc": disc, "traj": traj, "lineage": lineage, "yRange": [lo, hi], "yTicks": ticks,
            "refCode": open(os.path.join(d, "reference.v")).read()}


def main():
    results, template, out = sys.argv[1:4]
    data = {"tasks": [load(results, s) for s in TASKS]}
    html = open(template).read().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    if "--standalone" in sys.argv:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + html.replace("<header", "</head>\n<body>\n<header", 1) + "\n</body>\n</html>\n")
    open(out, "w").write(html)
    print(out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
