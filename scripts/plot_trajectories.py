"""Plot ADP / ADP_ref of every functional rollout and the best-so-far curve for each task.

    python scripts/plot_trajectories.py results/c910 docs/assets/trajectories.png
"""
import json
import os
import sys

import matplotlib.pyplot as plt

TASKS = [("lza_sky130", "LZA · Sky130"), ("lza_nangate45", "LZA · Nangate45"),
         ("lza_gf180", "LZA · GF180"), ("fadd_sky130", "FADD close path · Sky130")]
ACCENT, REF, DOT = "#1453c4", "#9a9184", "#15181c"


def main(results, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=False)
    for ax, (key, title) in zip(axes, TASKS):
        traj = json.load(open(os.path.join(results, key, "trajectory.json")))
        v = json.load(open(os.path.join(results, key, "verify.json")))
        xs, ys, best, bx, by = [], [], float("inf"), [], []
        for t in traj:
            xs += [t["step"]] * len(t["adp_ratio"])
            ys += t["adp_ratio"]
            if t["adp_ratio"]:
                best = min(best, t["adp_ratio"][0])
            if best < float("inf"):
                bx.append(t["step"]); by.append(best)
        ax.scatter(xs, ys, s=4, color=DOT, alpha=0.18, linewidths=0)
        ax.step(bx, by, where="post", color=ACCENT, lw=2)
        ax.axhline(1.0, color=REF, ls="--", lw=1.2)
        final = 1 - v["adp_reduction"]
        ax.annotate(f"−{100 * v['adp_reduction']:.1f}%", (bx[-1], final), xytext=(-4, 6),
                    textcoords="offset points", ha="right", color=ACCENT, fontsize=10, fontweight="bold")
        lo = max(0.0, round(final - 0.15, 1))
        ax.set_ylim(lo, 1.6)
        ax.set_xlim(0, len(traj) + 1)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("training step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d7dbd5", lw=0.6)
    axes[0].set_ylabel("ADP / ADP of C910 RTL")
    axes[0].text(2, 1.02, "C910 reference", color=REF, fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
