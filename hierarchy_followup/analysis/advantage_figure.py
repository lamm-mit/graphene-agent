"""One figure that states the advantage of hierarchy and its limits: the designs side by side (same mass, same cell) and
their strength, work to failure and cracked strength, at 300 A (T3, where hierarchy paid) and at 120 A (T1, where it did
not).  Writes figures/hierarchy/advantage.{png,svg,pdf}.

    python analysis/advantage_figure.py
"""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from atomistics.structures import design_space as D
from atomistics.descriptors.columns import vein_bands, in_vein

FIG = os.path.join(ROOT, "figures", "hierarchy")
plt.rcParams.update({"font.family": "Arial" if any("Arial" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "sans-serif", "font.size": 8})
P = {}
for f in ("hierarchy_followup_predictions.json", "hierarchy_T5_predictions.json"):
    P.update({p["name"]: p for p in json.load(open(os.path.join(ROOT, "experiments", "predictions", f)))["predictions"]})


def struct(name):
    p = P[name]; params = dict(p["params"])
    if p.get("seed") is not None and p["family"] != "pristine":
        params["seed"] = p["seed"]
    return D.generate(p["family"], **params)


def draw(ax, name, title):
    a = struct(name); d = a.info["design"]; L = (a.cell[0, 0], a.cell[1, 1]); Q = a.positions
    bands = vein_bands(d)
    if bands:
        v = np.array([in_vein(y, bands) for y in Q[:, 1]])
        ax.scatter(Q[~v, 0], Q[~v, 1], s=0.5 if L[0] > 200 else 1.4, c="#8fb3d9", linewidths=0); ax.scatter(Q[v, 0], Q[v, 1], s=0.5 if L[0] > 200 else 1.4, c="#1f3b73", linewidths=0)
    else:
        ax.scatter(Q[:, 0], Q[:, 1], s=0.5 if L[0] > 200 else 1.4, c="#5b7db1", linewidths=0)
    ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5, loc="left")


def val(own, names, key):
    xs = [own[n]["metrics"][key] for n in names if n in own]
    return (float(np.mean(xs)), float(np.ptp(xs)) if len(xs) > 1 else 0.0) if xs else (np.nan, 0.0)


def main():
    own = {r["name"]: r for r in db.own_records()}
    fig = plt.figure(figsize=(13, 8.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.9, 1.0], hspace=0.12, wspace=0.35, left=0.04, right=0.99, top=0.92, bottom=0.07)
    # ---- row 1: the designs at both scales
    designs = [("H3_1L_fine_r0", "300 Å, ONE level\nfine pores everywhere"), ("H3_1L_coarse_r0", "300 Å, ONE level\ncoarse bars and holes"), ("H3_2L_r0", "300 Å, TWO levels (hierarchical)\n30 Å veins every 100 Å + fine pores"),
               ("H1_ellipse_r0", "120 Å, ONE level\nfine pores"), ("H1_slit_r0", "120 Å, ONE level\nload-parallel slits"), ("H1_veinsX_r0", "120 Å, TWO levels (hierarchical)\n12 Å veins every 40 Å + fine pores")]
    for k, (n, t) in enumerate(designs):
        draw(fig.add_subplot(gs[0, k]), n, t)
    # ---- row 2: numbers, 300 A (left) and 120 A (right)
    groups = [("300 Å cell (T3), 60 Å crack", [("one-level fine", ["H3_1L_fine_r0", "H3_1L_fine_r1"], ["H3_1L_fine_L60_r0", "H3_1L_fine_L60_r1"]), ("one-level coarse", ["H3_1L_coarse_r0", "H3_1L_coarse_r1"], ["H3_1L_coarse_L60_r0"]),
                                               ("two-level", ["H3_2L_r0", "H3_2L_r1"], ["H3_2L_L60_r0", "H3_2L_L60_r1"])]),
              ("120 Å cell (T1), 20 Å crack", [("one-level fine", ["H1_ellipse_r0", "H1_ellipse_r1"], ["H1_ellipse_L20_r0", "H1_ellipse_L20_r1"]), ("one-level slits", ["H1_slit_r0", "H1_slit_r1"], ["H1_slit_L20_r0", "H1_slit_L20_r1"]),
                                               ("two-level", ["H1_veinsX_r0", "H1_veinsX_r1"], ["H1_veinsX_L20_r0", "H1_veinsX_L20_r1"])])]
    cols = {"one-level fine": "#8fb3d9", "one-level coarse": "#5b7db1", "one-level slits": "#5b7db1", "two-level": "#1f3b73"}
    for g, (title, items) in enumerate(groups):
        ax = fig.add_subplot(gs[1, 3 * g:3 * g + 3])
        metrics = [("strength (N/m)", "strength_Nm", False), ("work to failure (J/m²) × 10", "work_to_failure_J_m2", False), ("strength with the crack (N/m)", "strength_Nm", True)]
        x = np.arange(len(metrics)); w = 0.26
        for i, (lab, unc, cr) in enumerate(items):
            vals, errs = [], []
            for _, key, cracked in metrics:
                m, e = val(own, cr if cracked else unc, key)
                if key == "work_to_failure_J_m2":
                    m, e = 10 * m, 10 * e
                vals.append(m); errs.append(e / 2)
            b = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2, color=cols[lab], label=lab, edgecolor="white", linewidth=0.5)
            for xx, vv in zip(x + (i - 1) * w, vals):
                ax.text(xx, vv + 0.4, f"{vv:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=7.5); ax.set_ylim(0, 37); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_title(title + " — same mass (porosity 0.20), same cell; bars = mean of two registries, whiskers = their spread", fontsize=8, loc="left")
    fig.suptitle("Is the hierarchical design better than a one-level design of the same mass?  Yes at 300 Å (level ratio 8); no at 120 Å (level ratio 3), except under a crack", fontsize=9.5)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"advantage.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig); print("wrote", os.path.join(FIG, "advantage.png"))


if __name__ == "__main__":
    main()
