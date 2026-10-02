"""Stage 8: predicted-versus-observed figure from experiments/holdouts/holdout_evaluation.json."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.campaign_analysis import save


def main():
    p = os.path.join(ROOT, "experiments", "holdouts", "holdout_evaluation.json")
    if not os.path.exists(p):
        print("no evaluation"); return
    E = json.load(open(p))
    keys = [("strength_Nm", "strength (N/m)"), ("failure_strain", "failure strain"), ("work_to_failure_J_m2", "work to failure (J/m²)"), ("damage_localization", "damage localisation L"), ("modulus_2d_Nm", "2D modulus (N/m)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6))
    for ax, (k, lab) in zip(axes, keys):
        xs, ys, names = [], [], []
        for r in E["rows"]:
            if "observed" not in r: continue
            pv, ov = r["predicted"].get(k), r["observed"].get(k)
            if pv is None or ov is None or not np.isfinite(ov): continue
            xs.append(pv); ys.append(ov); names.append(r["name"])
        if xs:
            lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys)); pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
            ax.scatter(xs, ys, s=28, color="#d62728", zorder=3)
            for x, y, n in zip(xs, ys, names):
                ax.annotate(n.replace("S7_", ""), (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
            s = E["summary"].get(k)
            if s: ax.set_title(f"{lab}\nmedian |rel. error| {s['median_abs_rel_error']*100:.0f}% (n={s['n']})", fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.grid(alpha=0.3)
    save(fig, "holdout_prediction_vs_observation", "Holdout evaluation (Stage 8): quantitative predictions recorded before the simulations (x) versus simulated values (y) for the unseen designs; dashed line = perfect prediction. Titles give the median absolute relative error.")
    print("holdout figure written")


if __name__ == "__main__":
    main()
