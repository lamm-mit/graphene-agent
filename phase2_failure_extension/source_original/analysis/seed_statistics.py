"""Stage 6 seed statistics: individual seeds and summary (mean, std, n) per stochastic design group."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.campaign_analysis import M, save, FIG


def group_key(r):
    return (r.get("tags") or ["?"])[0] if r.get("stage") == "stage6_seeds" else r["name"]


def main():
    allr = {r["name"]: r for r in db.all_records() if r.get("status") == "completed" and r.get("campaign") == "discovery"}
    recs = [r for r in allr.values() if r.get("stage") == "stage6_seeds"]
    if not recs:
        print("no stage6 runs"); return
    groups = {}
    for r in recs:
        g = r.get("notes") or r["name"].rsplit("_s", 1)[0]
        groups.setdefault(g, []).append(r)
    # add the original (seed 1 / ordered) members from Stages 1-2
    originals = {"mesh_jitter2.0": ["S2_mesh_jitter2.0"], "mesh_sizedis0.15": ["S2_mesh_sizedis0.15"], "voronoi_n25_reg0.0": ["S2_voronoi_n25_reg0.0"], "voronoi_n25_reg0.6": ["S2_voronoi_n25_reg0.6"], "vac_random_2pct": ["S1_vac_random_2pct"], "mesh_p16_registry": ["S2_H1_p16", "S1_mesh_p16_phi0.2"]}
    for g, names in originals.items():
        for n in names:
            if n in allr and g in groups:
                groups[g].append(allr[n])
    keys = ["strength_Nm", "failure_strain", "work_to_failure_J_m2", "damage_localization"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    summary = {}
    gnames = sorted(groups)
    for ax, k in zip(axes, keys):
        for i, g in enumerate(gnames):
            vals = np.array([M(r, k) for r in groups[g]], float); vals = vals[np.isfinite(vals)]
            ax.plot(np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(vals)), vals, "o", ms=5, alpha=0.8)
            if len(vals):
                ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, fmt="k_", ms=14, capsize=4, lw=1.5)
                summary.setdefault(g, {})[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": int(len(vals)), "values": vals.tolist()}
        ax.set_xticks(range(len(gnames))); ax.set_xticklabels(gnames, rotation=60, fontsize=7); ax.set_ylabel(k); ax.grid(alpha=0.3)
    save(fig, "seed_statistics", "Random-seed statistics (Stage 6): individual simulations (dots) and mean +- standard deviation (black) of strength, failure strain, work to failure and damage localisation for stochastic architectures replicated with independent seeds.")
    json.dump(summary, open(os.path.join(FIG, "seed_statistics.json"), "w"), indent=1)
    print(json.dumps({g: {k: (round(v["mean"], 3), round(v["std"], 3), v["n"]) for k, v in s.items()} for g, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
