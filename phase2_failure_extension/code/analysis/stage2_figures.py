"""Stage 2 reconnaissance figures: stress-strain curves grouped by design variable, and a compact
metric summary per group (strength, failure strain, work, localisation) as bar charts."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.campaign_analysis import curve, M, save, FIG
plt.rcParams.update({"font.size": 8.5})

GROUPS = [("hierarchy ladder (φ=0.20, 120 Å cells)", lambda r: "hierarchy" in (r.get("tags") or [])),
          ("disorder (φ=0.20)", lambda r: "disorder" in (r.get("tags") or [])),
          ("anisotropy / load path", lambda r: any(t in (r.get("tags") or []) for t in ["anisotropy", "loadpath"])),
          ("porosity sweep + lattice orientation", lambda r: any(t in (r.get("tags") or []) for t in ["porosity", "orientation"]) or r["name"] == "S1_mesh_p16_phi0.2"),
          ("gradients and flaw halo", lambda r: any(t in (r.get("tags") or []) for t in ["gradient"]) or r["name"] == "S1_hole_d20"),
          ("defect organisation", lambda r: "defects" in (r.get("tags") or []))]


def main():
    recs = [r for r in db.all_records() if r.get("status") == "completed" and r.get("stage") in ("stage2_reconnaissance", "stage1_baselines")]
    if not recs:
        print("no runs"); return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    summary = {}
    for ax, (title, sel) in zip(axes.ravel(), GROUPS):
        rs = sorted([r for r in recs if sel(r) and (r["stage"] == "stage2_reconnaissance" or r["name"] in ("S1_mesh_p16_phi0.2", "S1_hole_d20"))], key=lambda r: r["name"])
        for i, r in enumerate(rs):
            e, s, nb = curve(r)
            ax.plot(e, s, "-", lw=1.2, color=plt.cm.tab20(i % 20), label=f"{r['name'].replace('S2_','').replace('S1_','')} ({M(r,'strength_Nm'):.1f} N/m, εf {M(r,'failure_strain'):.3f})")
            summary.setdefault(title, {})[r["name"]] = {k: M(r, k) for k in ["strength_Nm", "failure_strain", "first_damage_strain", "work_to_failure_J_m2", "damage_localization", "fracture_mode", "modulus_2d_Nm", "post_peak_load_retention"]}
        ax.set_title(title, fontsize=9); ax.set_xlabel("engineering strain"); ax.set_ylabel("2D stress (N/m)"); ax.grid(alpha=0.3); ax.legend(fontsize=5.5, ncol=1)
    save(fig, "stage2_grouped_curves", "Stage 2 reconnaissance: stress-strain curves grouped by design variable (hierarchy ladder at matched porosity, disorder, anisotropy/load-path organisation, porosity sweep and lattice orientation, gradients and flaw halo, defect organisation). Legends give the strength and failure strain of each individual simulation.")
    json.dump(summary, open(os.path.join(FIG, "stage2_summary.json"), "w"), indent=1)
    # bar summary
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    rs = sorted([r for r in recs if r["stage"] == "stage2_reconnaissance"], key=lambda r: (r.get("tags") or ["z"])[0] + r["name"])
    names = [r["name"].replace("S2_", "") for r in rs]; x = np.arange(len(rs))
    tags = [(r.get("tags") or ["?"])[0] for r in rs]; tagset = sorted(set(tags)); cols = [plt.cm.Set2(tagset.index(t) % 8) for t in tags]
    for ax, (k, lab) in zip(axes, [("strength_Nm", "strength (N/m)"), ("failure_strain", "failure strain"), ("work_to_failure_J_m2", "work to failure (J/m²)"), ("damage_localization", "damage localisation L")]):
        ax.bar(x, [M(r, k) for r in rs], color=cols); ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y")
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(names, rotation=75, fontsize=7)
    for t in tagset:
        axes[0].bar([], [], color=plt.cm.Set2(tagset.index(t) % 8), label=t)
    axes[0].legend(fontsize=7, ncol=len(tagset))
    save(fig, "stage2_metric_bars", "Stage 2 reconnaissance: strength, failure strain, work to failure and damage localisation of every simulated design, coloured by design-variable group.")
    print("stage2 figures written; groups:", {k: len(v) for k, v in summary.items()})


if __name__ == "__main__":
    main()
