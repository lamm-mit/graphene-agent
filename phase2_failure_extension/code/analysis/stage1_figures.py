"""Stage 1 baseline figures: stress-strain overlay of the baselines and flaw sensitivity (strength vs
crack length with a Griffith-type sigma ~ a^-1/2 fit, plus hole and vacancy comparisons)."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.campaign_analysis import curve, M, save, FIG, FAMCOL
plt.rcParams.update({"font.size": 9})


def main():
    recs = [r for r in db.all_records() if r.get("status") == "completed" and r.get("stage") == "stage1_baselines"]
    if not recs:
        print("no stage1 runs yet"); return
    by = {r["name"]: r for r in recs}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for r in sorted(recs, key=lambda r: r["name"]):
        e, s, nb = curve(r)
        ax[0].plot(e, s, "-", lw=1.3, label=f"{r['name'].replace('S1_','')} (σmax={M(r,'strength_Nm'):.1f}, εf={M(r,'failure_strain'):.3f})")
    ax[0].set_xlabel("engineering strain"); ax[0].set_ylabel("2D stress σxx (N/m)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=6.5); ax[0].set_title("Stage 1 baselines", fontsize=9)
    # flaw sensitivity
    cr = [(r["design"]["crack_len"], M(r, "strength_Nm"), M(r, "failure_strain")) for r in recs if r["family"] == "precrack"]
    if len(cr) >= 2:
        cr.sort(); a = np.array([c[0] for c in cr]); s = np.array([c[1] for c in cr])
        ax[1].plot(a, s, "o", color=FAMCOL["precrack"], label="central slit crack (2a = crack length)")
        p = np.polyfit(np.log(a), np.log(s), 1)
        aa = np.linspace(a.min() * 0.8, a.max() * 1.2, 50); ax[1].plot(aa, np.exp(p[1]) * aa ** p[0], "--", color=FAMCOL["precrack"], label=f"power-law fit σ ∝ a^{p[0]:.2f} (Griffith: −0.5)")
        s0 = M(by["S1_pristine_zz"], "strength_Nm") if "S1_pristine_zz" in by else None
        if s0: ax[1].axhline(s0, color="k", lw=0.8, ls=":", label=f"pristine zigzag {s0:.1f} N/m")
        if "S1_hole_d20" in by: ax[1].plot(20, M(by["S1_hole_d20"], "strength_Nm"), "s", color=FAMCOL["ring_around_hole"], label="circular hole d=20 Å")
        for nm, mk in [("S1_vac_random_2pct", "^"), ("S1_vac_clustered_2pct", "v")]:
            if nm in by: ax[1].plot(3.0 if "random" in nm else 8.0, M(by[nm], "strength_Nm"), mk, color="0.4", label=nm.replace("S1_", "") + " (x = nominal defect size)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("flaw size (Å)"); ax[1].set_ylabel("strength (N/m)"); ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=6.5); ax[1].set_title("flaw sensitivity", fontsize=9)
    save(fig, "stage1_baselines", "Stage 1 baselines. Left: stress-strain curves of pristine zigzag/armchair graphene, distributed and clustered vacancies (2%), central cracks of 10/20/30 A, a 20 A circular hole and the reference nanomesh. Right: strength versus flaw size for the cracks with a power-law fit (a Griffith-type scaling would give an exponent of -0.5), compared with the hole and the vacancy structures.")
    json.dump({r["name"]: {k: M(r, k) for k in ["modulus_2d_Nm", "strength_Nm", "strain_at_peak", "first_damage_strain", "failure_strain", "work_to_failure_J_m2", "damage_localization", "fracture_mode"]} for r in recs}, open(os.path.join(FIG, "stage1_summary.json"), "w"), indent=1)
    print("stage1 figure written")


if __name__ == "__main__":
    main()
