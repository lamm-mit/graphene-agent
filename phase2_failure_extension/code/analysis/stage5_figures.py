"""Stage 5 deep-study figures: (deep_hierarchy) vein-width and vein-direction series with the coarse and fine
limits; (deep_disorder) Voronoi regularity sweep with seed replicates and jitter series; (deep_scaling)
porosity scaling of three architectures and the net-section master plot sigma/minSF vs ligament width."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.campaign_analysis import M, Dd, save, curve, FAMCOL
plt.rcParams.update({"font.size": 8.5})


def recs_by_name():
    return {r["name"]: r for r in db.all_records() if r.get("status") == "completed" and r.get("campaign") == "discovery"}


def fig_hierarchy(by):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    series = [(4, "S2_H2_D40_W4"), (8, "S2_H2_D40_W8"), (12, "S2_H2_D40_W12"), (16, "S5_H2_D40_W16"), (20, "S5_H2_D40_W20")]
    series = [(w, by[n]) for w, n in series if n in by]
    for a, (k, lab) in zip(ax, [("strength_Nm", "strength (N/m)"), ("work_to_failure_J_m2", "work to failure (J/m²)"), ("n_damage_steps", "number of damage steps")]):
        a.plot([w for w, _ in series], [M(r, k) for _, r in series], "o-", color="#d62728", label="two-level mesh, D=40 Å, vein width W (xy veins)")
        for nm, lab2, col, ls in [("S2_H1_square_D40", "coarse level alone (square pores)", "k", "--"), ("S2_H1_p12", "fine level alone (period 12)", "#1f77b4", ":")]:
            if nm in by: a.axhline(M(by[nm], k), color=col, ls=ls, lw=1, label=lab2)
        for nm, lab2, mk in [("S4_H2_D40_W8_veins_x", "W=8, veins along load only", "s"), ("S4_H2_D40_W8_veins_y", "W=8, veins across load only", "v"), ("S2_H3_D30_120", "three-level (30/120 Å)", "^"), ("S5_H2_p8_D40_W8", "period 8 fine level, W=8", "D"), ("S5_H2_D40_W8_jitter2", "W=8, fine level jittered 2 Å", "P"), ("S5_H2_D40_W8_ellipseX", "W=8, fine pores elongated along load", "X")]:
            if nm in by: a.plot([8 if "W8" in nm or "p8" in nm else 6], [M(by[nm], k)], mk, ms=7, label=lab2)
        a.set_xlabel("vein width W (Å)"); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    save(fig, "deep_hierarchy", "Deep study of hierarchy at matched porosity 0.20: strength, work to failure and number of discrete damage events versus vein width of the two-level mesh (red), compared with the coarse level alone (dashed), the fine level alone (dotted), the vein-direction variants (veins only along / only across the load), the three-level mesh, a finer fine level (period 8), and the hierarchy x disorder and hierarchy x anisotropy interactions.")


def fig_disorder(by):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    regs = {}
    for n, r in by.items():
        if r["family"] == "voronoi_network" and r["params"].get("n_cells") == 25:
            regs.setdefault(r["params"].get("regularity", 0.0), []).append(r)
    for reg in sorted(regs):
        vals = [M(r, "strength_Nm") for r in regs[reg]]; ws = [M(r, "work_to_failure_J_m2") for r in regs[reg]]
        ax[0].plot([reg] * len(vals), vals, "o", color="#2ca02c", alpha=0.7); ax[0].errorbar(reg, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, fmt="k_", ms=12, capsize=3)
        ax[1].plot([reg] * len(ws), ws, "o", color="#2ca02c", alpha=0.7); ax[1].errorbar(reg, np.mean(ws), yerr=np.std(ws, ddof=1) if len(ws) > 1 else 0, fmt="k_", ms=12, capsize=3)
    ax[0].set_xlabel("Voronoi regularity"); ax[0].set_ylabel("strength (N/m)"); ax[0].grid(alpha=0.3); ax[0].set_title("polygonal networks (25 cells): seeds and mean ± sd", fontsize=8)
    ax[1].set_xlabel("Voronoi regularity"); ax[1].set_ylabel("work to failure (J/m²)"); ax[1].grid(alpha=0.3)
    # jitter / size dispersion series with seeds
    groups = {"ordered (p16)": ["S2_H1_p16", "S1_mesh_p16_phi0.2", "S6_mesh_p16_offset1", "S6_mesh_p16_offset2", "S6_mesh_p16_offset3"], "jitter 1 Å": ["S2_mesh_jitter1.0"], "jitter 2 Å": ["S2_mesh_jitter2.0", "S6_mesh_jitter2.0_s2", "S6_mesh_jitter2.0_s3"], "jitter 3 Å": ["S2_mesh_jitter3.0"],
              "size disp. 15%": ["S2_mesh_sizedis0.15", "S6_mesh_sizedis0.15_s2", "S6_mesh_sizedis0.15_s3"], "size disp. 30%": ["S2_mesh_sizedis0.3"]}
    x = 0
    for g, names in groups.items():
        rs = [by[n] for n in names if n in by]
        if not rs: continue
        ax[2].plot([x] * len(rs), [M(r, "strength_Nm") for r in rs], "o", color="#1f77b4"); ax[2].plot([x + 0.25] * len(rs), [10 * M(r, "work_to_failure_J_m2") for r in rs], "s", color="#ff7f0e")
        ax[2].text(x, 5.5, g, rotation=70, fontsize=6.5, ha="center"); x += 1
    ax[2].set_ylim(0, 22); ax[2].set_xticks([]); ax[2].set_ylabel("strength (N/m, circles) / 10 x work (J/m², squares)"); ax[2].grid(alpha=0.3); ax[2].set_title("ordered vs disordered pore lattices (individual runs)", fontsize=8)
    save(fig, "deep_disorder", "Deep study of disorder. Left/middle: strength and work to failure of periodic Voronoi networks versus seed regularity (dots: individual seeds; black: mean ± standard deviation). Right: individual runs of the ordered reference mesh (including lattice-registry replicates) and of jittered / size-dispersed meshes.")


def fig_scaling(by):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fams = {"parallel slits": ["S5_slit0_phi0.1", "S2_slit_0deg", "S5_slit0_phi0.3", "S5_slit0_phi0.4"], "square coarse mesh": ["S5_square_D40_phi0.1", "S2_H1_square_D40", "S5_square_D40_phi0.3"], "round fine mesh (p16)": ["S2_mesh_phi0.1", "S2_H1_p16", "S2_mesh_phi0.3", "S2_mesh_phi0.4"]}
    for (g, names), col in zip(fams.items(), ["#ff7f0e", "k", "#1f77b4"]):
        rs = [by[n] for n in names if n in by]
        ax[0].plot([r.get("porosity") for r in rs], [M(r, "strength_Nm") for r in rs], "o-", color=col, label=g)
        ax[1].plot([Dd(r, "min_solid_fraction_across_x") for r in rs], [M(r, "strength_Nm") for r in rs], "o", color=col, label=g)
    ax[0].set_xlabel("porosity φ"); ax[0].set_ylabel("strength (N/m)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3); ax[0].set_title("porosity scaling of three architectures", fontsize=8)
    xs = np.linspace(0.3, 1, 10)
    for sn, lab, ls in [(31, "σ = 31 × minSF (straight ligaments)", "--"), (17, "σ = 17 × minSF (fine round ligaments)", ":")]:
        ax[1].plot(xs, sn * xs, "k" + ls, lw=0.8, label=lab)
    ax[1].set_xlabel("minimum load-bearing solid fraction"); ax[1].set_ylabel("strength (N/m)"); ax[1].legend(fontsize=6.5); ax[1].grid(alpha=0.3)
    # master plot
    allr = [r for r in by.values() if (r.get("porosity") or 0) > 0.05 and r["family"] not in ("precrack", "ring_around_hole", "vacancies")]
    for f in sorted(set(r["family"] for r in allr)):
        rs = [r for r in allr if r["family"] == f]
        ax[2].scatter([Dd(r, "ligament_width_mean_A") for r in rs], [M(r, "strength_Nm") / Dd(r, "min_solid_fraction_across_x") for r in rs], s=18, color=FAMCOL.get(f, "0.3"), label=f, alpha=0.85)
    ax[2].axhline(38.8, color="k", lw=0.8, ls="--", label="pristine zigzag 38.8 N/m")
    ax[2].set_xlabel("mean ligament width (Å)"); ax[2].set_ylabel("net-section strength σmax / minSF (N/m)"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=6); ax[2].set_title("net-section master plot (porous designs)", fontsize=8)
    save(fig, "deep_scaling", "Scaling and the net-section rule. Left: strength versus porosity for parallel slits (whose net section is set by the slit width, not the slit length), the square coarse mesh and the round fine mesh. Middle: the same data versus the measured minimum load-bearing solid fraction with the two net-section lines. Right: net-section strength (strength divided by minimum solid fraction) versus mean ligament width for all porous designs; straight load-parallel ligaments approach the pristine strength, fine round-pore and random networks fall well below it.")


if __name__ == "__main__":
    by = recs_by_name(); fig_hierarchy(by); fig_disorder(by); fig_scaling(by); print("stage5 figures written")
