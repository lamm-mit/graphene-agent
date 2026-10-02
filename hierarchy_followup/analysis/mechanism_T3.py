"""Mechanism figure for the case where hierarchy paid (T3, 300 A): load carried per atom just before the peak, damage at
the peak and after the first major drop, for the two-level mesh and the two one-level meshes of the same mass.
Writes figures/hierarchy/mechanism_T3.{png,svg,pdf} and the event-selected progression panels of the three uncracked runs.

    python analysis/mechanism_T3.py
"""
from __future__ import annotations
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from experiments import db
from analysis import fracture_viz as FV
from atomistics.descriptors.columns import vein_bands, in_vein

FIG = os.path.join(ROOT, "figures", "hierarchy")
EV = 16.0217663
plt.rcParams.update({"font.family": "Arial" if any("Arial" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "sans-serif", "font.size": 8})
NAMES = [("H3_2L_r0", "two-level: 30 Å solid veins every 100 Å + elongated fine pores"), ("H3_1L_fine_r0", "one-level fine mesh (elongated pores, period 12 Å)"), ("H3_1L_coarse_r0", "one-level coarse mesh (50 Å bars)")]


def bonds(ax, T, k, lw=0.25):
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    if len(b):
        p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]; d = p1 - p0
        for a in range(2):
            d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
        ax.add_collection(LineCollection(np.stack([p0, p0 + d], 1), colors="0.75", linewidths=lw, zorder=1))


def main():
    own = {r["name"]: r for r in db.own_records()}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12.8))
    for row, (name, label) in enumerate(NAMES):
        rec = own[name]; T = FV.load_traj(db.trajectory_path(rec)); eps = T["eps_x"]; sxx = T["sigma_xx"] * EV
        ip = int(np.argmax(sxx)); nb = T["n_broken_cum"]; inc = np.diff(nb, prepend=nb[0])
        k_pre = max(0, int(np.searchsorted(eps, 0.8 * eps[ip])))          # ~80 % of the peak strain, before damage
        post = [k for k in range(ip + 1, len(sxx)) if sxx[k - 1] - sxx[k] > 0.15 * sxx[ip]]
        k_post = post[0] if post else min(ip + 2, len(sxx) - 1)
        # column 1: load carried per atom (per-atom virial xx, tension positive) before the peak
        ax = axes[row, 0]; P = T["positions"][k_pre].astype(float); c = T["cells"][k_pre]
        v = T["peratom_virial"][k_pre][:, 0].astype(float)          # per-atom virial xx, positive in tension; its mean is sigma * A / N
        share = v / v.mean()
        bonds(ax, T, k_pre)
        sc = ax.scatter(P[:, 0], P[:, 1], c=share, s=1.6, cmap="magma", vmin=0, vmax=2.0, linewidths=0, zorder=2)
        ax.set_title(f"{label}\nload share per atom at ε = {eps[k_pre]:.3f} (σ = {sxx[k_pre]:.1f} N/m, before damage)", fontsize=7.5, loc="left")
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01); cb.set_label("per-atom load / mean per-atom load", fontsize=7)
        bands = vein_bands(rec.get("design") or {})
        if bands:
            inv = np.array([in_vein(y, bands) for y in P[:, 1]])
            ax.text(0.02, 0.02, f"veins: 30 % of the section, {v[inv].sum() / v.sum() * 100:.0f} % of the load\n({share[inv].mean():.2f} x mean per atom; fine level {share[~inv].mean():.2f} x)", transform=ax.transAxes, fontsize=6.5, color="white", bbox=dict(facecolor="black", alpha=0.6, pad=2))
        # columns 2-3: damage at the peak and after the first major drop
        for col, k, what in ((1, ip, "peak"), (2, k_post, "after the first major drop")):
            ax = axes[row, col]; dmg = FV.damaged_atoms_until(rec, eps[k])
            sc2, lab = FV.draw_frame(ax, T, k, "energy_rel", damage_atoms=dmg, s=1.2, lw=0.25)
            ax.set_title(f"{what}: ε = {eps[k]:.3f}, σ = {sxx[k]:.1f} N/m, {int(nb[k])} bonds broken", fontsize=7.5, loc="left")
        for ax in axes[row]:
            ax.set_xlim(0, c[0, 0]); ax.set_ylim(0, c[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Where hierarchy paid (T3, 300 Å, porosity 0.20, load along x): load transfer before the peak and damage at and after it", fontsize=9)
    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, wspace=0.08, hspace=0.12)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"mechanism_T3.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)
    out = os.path.join(FIG, "panels")
    for name, _ in NAMES:
        if not os.path.exists(os.path.join(out, f"progression_{name}_energy_rel.png")):
            FV.progression_panel(own[name]["run_id"], out)
    print("wrote mechanism_T3 and the progression panels")


if __name__ == "__main__":
    main()
