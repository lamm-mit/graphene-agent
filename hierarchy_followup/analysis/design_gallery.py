"""Gallery of the follow-up designs (generated structures, atoms of load-parallel veins in a darker colour) at a common
physical scale, next to the phase-1 nested design they descend from.  Writes figures/hierarchy/designs_120A.* and
figures/hierarchy/designs_scale.*.

    python analysis/design_gallery.py
"""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from atomistics.structures import design_space as D
from atomistics.descriptors.columns import vein_bands, in_vein

FIG = os.path.join(ROOT, "figures", "hierarchy")
P = {p["name"]: p for p in json.load(open(os.path.join(ROOT, "experiments", "predictions", "hierarchy_followup_predictions.json")))["predictions"]}
plt.rcParams.update({"font.family": "Arial" if any("Arial" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "sans-serif", "font.size": 8})


def build(name):
    if name in P:
        p = P[name]; params = dict(p["params"])
        if p.get("seed") is not None and p["family"] != "pristine":
            params["seed"] = p["seed"]
        return D.generate(p["family"], **params)
    raise KeyError(name)


def draw(ax, atoms, title, s=1.6):
    d = atoms.info["design"]; L = (atoms.cell[0, 0], atoms.cell[1, 1])
    bands = vein_bands(d)
    P_ = atoms.positions
    if bands:
        v = np.array([in_vein(y, bands) for y in P_[:, 1]])
        ax.scatter(P_[~v, 0], P_[~v, 1], s=s, c="#8fb3d9", linewidths=0)
        ax.scatter(P_[v, 0], P_[v, 1], s=s, c="#1f3b73", linewidths=0)
    else:
        ax.scatter(P_[:, 0], P_[:, 1], s=s, c="#5b7db1", linewidths=0)
    ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=7, loc="left")


def label(name):
    a = build(name); d = a.info["design"]; L = (a.cell[0, 0], a.cell[1, 1])
    parts = [f"{name}", f"{L[0]:.0f} × {L[1]:.0f} Å, {len(a)} atoms, φ = {a.info['porosity']:.2f}"]
    if d.get("family") == "nanomesh_hier":
        parts.append(f"veins {d['vein_w']:.0f} Å every {d['domain']:.0f} Å ({d['vein_dirs']}), fine period {d['period']:.0f} Å → level ratio {d['domain'] / d['period']:.1f}")
    elif d.get("family") == "nanomesh_single":
        parts.append(f"one level, period {d['period']:.0f} Å" + (", elongated pores along the load" if d.get("aspect", 1) > 1 else (", square pores" if d.get("shape") == "square" else "")))
    elif d.get("family") == "seam_pores":
        parts.append(f"seams every {d['seam_spacing']:.0f} Å, pores {d['pore_d']:.0f} Å at pitch {d['pitch']:.1f} Å")
    elif d.get("family") == "slit_array":
        parts.append("load-parallel slits")
    if d.get("cracked"):
        parts.append(f"crack {d['crack_len']:.0f} Å perpendicular to the load")
    return a, "\n".join(parts)


def main():
    os.makedirs(FIG, exist_ok=True)
    # (1) the 120 A designs of T1, T4 and T2 at one scale
    names = ["H1_ellipse_r0", "H1_veinsX_r0", "H1_veinsX_L20_r0", "H1_veinsX_L40_r0", "H4_veinsX_W16_r0", "H4_veinsX_W20_r0", "H1_slit_r0", "H2_seamX_s20_r0"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 8.2), gridspec_kw=dict(hspace=0.32, wspace=0.08))
    for ax, n in zip(axes.ravel(), names):
        a, t = label(n); draw(ax, a, t)
    fig.suptitle("Follow-up designs in the 120 Å cell (T1, T4, T2); dark blue: atoms of the load-parallel veins; load along x", fontsize=9)
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"designs_120A.{ext}"), dpi=220 if ext == "png" else None)
    plt.close(fig)
    # (2) scale: phase-1-size two-level mesh next to the 300 A T3 designs at the same physical scale
    names = ["H1_veinsX_r0", "H3_2L_r0", "H3_1L_fine_r0", "H3_1L_coarse_r0", "H3_2L_L60_r0"]
    structs = [label(n) for n in names]
    widths = [s[0].cell[0, 0] for s in structs]
    fig, axes = plt.subplots(1, len(names), figsize=(sum(widths) / 300 * 5.2 + 1.0, 6.6), gridspec_kw=dict(width_ratios=widths, wspace=0.06))
    for ax, (a, t) in zip(axes, structs):
        draw(ax, a, t.split("\n")[0] + "\n" + t.split("\n")[1], s=0.9 if a.cell[0, 0] > 200 else 1.6)
    fig.suptitle("Same physical scale: the phase-1-sized two-level mesh (left) and the 300 Å T3 designs (level ratio 8, eight pore rows per domain)", fontsize=9)
    plt.subplots_adjust(top=0.84, bottom=0.03, left=0.01, right=0.99)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"designs_scale.{ext}"), dpi=220 if ext == "png" else None)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "designs_120A.png"), os.path.join(FIG, "designs_scale.png"))


if __name__ == "__main__":
    main()
