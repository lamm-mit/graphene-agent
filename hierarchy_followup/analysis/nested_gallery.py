"""Gallery of every nested (hierarchical) design of the campaign with its measured strength and work to failure.
Dark blue: coarse-level solid (veins / bars); mid blue: sub-veins (three-level design); light blue: fine level.
Writes figures/hierarchy/nested_designs.{png,svg,pdf}.

    python analysis/nested_gallery.py
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
ITEMS = [
    ("H1_veinsX_r0", "two levels, 120 Å (level ratio 3)\n12 Å solid veins every 40 Å, elongated fine pores"),
    ("H4_veinsX_W16_r0", "two levels, 120 Å (level ratio 3)\n16 Å solid veins every 40 Å, round fine pores"),
    ("H4_veinsX_W20_r0", "two levels, 120 Å (level ratio 3)\n20 Å solid veins every 40 Å, round fine pores"),
    ("H3_2L_r0", "two levels, 300 Å (level ratio 8)\n30 Å solid veins every 100 Å, elongated fine pores"),
    ("H5_VS_R_r0", "two levels, 300 Å\n30 Å solid veins, round-pore domains"),
    ("H5_VF_E_r0", "two levels, composite\nfibrous veins (slit bundles), elongated pores"),
    ("H5_VF_R_r0", "two levels, composite\nfibrous veins, round pores"),
    ("H5_VE_R_r0", "two levels, composite\nveins = aligned-pore mesh, round-pore domains"),
    ("H5_VR_E_r0", "two levels, composite\nveins = round-pore mesh, elongated-pore domains"),
    ("H5_GR_0_r0", "two levels, composite grid\nsquare grid whose bars are a round-pore mesh"),
    ("H5_GFx_0_r0", "two levels, composite grid\nbars along the load = slit bundles, other bars solid"),
    ("H5_V3_E_r0", "THREE levels, 300 Å\n30 Å veins / 8 Å sub-veins every 33 Å / elongated pores"),
]


def struct(name):
    p = P[name]; params = dict(p["params"])
    if p.get("seed") is not None and p["family"] != "pristine":
        params["seed"] = p["seed"]
    return D.generate(p["family"], **params)


def main():
    own = {r["name"]: r for r in db.own_records()}
    fig, axes = plt.subplots(3, 4, figsize=(15, 12.2))
    for ax, (name, title) in zip(axes.ravel(), ITEMS):
        a = struct(name); d = a.info["design"]; L = (a.cell[0, 0], a.cell[1, 1]); Q = a.positions
        s = 0.6 if L[0] > 200 else 1.8
        bands = vein_bands(d); v = np.array([in_vein(y, bands) for y in Q[:, 1]]) if bands else np.zeros(len(Q), bool)
        if d.get("coarse") == "square":       # bars across the load too
            n = round(L[0] / d["domain"]); sx = L[0] / n
            v |= np.abs(((Q[:, 0] - d.get("vein_offset", 0.0) + sx / 2) % sx) - sx / 2) < d["bar_w"] / 2
        sub = np.zeros(len(Q), bool)
        if d.get("fill_domain") == "subveins_ellipse":
            dp = d["domain_params"]; n2 = round(L[1] / dp["sub_domain"]); sy2 = L[1] / n2
            sub = (np.abs(((Q[:, 1] - d.get("vein_offset", 0.0) + sy2 / 2) % sy2) - sy2 / 2) < dp["sub_w"] / 2) & ~v
        ax.scatter(Q[~v & ~sub, 0], Q[~v & ~sub, 1], s=s, c="#8fb3d9", linewidths=0)
        ax.scatter(Q[sub, 0], Q[sub, 1], s=s, c="#4a72b0", linewidths=0)
        ax.scatter(Q[v, 0], Q[v, 1], s=s, c="#1f3b73", linewidths=0)
        ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        r = own.get(name); m = r["metrics"] if r else {}
        reg2 = own.get(name.replace("_r0", "_r1")); m2 = reg2["metrics"] if reg2 else {}
        num = (f"σ = {m.get('strength_Nm', float('nan')):.1f}" + (f" / {m2['strength_Nm']:.1f}" if m2 else "") + f" N/m,  W = {m.get('work_to_failure_J_m2', float('nan')):.2f}" + (f" / {m2['work_to_failure_J_m2']:.2f}" if m2 else "") + " J/m²") if r else ""
        ax.set_title(f"{title}\n{L[0]:.0f} × {L[1]:.0f} Å, {len(a)} atoms, φ = {a.info['porosity']:.2f};  {num}", fontsize=7.2, loc="left")
    fig.suptitle("Nested designs of the campaign (load along x; dark = coarse-level solid, mid = sub-veins, light = fine level); numbers: registry 0 / registry 1", fontsize=9.5)
    plt.subplots_adjust(top=0.93, bottom=0.01, left=0.01, right=0.99, wspace=0.06, hspace=0.22)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"nested_designs.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig); print("wrote", os.path.join(FIG, "nested_designs.png"))


if __name__ == "__main__":
    main()
