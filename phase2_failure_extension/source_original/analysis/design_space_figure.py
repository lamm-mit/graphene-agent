"""Figures 1-3 of the report: (1) reference images -> extracted principles (schematic table already in text),
(2) translation from imagery to atomic carbon architecture (image thumbnail -> principle -> generated structure),
(3) the explored design space: family gallery and hierarchy ladder with descriptors."""
from __future__ import annotations
import sys, os, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
FIG = os.path.join(ROOT, "figures", "design"); os.makedirs(FIG, exist_ok=True)


def specs_by_name():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "experiments", "specs", "stage*", "**", "*.json*"), recursive=True):
        try:
            for s in json.load(open(f))["structures"]:
                out[s["name"]] = s
        except Exception:
            pass
    return out


def draw(ax, atoms, title, s=0.9):
    ax.scatter(atoms.positions[:, 0], atoms.positions[:, 1], s=s, c="k", linewidths=0)
    ax.set_xlim(0, atoms.cell[0, 0]); ax.set_ylim(0, atoms.cell[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5)


def fig_translation(S):
    doc = json.load(open(os.path.join(ROOT, "analysis", "image_interpretation.json")))
    rows = [("img1", "P1 nested hierarchy / P2 loops", "S2_H2_D40_W8", "two-level nested nanomesh"),
            ("img2", "P1/P3/P5 closed coarse walls, scale separation", "S2_H3_D30_120", "three-level nested nanomesh"),
            ("img3", "P6 disorder / P7 tortuous paths", "S2_voronoi_n25_reg0.0", "periodic Voronoi ligament network"),
            ("img4", "P8 straight paths / P9 connectivity", "S2_strut_060120", "0/60/120 strut lattice"),
            ("img5", "P10 gradient / P11 halo around a flaw", "S2_hole20_rings2", "hole with two pore rings")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)), gridspec_kw={"width_ratios": [1, 1.2, 1]})
    for (img, princ, sname, lab), axr in zip(rows, axes):
        im = next(i for i in doc["images"] if i["id"] == img)
        axr[0].imshow(mpimg.imread(os.path.join(ROOT, im["file"]))); axr[0].axis("off"); axr[0].set_title(f"{img}: {im['short']}", fontsize=7.5)
        axr[1].axis("off"); axr[1].text(0.02, 0.5, princ + "\n\n→ " + lab, fontsize=8, va="center", wrap=True)
        s = S[sname]; a = D.generate(s["family"], **s["params"]); draw(axr[2], a, f"{sname}\nN={len(a)}, φ={a.info['porosity']:.2f}")
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(FIG, f"translation.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)
    open(os.path.join(FIG, "translation.caption.txt"), "w").write("Translation from imagery to atomistic carbon architecture: each reference image (left) motivated one or more design principles (middle) that were instantiated as an explicit periodic graphene-derived structure (right; atoms shown as dots). The structures are not traced from the images; they realise the abstract principle with controlled parameters and matched porosity.")


def fig_space(S):
    names = ["S1_pristine_zz", "S1_vac_clustered_2pct", "S1_precrack_L20", "S1_hole_d20", "S1_mesh_p16_phi0.2",
             "S2_H1_p12", "S2_H1_p16", "S2_H1_p32", "S2_H1_square_D40", "S2_H2_D24_W8", "S2_H2_D40_W8", "S2_H2_D60_W8", "S2_H2_D40_W4", "S2_H2_D40_W12", "S2_H3_D30_120",
             "S2_mesh_jitter2.0", "S2_mesh_sizedis0.3", "S2_voronoi_n25_reg0.0", "S2_voronoi_n25_reg1.0", "S2_voronoi_n50_reg0.0",
             "S2_slit_0deg", "S2_slit_45deg", "S2_slit_90deg", "S2_ellipse_0deg", "S2_ellipse_90deg", "S2_strut_060120", "S2_strut_090", "S2_strut_3090150",
             "S2_mesh_phi0.1", "S2_mesh_phi0.4", "S2_graded_x", "S2_graded_radial", "S2_hole20_rings2", "S2_hole20_background", "S2_vac_clustered_0.03"]
    names = [n for n in names if n in S]
    ncol = 7; nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.4 * nrow))
    descs = {}
    for ax, n in zip(axes.ravel(), names):
        s = S[n]; a = D.generate(s["family"], **s["params"]); draw(ax, a, f"{n.replace('S1_','').replace('S2_','')}\nN={len(a)} φ={a.info['porosity']:.2f} H={a.info['design'].get('hierarchy_levels')}", s=0.5)
        descs[n] = {"n_atoms": len(a), "porosity": a.info["porosity"], "family": s["family"], "hierarchy_levels": a.info["design"].get("hierarchy_levels")}
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(FIG, f"design_space.{ext}"), dpi=170 if ext == "png" else None)
    plt.close(fig)
    json.dump(descs, open(os.path.join(FIG, "design_space.json"), "w"), indent=1)
    open(os.path.join(FIG, "design_space.caption.txt"), "w").write("The explored design space (Stages 1-2): baselines (pristine, vacancies, cracks, hole, reference mesh), the hierarchy ladder (single-scale meshes with three periods, coarse level alone, two-level meshes with three scale ratios and three vein widths, three-level mesh) at matched porosity 0.20, disorder (jitter, size dispersion, Voronoi networks of different regularity and cell count), anisotropy and load-path organisation (slits at 0/45/90 degrees, elliptical pores, strut lattices with and without a strut family along the load), porosity sweep, gradients, and flaw halos. N = atoms, phi = porosity, H = designed hierarchy depth. Loading is along x (horizontal).")


if __name__ == "__main__":
    S = specs_by_name(); fig_translation(S); fig_space(S); print("design figures written")
