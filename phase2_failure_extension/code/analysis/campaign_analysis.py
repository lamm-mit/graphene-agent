"""Campaign analysis: loads the experiment database and produces the standard figures and tables:
  - stress-strain overview per stage / family
  - descriptor -> property relations (strength, failure strain, work, localisation vs porosity,
    ligament width, minimum solid fraction, anisotropy, tortuosity, hierarchy)
  - hierarchy ladder (depth, scale ratio, vein width) at matched porosity
  - localisation vs fracture mode; damage-event statistics
  - Pareto fronts (non-dominated sets) for several objective pairs, with structural diversity
  - summary tables (CSV / markdown)
All figures are generated from the stored simulation data only."""
from __future__ import annotations
import sys, os, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
EV = 16.0217663
FIG = os.path.join(ROOT, "figures", "campaign")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7})
FAMCOL = {"pristine": "k", "vacancies": "0.5", "precrack": "#8c564b", "ring_around_hole": "#e377c2", "nanomesh_single": "#1f77b4", "nanomesh_hier": "#d62728",
          "voronoi_network": "#2ca02c", "slit_array": "#ff7f0e", "strut_lattice": "#9467bd", "graded_pores": "#17becf"}


def save(fig, name, caption):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
    open(os.path.join(FIG, f"{name}.caption.txt"), "w").write(caption)
    plt.close(fig)


def load(stages=None, campaigns=("discovery",)):
    recs = [r for r in db.all_records() if r.get("status") == "completed" and (campaigns is None or r.get("campaign") in campaigns)]
    if stages:
        recs = [r for r in recs if r.get("stage") in stages]
    return recs


def curve(r):
    d = np.load(os.path.join(ROOT, r["trajectory"]))
    return d["eps_x"], d["sigma_xx"] * EV, d["n_broken_cum"]


def M(r, k):
    return r.get("metrics", {}).get(k, np.nan)


def Dd(r, k):
    return r.get("descriptors", {}).get(k, np.nan)


def fig_stress_strain(recs, name, title, group_key="family", max_per_group=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    groups = {}
    for r in recs:
        groups.setdefault(r.get(group_key), []).append(r)
    for g, rs in groups.items():
        for i, r in enumerate(rs[:max_per_group] if max_per_group else rs):
            e, s, nb = curve(r)
            ax.plot(e, s, "-", lw=1.1, color=FAMCOL.get(r["family"], None) if group_key == "family" else None, alpha=0.85, label=f"{r['name']}" if len(recs) <= 14 else (g if i == 0 else None))
            k = int(np.argmax(s)); ax.plot(e[k], s[k], "o", ms=3, color="k")
    ax.set_xlabel("engineering strain (-)"); ax.set_ylabel("2D stress σxx (N/m)"); ax.grid(alpha=0.3); ax.legend(ncol=2); ax.set_title(title, fontsize=9)
    save(fig, name, title + ". Every curve is an individual AQS simulation (uniaxial stress, REBO2+S); dots mark the peak stress.")


def fig_property_maps(recs, name="property_maps"):
    xs = [("porosity", "porosity φ", lambda r: r.get("porosity")), ("ligament_width_mean_A", "mean ligament width (Å)", lambda r: Dd(r, "ligament_width_mean_A")),
          ("min_solid_fraction_across_x", "minimum load-bearing solid fraction", lambda r: Dd(r, "min_solid_fraction_across_x")), ("fraction_undercoordinated", "fraction of edge (2-coordinated) atoms", lambda r: Dd(r, "fraction_undercoordinated")),
          ("anisotropy_index", "anisotropy index", lambda r: Dd(r, "anisotropy_index")), ("tortuosity_x", "load-path tortuosity", lambda r: Dd(r, "tortuosity_x"))]
    ys = [("strength_Nm", "strength (N/m)"), ("failure_strain", "failure strain"), ("work_to_failure_J_m2", "work to failure (J/m²)"), ("damage_localization", "damage localisation L")]
    fig, axes = plt.subplots(len(ys), len(xs), figsize=(3.0 * len(xs), 2.6 * len(ys)))
    fams = sorted(set(r["family"] for r in recs))
    for i, (yk, yl) in enumerate(ys):
        for j, (xk, xl, fx) in enumerate(xs):
            ax = axes[i, j]
            for f in fams:
                rs = [r for r in recs if r["family"] == f]
                ax.scatter([fx(r) for r in rs], [M(r, yk) for r in rs], s=18, color=FAMCOL.get(f, "0.3"), label=f if (i == 0 and j == 0) else None, alpha=0.85, edgecolors="none")
            if i == len(ys) - 1: ax.set_xlabel(xl)
            if j == 0: ax.set_ylabel(yl)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncol=2)
    save(fig, name, "Structure-property maps of all completed simulations: strength, failure strain, work to failure and damage localisation as functions of measured structural descriptors (porosity, mean ligament width, minimum load-bearing solid fraction across the loading direction, fraction of two-coordinated edge atoms, anisotropy index and load-path tortuosity). Each point is one simulation; colours denote architecture families.")


def fig_hierarchy(recs, name="hierarchy_ladder"):
    hs = [r for r in recs if "hierarchy" in (r.get("tags") or []) or (r["family"] == "nanomesh_hier" and r["stage"] in ("stage2_reconnaissance", "stage5_deep"))]
    if not hs:
        return
    hs = sorted(hs, key=lambda r: (r.get("hierarchy_levels") or 0, Dd(r, "pore_scale_ratio") or 0, r["name"]))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    plt.subplots_adjust(wspace=0.35)
    names = [r["name"].replace("S2_", "").replace("S5_", "") for r in hs]; x = np.arange(len(hs))
    for ax, (k, lab) in zip(axes, [("strength_Nm", "strength (N/m)"), ("failure_strain", "failure strain"), ("work_to_failure_J_m2", "work to failure (J/m²)"), ("damage_localization", "damage localisation L")]):
        vals = [M(r, k) for r in hs]
        cols = [{0: "0.6", 1: "#1f77b4", 2: "#d62728", 3: "#9467bd"}.get(r.get("hierarchy_levels") or 1, "0.3") for r in hs]
        ax.bar(x, vals, color=cols); ax.set_xticks(x); ax.set_xticklabels(names, rotation=70, fontsize=6.5); ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y")
    axes[0].set_title("hierarchy ladder at matched porosity φ≈0.20 (blue: 1 level, red: 2 levels, purple: 3 levels)", fontsize=8, loc="left")
    save(fig, name, "Hierarchy ladder: mechanical metrics of single-scale nanomeshes (1 level: fine, reference and coarse periods; square coarse pores), two-level nested meshes with different domain sizes (scale ratios 2, 3.3, 5) and vein widths (4, 8, 12 A), and a three-level mesh, all at matched porosity (0.20 +- 0.01) and identical footprint.")


def pareto_front(points, maximize=(True, True)):
    idx = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j: continue
            better_eq = all((q[k] >= p[k]) if maximize[k] else (q[k] <= p[k]) for k in range(2))
            better = any((q[k] > p[k]) if maximize[k] else (q[k] < p[k]) for k in range(2))
            if better_eq and better:
                dominated = True; break
        if not dominated:
            idx.append(i)
    return idx


def fig_pareto(recs, name="pareto_fronts", porous_only=False):
    if porous_only:
        recs = [r for r in recs if (r.get("porosity") or 0) > 0.05]
    pairs = [(("strength_Nm", True), ("failure_strain", True), "strength vs failure strain"),
             (("strength_Nm", True), ("work_to_failure_J_m2", True), "strength vs work to failure"),
             (("specific_work_eV_per_atom", True), ("porosity", True), "specific work (per atom) vs porosity (mass saving)"),
             (("modulus_2d_Nm", True), ("failure_strain", True), "stiffness vs failure strain"),
             (("strength_Nm", True), ("damage_localization", False), "strength vs delocalised damage (low L)"),
             (("work_to_failure_J_m2", True), ("post_peak_load_retention", True), "work vs post-peak load retention")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fronts = {}
    for ax, ((xk, xm), (yk, ym), title) in zip(axes.ravel(), pairs):
        getx = (lambda r: r.get("porosity")) if xk == "porosity" else (lambda r: M(r, xk)); gety = (lambda r: r.get("porosity")) if yk == "porosity" else (lambda r: M(r, yk))
        pts = [(getx(r), gety(r), r) for r in recs]; pts = [p for p in pts if p[0] is not None and p[1] is not None and np.isfinite(p[0]) and np.isfinite(p[1])]
        for f in sorted(set(p[2]["family"] for p in pts)):
            ps = [p for p in pts if p[2]["family"] == f]; ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=16, color=FAMCOL.get(f, "0.3"), label=f, alpha=0.8, edgecolors="none")
        fr = pareto_front([(p[0], p[1]) for p in pts], (xm, ym)); fr = sorted(fr, key=lambda i: pts[i][0])
        ax.plot([pts[i][0] for i in fr], [pts[i][1] for i in fr], "k-", lw=1, alpha=0.7)
        for i in fr:
            ax.annotate(pts[i][2]["name"].replace("S1_", "").replace("S2_", ""), (pts[i][0], pts[i][1]), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
        fronts[title] = [pts[i][2]["name"] for i in fr]
        ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(title, fontsize=8); ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6)
    save(fig, name, ("Pareto analysis restricted to porous architectures (porosity > 0.05): " if porous_only else "Pareto analysis of all simulated structures: ") + "non-dominated sets (black lines, labelled) for six objective pairs; each point is one simulated architecture. The fronts are structurally diverse by construction of the design space (no optimiser was used to collapse onto one family).")
    json.dump(fronts, open(os.path.join(FIG, "pareto_fronts.json"), "w"), indent=1)
    return fronts


def fig_localisation(recs, name="localisation_modes"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    modes = sorted(set(M(r, "fracture_mode") for r in recs))
    for i, m in enumerate(modes):
        rs = [r for r in recs if M(r, "fracture_mode") == m]
        ax[0].scatter([M(r, "damage_localization") for r in rs], [M(r, "post_peak_load_retention") for r in rs], s=20, label=m)
        ax[1].scatter([M(r, "n_damage_steps") for r in rs], [M(r, "strain_range_post_peak") for r in rs], s=20, label=m)
    ax[0].set_xlabel("damage localisation L (entropy-based)"); ax[0].set_ylabel("post-peak load retention"); ax[0].legend(fontsize=6); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("number of discrete damage steps"); ax[1].set_ylabel("strain range from peak to failure"); ax[1].grid(alpha=0.3)
    save(fig, name, "Localised versus progressive fracture: damage localisation, post-peak load retention, number of discrete damage events and post-peak strain range, coloured by the automatically assigned fracture-mode class.")


def summary_table(recs, path):
    import csv
    keys = ["run_id", "name", "stage", "family", "seed", "n_atoms", "porosity", "hierarchy_levels", "viability", "modulus_2d_Nm", "strength_Nm", "strain_at_peak", "first_damage_strain", "failure_strain",
            "work_to_failure_J_m2", "specific_work_eV_per_atom", "post_peak_load_retention", "n_broken_total", "n_damage_steps", "damage_localization", "fracture_mode", "termination"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in recs:
            w.writerow([r.get(k, M(r, k)) if k in r else M(r, k) for k in keys])


if __name__ == "__main__":
    recs = load()
    print(len(recs), "completed discovery runs")
    if recs:
        for st in sorted(set(r["stage"] for r in recs)):
            rs = [r for r in recs if r["stage"] == st]
            fig_stress_strain(rs, f"stress_strain_{st}", f"stress-strain curves, {st} ({len(rs)} simulations)")
        fig_property_maps(recs); fig_hierarchy(recs); fronts = fig_pareto(recs); fronts_porous = fig_pareto(recs, "pareto_fronts_porous", porous_only=True); fig_localisation(recs)
        summary_table(recs, os.path.join(FIG, "summary_table.csv"))
        print("pareto fronts:", json.dumps(fronts, indent=1))
