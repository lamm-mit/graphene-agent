"""A wordless film (about 2.5 min, MP4 H.264, 2560x1440, 30 fps) that follows the trajectory by which the agent explored the
world of hierarchical graphene metamaterials and how its decisions were made, with the first-principles reasoning made
visible: the rule-of-mixtures contrast, the force-balance argument that derives the weakest-column rule and its check
against the simulated stress field, the ligament-width origin of the two strength classes, the agent's own hypotheses
tested by matched pairs, the hashed holdout predictions with their misses, the alignment index built from the structure
tensor, the registered interpolation that predicted the alignment sweep, the scale argument behind the follow-up, the
prediction composed from the rule before the 300 A runs, the mechanisms at 300 A, the outcome of every registered test,
and the principles that survived.

Version 2 (expanded).  Version 1 (analysis/reasoning_movie_v1.py) provides the rendering machinery and the scenes that
are reused unchanged (design language, fracture scenes, outcomes, principles).  No words: the only text is numbers, unit
symbols, single-symbol axis labels and three symbolic relations (force balance, the net-section rule, the alignment
index).  Rendering conventions are those of the other movies; every number is read from the archives (phase-1 reference
copies, the phase-1 prediction and evaluation files, and the follow-up database); nothing is drawn by hand.

    python analysis/reasoning_movie.py --preview            # 960x540, 8 fps storyboard check (about 3 min)
    python analysis/reasoning_movie.py                      # 2560x1440, 30 fps master + 1080p version + poster frames

Outputs: movies_hq/R01_reasoning_trajectory.mp4 (+ _1080p.mp4), R01_reasoning_trajectory.json, R01_README.md, R01_frames/.
"""
from __future__ import annotations
import os, sys, json, time, argparse, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Arc, Rectangle, ConnectionPatch
import analysis.reasoning_movie_v1 as V1
from analysis.reasoning_movie_v1 import (Movie, Panel, struct, bonds_of, draw_struct, square_axes, style_chart, load_arrows, ease, column_profile, scene_language, fracture_scene,
                                          scene_outcomes, scene_principles, test_layout, TESTS, compact, OWN, REF, ALL, RREF, PRED, FAM, DARK, GREY, RED, BLUE, ORANGE, GREEN, PURPLE, BROWN, PINK, CYAN, CMAP, EV)
from analysis.assess_phase1 import ordered_mesh_level
from experiments.campaign_paper_sweeps import alignment_index
from atomistics.descriptors.columns import vein_bands
from experiments import db
LOG = V1.LOG
P1 = os.path.join(os.path.dirname(ROOT), "carbon_discovery", "experiments")            # phase-1/2 archive, read only
HOLD = json.load(open(os.path.join(P1, "holdouts", "holdout_evaluation.json")))
SWEEP = json.load(open(os.path.join(P1, "holdouts", "paper_sweeps_evaluation.json")))
STAGE4 = {p["name"]: p for p in json.load(open(os.path.join(P1, "predictions", "stage4_predictions.json")))["predictions"]}
HYP = json.load(open(os.path.join(P1, "predictions", "hypotheses.json")))
FIELD_CMAP = "inferno"


# ----------------------------------------------------------------------------- helpers
def stress_field(name, frac_of_peak=0.6, principal=False, strain=None):
    """Per-atom in-plane stress (N/m) of a stored state before damage: sigma_xx or the maximum principal stress."""
    p = Panel(name); e = strain if strain is not None else frac_of_peak * p.peak_eps
    k = max(1, min(int(np.searchsorted(p.eps, e)), p.n - 1)); T = p.T
    P = T["positions"][k][:, :2].astype(float); c = T["cells"][k]; L = np.array([c[0, 0], c[1, 1]], float); V = T["peratom_virial"][k].astype(float)
    S = V / (L[0] * L[1] / len(P)) * EV
    val = S[:, 0] if not principal else 0.5 * (S[:, 0] + S[:, 1]) + np.sqrt((0.5 * (S[:, 0] - S[:, 1])) ** 2 + S[:, 2] ** 2)
    return p, k, np.mod(P, L), L, val, float(p.sig[k])


def draw_field(ax, P, L, val, px_per_A, vmin, vmax, cmap=FIELD_CMAP, alpha=1.0):
    atom_pt = 0.95 * px_per_A * 72 / 100.0
    lc = LineCollection(bonds_of(P, L), colors="0.55", linewidths=max(0.5, 0.18 * atom_pt), zorder=1, alpha=alpha); ax.add_collection(lc)
    sc = ax.scatter(P[:, 0], P[:, 1], s=atom_pt ** 2, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2, alpha=alpha)
    return lc, sc


def frame_fade(mv, box, seconds, poster=None):
    """Fade a region from white (a local overlay), used to bring in a panel."""
    ov = mv.fig.add_axes([0, 0, 1, 1], zorder=60); ov.axis("off"); ov.patch.set_alpha(0)
    r = ov.add_patch(Rectangle((box[0] - 0.01, box[1] - 0.01), box[2] + 0.02, box[3] + 0.02, transform=ov.transAxes, facecolor="white", edgecolor="none", zorder=60))
    for k in range(mv.n(seconds)):
        r.set_alpha(1 - (k + 1) / mv.n(seconds)); mv.frame(poster if k == mv.n(seconds) - 1 else None)
    ov.remove()


def bracket(ax, p0, p1, label, mv, offset=(0, 0), color=DARK):
    """Dimension line between two points with a number, in data coordinates."""
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1.8 * mv.s, shrinkA=0, shrinkB=0), zorder=6)
    xm, ym = (p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1]
    return ax.text(xm, ym, label, fontsize=mv.fs(15), color=color, ha="center", va="center", zorder=7, bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))


def pred_obs_axes(mv, box, lim=(4, 30)):
    ax = mv.fig.add_axes(box); style_chart(ax, r"$\sigma_\mathrm{pred}$", r"$\sigma$", mv.fs(11)); ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    xx = np.array(lim); ax.fill_between(xx, 0.85 * xx, 1.15 * xx, color="0.92", zorder=0); ax.plot(xx, xx, "-", color=DARK, lw=1.2 * mv.s, zorder=1)
    return ax


# ----------------------------------------------------------------------------- scene 2: exploration and the rule of mixtures
def scene_exploration(mv, seconds=8.0):
    mv.begin("2 exploration: the 96 discovery runs in run order; strength is not a rule of mixtures, the modulus nearly is")
    disc = sorted([r for n, r in REF.items() if n.startswith("S")], key=lambda r: r["run_id"])
    rho = np.array([1 - r["descriptors"]["porosity"] for r in disc]); sig = np.array([r["metrics"]["strength_Nm"] for r in disc]); Y = np.array([r["metrics"]["modulus_2d_Nm"] for r in disc]); fam = [r["family"] for r in disc]
    fs = mv.fs(11)
    axc = mv.fig.add_axes([0.07, 0.12, 0.50, 0.80]); style_chart(axc, r"$\bar{\rho}$", r"$\sigma$", fs); axc.set_xlim(0.55, 1.03); axc.set_ylim(0, 42)
    pts = axc.scatter([], [], s=[], edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    box = [0.625, 0.27, 0.46 * mv.H / mv.W, 0.46]; axs = mv.fig.add_axes(box); axs.set_facecolor("white")
    for sp in axs.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
    n_show = 24; show_idx = np.linspace(0, len(disc) - 1, n_show).round().astype(int); frames = mv.n(seconds); per_point = frames / len(disc); cur = [None]; order_log = []
    def show_struct(i):
        for a in (cur[0] or []): a.remove()
        P, L, _ = struct(disc[i]["name"]); px = box[2] * mv.W / (max(L) + 3)
        cur[0] = draw_struct(axs, P, L, px, color=FAM.get(fam[i], DARK)); square_axes(axs, L)
        for sp in axs.spines.values(): sp.set_edgecolor(FAM.get(fam[i], DARK))
    show_struct(0); mv.fade_in(0.5)
    for k in range(frames):
        m = min(len(disc), int(k / per_point) + 1)
        sizes = np.full(m, 90.0 * mv.s ** 2); sizes[-1] = 90.0 * mv.s ** 2 * (1 + 2.5 * max(0.0, 1 - (k - (m - 1) * per_point) / per_point))
        pts.set_offsets(np.c_[rho[:m], sig[:m]]); pts.set_sizes(sizes); pts.set_facecolor([FAM.get(f, DARK) for f in fam[:m]])
        j = int(np.searchsorted(show_idx, m - 1, side="right") - 1)
        if j >= 0 and show_idx[j] != getattr(show_struct, "last", -1):
            show_struct(show_idx[j]); show_struct.last = show_idx[j]; order_log.append(disc[show_idx[j]]["name"])
        mv.frame(poster="s2_exploration" if k == frames // 2 else None)
    from scipy.spatial import ConvexHull
    hulls = {}
    for f in sorted(set(fam)):
        idx = [i for i, g in enumerate(fam) if g == f]
        if len(idx) >= 3:
            X = np.c_[rho[idx], sig[idx]]
            try:
                h = ConvexHull(X); hulls[f] = axc.fill(X[h.vertices, 0], X[h.vertices, 1], color=FAM.get(f, DARK), alpha=0.0, zorder=1, lw=0)[0]
            except Exception: pass
    for k in range(mv.n(1.0)):
        for p in hulls.values(): p.set_alpha(0.13 * (k + 1) / mv.n(1.0))
        mv.frame(poster="s2_envelopes" if k == mv.n(1.0) - 1 else None)
    # the rule of mixtures: the line through the origin and the pristine sheet; the strengths lie far below it
    s0 = REF["S1_pristine_zz"]["metrics"]["strength_Nm"]; Y0 = REF["S1_pristine_zz"]["metrics"]["modulus_2d_Nm"]
    rom, = axc.plot([], [], "--", color=DARK, lw=2.2 * mv.s, zorder=2)
    for k in range(mv.n(1.0)):
        t = ease((k + 1) / mv.n(1.0)); xx = np.array([0.55, 0.55 + t * 0.48]); rom.set_data(xx, s0 * xx); mv.frame()
    mv.hold(0.8, poster="s2_rule_of_mixtures")
    # the modulus follows the same construction
    ibox = [0.125, 0.56, 0.21, 0.32]; axi = mv.fig.add_axes(ibox, zorder=5); style_chart(axi, r"$\bar{\rho}$", r"$Y$", mv.fs(9)); axi.set_xlim(0.55, 1.03); axi.set_ylim(0, 290); axi.patch.set_alpha(0.92)
    axi.scatter(rho, Y, s=40 * mv.s ** 2, c=[FAM.get(f, DARK) for f in fam], edgecolors="white", linewidths=0.4 * mv.s, zorder=3); axi.plot([0.55, 1.03], [Y0 * 0.55, Y0 * 1.03], "--", color=DARK, lw=1.8 * mv.s)
    frame_fade(mv, ibox, 0.8); mv.hold(1.8, poster="s2_modulus")
    LOG["scene2"] = dict(n_designs=len(disc), families=sorted(set(fam)), thumbnails_in_order=order_log, pristine_strength_Nm=s0, pristine_modulus_Nm=Y0)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 3: hypotheses tested by matched pairs
def scene_hypotheses(mv):
    mv.begin("3 hypotheses formed from the first 40 runs and tested by matched pairs: pore shape does not matter at equal section (H-A refuted); load-parallel veins set the strength of a nested mesh (H-F supported)")
    fs = mv.fs(11)
    rows = [("H-A", ["S4_square_D40", "S4_circle_D40_sameMinSF", "S4_circle_D40_samePhi"], BLUE), ("H-F", ["S2_H2_D40_W8", "S4_H2_D40_W8_veins_x", "S4_H2_D40_W8_veins_y"], RED)]
    tw = 0.135; ys = (0.55, 0.08); axes_p = []
    for (hid, names, col), y in zip(rows, ys):
        thumbs = []
        for i, n in enumerate(names):
            bx = [0.06 + i * (tw + 0.03), y, tw, tw * mv.W / mv.H]; ax = mv.fig.add_axes(bx); P, L, _ = struct(n); draw_struct(ax, P, L, tw * mv.W / (max(L) + 3)); square_axes(ax, L)
            for sp in ax.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
            thumbs.append(ax)
        load_arrows(mv.fig, thumbs[0].get_position().bounds, s=0.7 * mv.s)
        axp = mv.fig.add_axes([0.62, y + 0.02, 0.34, tw * mv.W / mv.H - 0.02]); style_chart(axp, "", r"$\sigma$", fs); axp.set_xlim(-0.6, 2.6); axp.set_ylim(0, 22); axp.set_xticks([])
        axes_p.append((axp, names, col))
    mv.fade_in(0.5); mv.hold(0.8, poster="s3_pairs")
    log = []
    for axp, names, col in axes_p:
        for i, n in enumerate(names):
            pr = STAGE4.get(n, {}).get("predicted", {}).get("strength_Nm")
            if pr is not None:
                axp.plot([i], [pr], "o", ms=17 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.2 * mv.s, zorder=4); mv.hold(0.3)
    mv.hold(0.5, poster="s3_predictions")
    for axp, names, col in axes_p:
        for i, n in enumerate(names):
            pr = STAGE4.get(n, {}).get("predicted", {}).get("strength_Nm"); ob = REF[n]["metrics"]["strength_Nm"]; log.append(dict(name=n, predicted=pr, observed=ob))
            if pr is None:
                axp.plot([i], [ob], "o", ms=17 * mv.s, markerfacecolor=GREY, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=5); mv.hold(0.3); continue
            c = GREEN if abs(ob - pr) / ob <= 0.15 else RED
            ln, = axp.plot([], [], "-", color=c, lw=2.5 * mv.s, zorder=3); dot, = axp.plot([], [], "o", ms=17 * mv.s, markerfacecolor=c, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=5)
            for k in range(mv.n(0.4)):
                t = ease((k + 1) / mv.n(0.4)); yv = pr + t * (ob - pr); ln.set_data([i, i], [pr, yv]); dot.set_data([i], [yv]); mv.frame()
            mv.hold(0.15)
    mv.hold(1.6, poster="s3_outcomes")
    LOG["scene3"] = dict(hypotheses_written=HYP["written"], based_on_records=HYP["based_on_records"], pairs=log, outcomes={h["id"]: str(h.get("outcome", ""))[:160] for h in HYP["hypotheses"] if h["id"] in ("H-A", "H-F")})
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 4: the weakest column from force balance
def scene_rule(mv):
    mv.begin("4 first principle: the force through every column is the same, so the column with the least material carries the highest stress and fails first: strength = weakest column fraction x ligament strength")
    fs = mv.fs(11); name = "S2_graded_x"
    p, kf, P, L, val, sig = stress_field(name, 0.6); prof, xs = column_profile(P, L); imin = int(np.argmin(prof))
    # measured mean per-atom stress in a 3 A column window, relative to the applied stress; force balance predicts sigma/sigma_col = A_col/A
    ratio = np.array([val[np.abs(((P[:, 0] - x + L[0] / 2) % L[0]) - L[0] / 2) < 1.5].mean() for x in xs]) / sig
    inv = 1.0 / np.maximum(ratio, 0.2); inv = np.convolve(np.pad(inv, 2, mode="wrap"), np.ones(5) / 5, mode="valid")
    boxw = 0.56; box = [0.05, 0.36, boxw * mv.H / mv.W, boxw]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
    lc, sc = draw_field(ax, P, L, val / sig, box[2] * mv.W / (max(L) + 3), 0.0, 2.5); square_axes(ax, L); load_arrows(mv.fig, box, s=mv.s)
    cax = mv.fig.add_axes([box[0] + box[2] + 0.045, box[1], 0.007, box[3]]); cb = mv.fig.colorbar(sc, cax=cax); cb.ax.tick_params(labelsize=mv.fs(10))
    band = ax.axvspan(0, 3, color=BLUE, alpha=0.3, zorder=4)
    axp = mv.fig.add_axes([box[0], 0.10, box[2], 0.20]); style_chart(axp, "", "", fs); axp.set_xlim(0, L[0]); axp.set_ylim(0, 1.25); axp.set_xticks([]); axp.set_yticks([0, 0.5, 1.0])
    l_geo, = axp.plot([], [], "-", color=BLUE, lw=2.6 * mv.s); l_str, = axp.plot([], [], "-", color=RED, lw=2.2 * mv.s); dot, = axp.plot([], [], "o", color=BLUE, ms=9 * mv.s)
    # right: the rule through all porous phase-1 designs
    axc = mv.fig.add_axes([0.60, 0.12, 0.37, 0.64]); style_chart(axc, r"$A_\mathrm{min}/A$", r"$\sigma$", fs); axc.set_xlim(0.1, 1.05); axc.set_ylim(0, 42)
    por = [r for n, r in REF.items() if n.startswith("S") and r["family"] not in ("pristine", "vacancies")]
    X = np.array([r["descriptors"]["min_solid_fraction_across_x"] for r in por]); Yv = np.array([r["metrics"]["strength_Nm"] for r in por]); C = [FAM.get(r["family"], DARK) for r in por]
    pts = axc.scatter([], [], s=70 * mv.s ** 2, edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    l1, = axc.plot([], [], "-", color=DARK, lw=2.2 * mv.s, zorder=2); l2, = axc.plot([], [], "-", color=DARK, lw=2.2 * mv.s, zorder=2)
    f1 = mv.fig.text(0.785, 0.90, r"$\sigma_\mathrm{col}\,A_\mathrm{col} = \sigma\,A$", fontsize=mv.fs(26), ha="center", va="center", alpha=0.0)
    f2 = mv.fig.text(0.785, 0.82, r"$\sigma_\mathrm{max} = \dfrac{A_\mathrm{min}}{A}\;\sigma_\mathrm{lig}$", fontsize=mv.fs(26), ha="center", va="center", alpha=0.0)
    mv.fade_in(0.5)
    n = mv.n(2.8)
    for k in range(n):
        t = (k + 1) / n; i = int(t * (len(xs) - 1)); band.set_x(xs[i] - 1.5); band.set_width(3.0)
        l_geo.set_data(xs[:i + 1], prof[:i + 1]); l_str.set_data(xs[:i + 1], inv[:i + 1]); dot.set_data([xs[i]], [prof[i]]); mv.frame()
    mv.hold(0.6, poster="s4_column_balance")
    for k in range(mv.n(0.6)): f1.set_alpha((k + 1) / mv.n(0.6)); mv.frame()
    mv.hold(0.8)
    band.set_color(RED); band.set_alpha(0.4); band.set_x(xs[imin] - 1.5); dot.set_data([xs[imin]], [prof[imin]]); dot.set_color(RED); axp.axhline(prof[imin], color=RED, lw=1.5 * mv.s, ls="--")
    for k in range(mv.n(0.6)): f2.set_alpha((k + 1) / mv.n(0.6)); mv.frame()
    mv.hold(0.8, poster="s4_weakest_column")
    order = np.argsort(X)
    for k in range(mv.n(2.0)):
        m = max(1, int((k + 1) / mv.n(2.0) * len(por))); idx = order[:m]; pts.set_offsets(np.c_[X[idx], Yv[idx]].reshape(-1, 2)); pts.set_facecolor([C[i] for i in idx]); mv.frame()
    s_hi, s_lo = RREF["classes"]["straight"]["stored_median"], RREF["classes"]["fine_round"]["stored_median"]
    for k in range(mv.n(1.2)):
        t = ease((k + 1) / mv.n(1.2)); xx = np.array([0, 1.05 * t]); l1.set_data(xx, s_hi * xx); l2.set_data(xx, s_lo * xx); mv.frame()
    mv.hold(1.6, poster="s4_net_section_rule")
    LOG["scene4"] = dict(structure=name, frame=int(kf), strain=float(p.eps[kf]), applied_stress_Nm=sig, min_column_fraction=float(prof[imin]), corr_geometry_vs_inverse_stress=float(np.corrcoef(prof, inv)[0, 1]), n_points=len(por), lines_Nm=[s_hi, s_lo])
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 5: why two ligament strengths
def scene_ligaments(mv):
    mv.begin("5 why two ligament strengths: the strength per unit section of a ligament falls with its width, because edge atoms carry a larger share of a narrow ligament")
    fs = mv.fs(11)
    ex = [("S2_slit_0deg", "straight strips"), ("S2_H1_p16", "narrow round-pore ligaments")]
    fields = [stress_field(n, 0.8, principal=True) for n, _ in ex]; vmax = float(max(np.percentile(f[4], 99) for f in fields))
    boxw = 0.40; boxes = [[0.05, 0.53, boxw * mv.H / mv.W, boxw], [0.05, 0.08, boxw * mv.H / mv.W, boxw]]
    for (n, lab), f, box in zip(ex, fields, boxes):
        p, k, P, L, val, sig = f; ax = mv.fig.add_axes(box); ax.set_facecolor("white")
        draw_field(ax, P, L, val, box[2] * mv.W / (max(L) + 3), 0.0, vmax); square_axes(ax, L); load_arrows(mv.fig, box, s=0.8 * mv.s)
        for sp in ax.spines.values(): sp.set_edgecolor(FAM.get(REF[n]["family"], DARK)); sp.set_linewidth(2.5 * mv.s)
    cax = mv.fig.add_axes([boxes[0][0] + boxes[0][2] + 0.045, 0.08, 0.007, 0.85]); cb = mv.fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=FIELD_CMAP), cax=cax); cb.ax.tick_params(labelsize=mv.fs(10))
    meshes = [x for x in REF.values() if x["family"] in ("nanomesh_single", "nanomesh_hier") and x["name"][0] in "SP"]
    w = np.array([x["descriptors"]["ligament_width_mean_A"] for x in meshes]); sl = np.array([x["metrics"]["strength_Nm"] / x["descriptors"]["min_solid_fraction_across_x"] for x in meshes])
    axc = mv.fig.add_axes([0.42, 0.14, 0.54, 0.78]); style_chart(axc, r"$w$ (Å)", r"$\sigma / (A_\mathrm{min}/A)$", fs); axc.set_xlim(5, 34); axc.set_ylim(8, 36)
    pts = axc.scatter([], [], s=80 * mv.s ** 2, edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    s_hi, s_lo = RREF["classes"]["straight"]["stored_median"], RREF["classes"]["fine_round"]["stored_median"]
    lh = axc.axhline(s_hi, color=DARK, lw=2 * mv.s, alpha=0.0); ll = axc.axhline(s_lo, color=DARK, lw=2 * mv.s, alpha=0.0)
    mv.fade_in(0.5); mv.hold(1.4, poster="s5_ligament_fields")
    order = np.argsort(w); cols = [FAM.get(x["family"], DARK) for x in meshes]
    for k in range(mv.n(2.0)):
        m = max(1, int((k + 1) / mv.n(2.0) * len(meshes))); idx = order[:m]; pts.set_offsets(np.c_[w[idx], sl[idx]].reshape(-1, 2)); pts.set_facecolor([cols[i] for i in idx]); mv.frame()
    for k in range(mv.n(0.8)): lh.set_alpha((k + 1) / mv.n(0.8)); ll.set_alpha((k + 1) / mv.n(0.8)); mv.frame()
    mv.hold(1.6, poster="s5_width_scatter")
    LOG["scene5"] = dict(examples=[n for n, _ in ex], n_meshes=len(meshes), corr_width_ligament_strength=float(np.corrcoef(w, sl)[0, 1]), class_lines_Nm=[s_hi, s_lo], field_vmax_Nm=vmax)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 6a: the holdouts
def scene_holdout(mv):
    mv.begin("6a twelve holdout designs predicted and hashed before simulation: most within 15 %, two large misses; the nested design with both levels aligned was underestimated by a third")
    fs = mv.fs(11); rows = HOLD["rows"]
    ax = pred_obs_axes(mv, [0.06, 0.10, 0.50, 0.84], lim=(4, 30))
    hollow = [ax.plot([], [], "o", ms=16 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.2 * mv.s, zorder=4)[0] for _ in rows]
    tw = 0.20; tb = [[0.66, 0.53, tw, tw * mv.W / mv.H], [0.66, 0.10, tw, tw * mv.W / mv.H]]; big = ["S7_H2_veinsX_W12_ellipseX", "S7_slit_20deg"]
    mv.fade_in(0.5)
    for i, r in enumerate(rows):
        pr = r["predicted"]["strength_Nm"]; hollow[i].set_data([pr], [pr]); mv.hold(0.12)
    mv.hold(0.6, poster="s6a_predictions")
    log = []
    for i, r in enumerate(rows):
        pr, ob = r["predicted"]["strength_Nm"], r["observed"]["strength_Nm"]; c = GREEN if abs(ob - pr) / pr <= 0.15 else RED; log.append(dict(name=r["name"], predicted=pr, observed=ob))
        ln, = ax.plot([], [], "-", color=c, lw=2.5 * mv.s, zorder=3); dot, = ax.plot([], [], "o", ms=16 * mv.s, markerfacecolor=c, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=5)
        for k in range(mv.n(0.3)):
            t = ease((k + 1) / mv.n(0.3)); yv = pr + t * (ob - pr); ln.set_data([pr, pr], [pr, yv]); dot.set_data([pr], [yv]); mv.frame()
        if r["name"] in big:
            j = big.index(r["name"]); axt = mv.fig.add_axes(tb[j]); P, L, _ = struct(r["name"]); draw_struct(axt, P, L, tw * mv.W / (max(L) + 3)); square_axes(axt, L)
            for sp in axt.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(3 * mv.s)
            mv.fig.add_artist(ConnectionPatch(xyA=(pr, ob), coordsA=ax.transData, xyB=(0, 0.5), coordsB=axt.transAxes, color=RED, lw=1.4 * mv.s, ls=":", zorder=1))
            frame_fade(mv, tb[j], 0.4)
        mv.hold(0.08)
    mv.hold(1.8, poster="s6a_misses")
    LOG["scene6a"] = dict(prediction_written=HOLD["prediction_written"], sha256=HOLD["prediction_sha256"][:16], rows=log)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 6b: the alignment index
def scene_alignment_index(mv):
    mv.begin("6b a measurable definition of alignment: the structure tensor of the solid phase gives an anisotropy a and a ligament angle φ to the load; A = a cos 2φ")
    ex = ["S2_slit_0deg", "S4_square_D40", "S2_slit_90deg"]; tw = 0.44; boxes = [[0.04 + k * 0.32, 0.30, tw * mv.H / mv.W, tw] for k in range(3)]
    f = mv.fig.text(0.5, 0.90, r"$A = a\,\cos 2\varphi$", fontsize=mv.fs(30), ha="center", va="center", alpha=0.0)
    mv.fade_in(0.5); vals = []
    for n, box in zip(ex, boxes):
        ax = mv.fig.add_axes(box); P, L, _ = struct(n); draw_struct(ax, P, L, box[2] * mv.W / (max(L) + 3), color="0.35"); square_axes(ax, L)
        for sp in ax.spines.values(): sp.set_edgecolor(FAM.get(REF[n]["family"], DARK)); sp.set_linewidth(2.5 * mv.s)
        d = REF[n]["descriptors"]; a = d["anisotropy_index"]; phi = d["ligament_orientation_deg"]; A = alignment_index(d); vals.append(dict(name=n, a=a, phi_deg=phi, A=A))
        cx, cy = L / 2; R = 0.30 * max(L); ax.plot([cx - 0.45 * L[0], cx + 0.45 * L[0]], [cy, cy], "--", color=DARK, lw=1.5 * mv.s, zorder=5)
        ell = Ellipse((cx, cy), 2 * R * math.sqrt(1 + a), 2 * R * math.sqrt(max(1 - a, 0.05)), angle=phi, facecolor=RED, alpha=0.0, edgecolor=RED, lw=3 * mv.s, zorder=6); ax.add_patch(ell)
        frame_fade(mv, box, 0.4)
        for k in range(mv.n(0.7)):
            t = ease((k + 1) / mv.n(0.7)); ell.set_alpha(0.25 * t); ell.width = 2 * R * math.sqrt(1 + a) * t + 1e-3; ell.height = 2 * R * math.sqrt(max(1 - a, 0.05)) * t + 1e-3; mv.frame()
        if 5 < (phi % 180) < 175:
            ax.add_patch(Arc((cx, cy), 0.9 * R, 0.9 * R, theta1=0, theta2=phi % 180, color=DARK, lw=2 * mv.s, zorder=7))
        ax.text(cx, cy - 0.47 * L[1], f"{A:+.2f}", fontsize=mv.fs(22), ha="center", va="top", color=DARK, zorder=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
        mv.hold(0.5)
    for k in range(mv.n(0.6)): f.set_alpha((k + 1) / mv.n(0.6)); mv.frame()
    mv.hold(1.4, poster="s6b_alignment_index")
    LOG["scene6b"] = dict(examples=vals)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 6c: the registered sweep, the trend and the premium
def scene_sweep(mv):
    mv.begin("6c the registered interpolation in A predicts the eight new alignment designs; 19 single-level meshes lie on one trend, the 21 nested meshes below it: the hierarchy premium is negative")
    fs = mv.fs(11)
    meshes = [r for r in REF.values() if ordered_mesh_level(r) and 0.17 <= (r.get("porosity") or 0) <= 0.23]
    A = np.array([alignment_index(r["descriptors"]) for r in meshes]); S = np.array([r["metrics"]["strength_Nm"] for r in meshes]); lev = np.array([r.get("hierarchy_levels", 1) for r in meshes]); names = [r["name"] for r in meshes]
    one = lev == 1; fit = np.polyfit(A[one], S[one], 1)
    sweep = [r for r in SWEEP["rows"] if r["series"] != "angle_sweep" and r["series"] != "overlap_control"]
    axc = mv.fig.add_axes([0.40, 0.12, 0.56, 0.80]); style_chart(axc, r"$A$", r"$\sigma$", fs); axc.set_xlim(-0.35, 0.72); axc.set_ylim(5, 24)
    mk = {1: ("o", BLUE, 90), 2: ("s", RED, 100), 3: ("^", PURPLE, 110)}
    old = [i for i in range(len(meshes)) if not names[i].startswith("P3")]
    box = [0.04, 0.30, 0.44 * mv.H / mv.W, 0.44]; axs = mv.fig.add_axes(box); axs.set_facecolor("white"); axs.axis("off")
    mv.fade_in(0.5)
    scs = {l: axc.scatter([], [], marker=mk[l][0], s=mk[l][2] * mv.s ** 2, c=mk[l][1], edgecolors="white", linewidths=0.6 * mv.s, zorder=3) for l in (1, 2, 3)}
    for k in range(mv.n(1.5)):
        m = max(1, int((k + 1) / mv.n(1.5) * len(old))); ii = old[:m]
        for l in (1, 2, 3):
            jj = [i for i in ii if min(lev[i], 3) == l]; scs[l].set_offsets(np.c_[A[jj], S[jj]].reshape(-1, 2) if jj else np.zeros((0, 2)))
        mv.frame()
    mv.hold(0.5, poster="s6c_phase1_meshes")
    cur = [None]; log = []
    for r in sweep:
        for a in (cur[0] or []): a.remove()
        P, L, _ = struct(r["name"]); cur[0] = draw_struct(axs, P, L, box[2] * mv.W / (max(L) + 3), color=BLUE if r["series"] == "H1_ellipse" else RED); square_axes(axs, L); axs.axis("off")
        hol, = axc.plot([r["A"]], [r["mech"]], "o", ms=16 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.2 * mv.s, zorder=4); mv.hold(0.25)
        c = GREEN if abs(r["simulated"] - r["mech"]) / r["simulated"] <= 0.15 else RED; log.append(dict(name=r["name"], A=r["A"], predicted=r["mech"], simulated=r["simulated"]))
        ln, = axc.plot([], [], "-", color=c, lw=2.5 * mv.s, zorder=3); dot, = axc.plot([], [], "o", ms=16 * mv.s, markerfacecolor=c, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=5)
        for k in range(mv.n(0.3)):
            t = ease((k + 1) / mv.n(0.3)); yv = r["mech"] + t * (r["simulated"] - r["mech"]); ln.set_data([r["A"], r["A"]], [r["mech"], yv]); dot.set_data([r["A"]], [yv]); mv.frame()
    mv.hold(0.6, poster="s6c_sweep")
    trend, = axc.plot([], [], "-", color=BLUE, lw=2.4 * mv.s, zorder=2)
    for k in range(mv.n(1.0)):
        t = ease((k + 1) / mv.n(1.0)); xx = np.array([A[one].min(), A[one].min() + t * (A[one].max() - A[one].min())]); trend.set_data(xx, np.polyval(fit, xx)); mv.frame()
    nested = np.where(lev >= 2)[0]
    for k in range(mv.n(1.5)):
        m = int((k + 1) / mv.n(1.5) * len(nested))
        for i in nested[:m]:
            if not hasattr(axc, "_stems"): axc._stems = set()
            if i not in axc._stems: axc.plot([A[i], A[i]], [np.polyval(fit, A[i]), S[i]], "-", color=RED, lw=1.6 * mv.s, alpha=0.7, zorder=2); axc._stems.add(i)
        mv.frame()
    mv.hold(1.8, poster="s6c_hierarchy_premium")
    prem = S[lev >= 2] - np.polyval(fit, A[lev >= 2])
    LOG["scene6c"] = dict(n_meshes=len(meshes), n_one=int(one.sum()), n_nested=int((lev >= 2).sum()), trend=[float(fit[0]), float(fit[1])], premium_mean=float(prem.mean()), premium_std=float(prem.std()), sweep=log)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 7: the decision
def scene_decision(mv):
    mv.begin("7 decision: is the conclusion bounded by scale? the levels of the best nested design are separated by a factor of three; the same design is built at a factor of eight; the stiff veins should carry the load around a crack; five tests are registered with hashed predictions")
    fs = mv.fs(11)
    P1_, L1, d1 = struct("H1_veinsX_r0"); P3_, L3, d3 = struct("H3_2L_r0")
    boxw = 0.80; box = [0.5 - boxw * mv.H / mv.W / 2, 0.10, boxw * mv.H / mv.W, boxw]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(2 * mv.s)
    px1 = box[2] * mv.W / (max(L1) + 3); px3 = box[2] * mv.W / (max(L3) + 3)
    a1 = draw_struct(ax, P1_, L1, px1, color=RED); a3 = draw_struct(ax, P3_, L3, px3, alpha=0.0); square_axes(ax, L1, pad=9); load_arrows(mv.fig, box, s=mv.s)
    mv.fade_in(0.5, poster="s7_120A"); mv.hold(0.5)
    def brackets(design, L, texts):
        b = vein_bands(design); yc, hw, sy = b[0]; per = design.get("period", 12)
        arts = [bracket(ax, (-4, yc), (-4, yc + sy), texts[0], mv, offset=(-9 * L[0] / 120, 0)), bracket(ax, (L[0] + 4, yc - hw), (L[0] + 4, yc + hw), texts[1], mv, offset=(9 * L[0] / 120, 0)),
                bracket(ax, (L[0] / 2 - per / 2, -4), (L[0] / 2 + per / 2, -4), texts[2], mv, offset=(0, -7 * L[0] / 120))]
        return arts
    br1 = brackets(d1, L1, ("40 Å", "12 Å", "12 Å")); mv.hold(1.6, poster="s7_brackets_120A")
    for t_ in ax.texts: t_.remove()
    for ch in list(ax.get_children()):
        if ch.__class__.__name__ == "Annotation": ch.remove()
    n = mv.n(2.2)
    for k in range(n):
        t = ease((k + 1) / n); side = max(L1) + t * (max(L3) - max(L1)); c = L1 / 2 + t * (L3 / 2 - L1 / 2); pad = 9 + t * 14
        ax.set_xlim(c[0] - side / 2 - pad, c[0] + side / 2 + pad); ax.set_ylim(c[1] - side / 2 - pad, c[1] + side / 2 + pad)
        pt = 0.95 * (box[2] * mv.W / (side + 3)) * 72 / 100.0
        for a in (a1, a3): a[1].set_sizes([pt ** 2]); a[0].set_linewidths([max(0.5, 0.18 * pt)])
        a3[0].set_alpha(t); a3[1].set_alpha(t); mv.frame()
    mv.hold(0.4)
    br3 = brackets(d3, L3, ("100 Å", "30 Å", "12 Å")); mv.hold(1.6, poster="s7_brackets_300A")
    # the crack-arrest argument: before any damage the stiff veins carry the load around a crack inside a domain
    for k in range(mv.n(0.4)):
        t = (k + 1) / mv.n(0.4); a1[0].set_alpha(1 - t); a1[1].set_alpha(1 - t); a3[0].set_alpha(1 - t); a3[1].set_alpha(1 - t); mv.frame()
    ax.remove(); mv.fig.patches.clear()
    p, kf, Pc, Lc, val, sig = stress_field("H1_veinsX_L20_r0", 0.5)
    box2 = [0.5 - boxw * mv.H / mv.W / 2, 0.10, boxw * mv.H / mv.W, boxw]; ax2 = mv.fig.add_axes(box2); ax2.set_facecolor("white")
    for sp in ax2.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(2 * mv.s)
    lc, sc = draw_field(ax2, Pc, Lc, val / sig, box2[2] * mv.W / (max(Lc) + 3), 0.0, 2.5); square_axes(ax2, Lc); load_arrows(mv.fig, box2, s=mv.s)
    cax = mv.fig.add_axes([box2[0] + box2[2] + 0.045, box2[1], 0.008, box2[3]]); cb = mv.fig.colorbar(sc, cax=cax); cb.ax.tick_params(labelsize=mv.fs(11))
    frame_fade(mv, box2, 0.6); mv.hold(2.0, poster="s7_crack_load_transfer")
    for k in range(mv.n(0.4)):
        t = (k + 1) / mv.n(0.4); lc.set_alpha(1 - t); sc.set_alpha(1 - t); mv.frame()
    ax2.remove(); cax.remove(); mv.fig.patches.clear()
    tests = TESTS; thumbs, axp = test_layout(mv, mv.fig, tests)
    for k in range(mv.n(0.8)):
        t = (k + 1) / mv.n(0.8)
        for axt in thumbs:
            for c_ in axt.collections: c_.set_alpha(t)
        mv.frame()
    for i, (n_, key) in enumerate(tests):
        axp.plot([i], [PRED[n_]["predicted"][key]], "o", ms=18 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.4 * mv.s, zorder=4); mv.hold(0.3)
    mv.hold(1.4, poster="s7_tests_registered")
    LOG["scene7"] = dict(level_ratio_small=d1.get("scale_ratio"), level_ratio_large=d3.get("scale_ratio"), crack_field=dict(run="H1_veinsX_L20_r0", strain=float(p.eps[kf]), applied_stress_Nm=sig),
                         tests=[t[0] for t in tests], predicted_strength_Nm={t[0]: PRED[t[0]]["predicted"][t[1]] for t in tests}, prediction_files=["hierarchy_followup_predictions.json", "hierarchy_T5_predictions.json"])
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 8: composing the prediction from the rule
def scene_compose(mv):
    mv.begin("8 the prediction is composed from the rule before the 300 Å runs: vein fraction x 28.9 + pore-level fraction x 16.5 N/m per unit section; the alignment trend predicts differently; the simulations decide")
    fs = mv.fs(11); designs = [("H3_1L_fine", "one level, fine"), ("H3_1L_coarse", "one level, coarse"), ("H3_2L", "two levels")]
    C = RREF["classes"]; s_straight, s_fine = C["straight"]["median"], C["fine_round"]["median"]
    thumbs, axp = test_layout(mv, mv.fig, [(n + "_r0", "strength_Nm_rule") for n, _ in designs], y_thumb=0.60, y_axis=0.12, h_axis=0.42, x0=0.10, x1=0.90, alpha=1.0)
    axp.set_ylim(0, 26)
    mv.fade_in(0.5); mv.hold(0.6)
    log = []
    for i, (n, lab) in enumerate(designs):
        p = PRED[n + "_r0"]; col = p["crack_column"]; rule = p["predicted"]["strength_Nm_rule"]; vein = col["SF_vein"] * s_straight; rest = rule - vein
        parts = [(vein, RED)] if vein > 0 else []; parts.append((rest, BLUE if n != "H3_1L_coarse" else RED))
        y0 = 0.0
        for h, c in parts:
            bar = axp.bar([i], [0], 0.42, bottom=y0, color=c, zorder=3)[0]
            for k in range(mv.n(0.6)):
                t = ease((k + 1) / mv.n(0.6)); bar.set_height(h * t); mv.frame()
            y0 += h
        mv.hold(0.2); log.append(dict(design=n, SF_vein=col["SF_vein"], SF_fine=col["SF_fine"], rule=rule, trend=p["predicted"]["strength_Nm_trend"]))
    mv.hold(0.5, poster="s8_rule_composed")
    for i, (n, lab) in enumerate(designs):
        axp.plot([i], [PRED[n + "_r0"]["predicted"]["strength_Nm_trend"]], "D", ms=15 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.2 * mv.s, zorder=4); mv.hold(0.3)
    mv.hold(0.8, poster="s8_trend_prediction")
    for i, (n, lab) in enumerate(designs):
        obs = float(np.mean([OWN[m]["metrics"]["strength_Nm"] for m in (n + "_r0", n + "_r1") if m in OWN])); rule = PRED[n + "_r0"]["predicted"]["strength_Nm_rule"]; log[i]["observed"] = obs
        c = GREEN if abs(obs - rule) / obs <= 0.15 else RED
        ln, = axp.plot([], [], "-", color=c, lw=2.5 * mv.s, zorder=5); dot, = axp.plot([], [], "o", ms=18 * mv.s, markerfacecolor=c, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=6)
        for k in range(mv.n(0.5)):
            t = ease((k + 1) / mv.n(0.5)); yv = rule + t * (obs - rule); ln.set_data([i, i], [rule, yv]); dot.set_data([i], [yv]); mv.frame()
        mv.hold(0.2)
    mv.hold(1.8, poster="s8_observed")
    LOG["scene8"] = dict(class_strengths_Nm=dict(straight=s_straight, fine_round=s_fine), designs=log)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- index
BEATS = {
 "1": "The design language. One 120 Å cell of pristine graphene, then the motifs the agent inferred from the reference images: a single-level mesh of round pores, elongated pores, a load-parallel slit array, a two-level mesh with solid veins around fine pores, and a three-level mesh. Arrows mark the tension axis (x).",
 "2": "Exploration and the first contrast. The 96 discovery runs of phase 1 appear in run order (strength against relative areal density, coloured by family, thumbnails of every fourth design), the family envelopes fade in, and the rule-of-mixtures line through the pristine sheet is drawn: the strengths lie far below it. The inset repeats the construction for the modulus, which does follow the line: mass sets stiffness, not strength.",
 "3": "Hypotheses tested by matched pairs. From its first 40 runs the agent wrote five competing hypotheses with predictions and then ran discriminating pairs. Top row: a coarse square-pore mesh and two round-pore meshes at the same minimum section or the same porosity; hollow circles are the strengths predicted by the hypothesis that necked ligaments are weaker; the observed strengths (filled) refute it, pore shape does not matter at the coarse scale. Bottom row: a nested mesh with veins in both directions (reference, grey), veins only along the load and veins only across it; the predictions hold, load-parallel veins set the strength of a nested mesh.",
 "4": "The first principle. A graded-pore mesh is coloured by the per-atom stress relative to the applied stress before any damage. A band sweeps its columns: the blue curve is the solid fraction of each column, the red curve the applied stress divided by the mean stress in the column. Force balance across every column makes the two coincide (σ_col A_col = σ A), so the column with the least material carries the highest stress and fails first; with a ligament strength σ_lig the strength follows as σ_max = (A_min/A) σ_lig. All 81 porous designs are then placed against their minimum column fraction, and the two lines of the rule appear (30 and 17 N/m per unit section).",
 "5": "Why two lines. A slit array (straight strips) and a fine round-pore mesh (narrow ligaments) coloured by the maximum principal stress per atom at 80 % of their peak strain; at right the strength per unit section of all 57 ordered meshes against their mean ligament width: it falls with the width because edge atoms carry a larger share of a narrow ligament; the two class medians are drawn.",
 "6a": "Predictions before the holdouts. The twelve holdout designs of phase 1 with their strengths predicted and hashed before simulation (hollow, on the diagonal) and the simulated strengths (filled; green within 15 %, red beyond; band ±15 %). Two large misses are framed: the nested mesh with veins and pores both aligned with the load was underestimated by a third, the 20° slit array overestimated by a factor of 2.7.",
 "6b": "A measurable definition of alignment. For a slit array along the load, a square mesh and a slit array across the load, the structure tensor of the solid phase is drawn as an ellipse: its anisotropy a and the angle φ of its ligament axis to the load give A = a cos 2φ, +0.73, +0.03 and −0.62.",
 "6c": "The registered interpolation. The phase-1 meshes appear on the strength–alignment chart (circles one level, squares two levels, triangles three levels); the eight alignment-sweep designs of phase 2 then arrive as hollow circles at the strength predicted by a registered interpolation in A and fill at their simulated values (median error 11 %); the single-level trend is drawn and the stems show every nested mesh below it: the hierarchy premium is −1.6 ± 1.4 N/m. This is where the original campaign stopped.",
 "7": "The decision. Dimension lines on the best nested design: 40 Å vein period, 12 Å veins, 12 Å fine-pore period, a level ratio of three; the cell shrinks to its true size inside the same design at 100 Å, 30 Å and 12 Å, a level ratio of eight. A crack inside a domain is then shown before any damage, coloured by the stress relative to the applied stress: the stiff veins carry the load around it, the first-principles reason to expect arrest. Five tests are laid out with a hollow circle at the strength predicted and hashed for each before it was run.",
 "8": "Composing the prediction. For the three 300 Å designs the bars are built from the rule: the vein fraction of the weakest column times 28.9 N/m (red) plus the pore-level fraction times 16.5 N/m (blue); hollow diamonds are the predictions of the phase-1 alignment trend; the filled circles are the simulated strengths: the column rule is right to within 2 N/m, the alignment trend misses by 3–4 N/m in both directions.",
 "9a": "Mechanisms at 300 Å. The two-level mesh coloured by the load carried per atom before any damage (the veins carry 41 % of the load on 33 % of the atoms), then both meshes loaded to failure with the energy colouring: the one-level mesh tears row after row; the two-level mesh loses its fine level compartment by compartment, each domain crack held at the veins, which break last.",
 "9b": "A contained crack. A 60 Å crack inside a domain of the two-level mesh: no vein bond breaks before the peak, the first damage appears in other domains, and after the peak the crack is deflected along the veins into two branches per tip.",
 "9c": "Fiber-bundle veins. Veins made of load-parallel slit bundles fail strip by strip while the bundle keeps carrying load.",
 "10": "Outcomes. Filled circles land on the observed strengths of the five test designs (green: within 15 % of the hollow prediction); above, each registered hypothesis against its criterion (dashed line: met exactly; green above, red below): only the scale-separation test passed. At right, the strength retained by all 32 cracked runs against the net-section rule: the two-level meshes reach the line, the pristine sheet and the slit array fall far below it, no design lies above it.",
 "11": "Principles. The weakest column sets the strength; hierarchy compartmentalises failure; the building blocks set what happens after the peak.",
}
SCENE_KEYS = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6a": "6a", "6b": "6b", "6c": "6c", "7": "7", "8": "8", "9a": "9a", "9b": "9b", "9c": "9c", "10": "10", "11": "11"}


def write_readme(meta, outdir):
    d = meta; D = d["data"]
    lines = ["## R01: The reasoning trajectory (a wordless film, version 2)", "",
             f"`{d['file']}` ({d['seconds']} s; `R01_reasoning_trajectory_1080p.mp4` is the 1920x1080 version). Generated by `analysis/reasoning_movie.py` (version 2; the 95 s version 1 is kept as `R01_reasoning_trajectory_v1_short.mp4`); scene times and the data behind every scene are in `R01_reasoning_trajectory.json`, poster frames in `R01_frames/`. The film follows how the agent explored hierarchical graphene metamaterials and how its decisions were made, with the first-principles reasoning made visible: the rule-of-mixtures contrast, the force-balance derivation of the weakest-column rule checked against the simulated stress field, the ligament-width origin of the two strength classes, the agent's hypotheses tested by matched pairs, the hashed holdout predictions and their misses, the alignment index built from the structure tensor, the registered interpolation that predicted the alignment sweep, the scale argument behind the follow-up, the prediction composed from the rule before the 300 Å runs, the mechanisms at 300 Å, the outcome of every registered test, and the principles that survived. No words: the only text is numbers, unit symbols, single-symbol axis labels and three symbolic relations. Visual language: hollow circle = predicted before the run, filled = observed (green within 15 %, red beyond); hollow diamond = the competing prediction. Conventions: {d['conventions']}.", "",
             "| time (s) | scene | what is shown |", "|---|---|---|"]
    for sc in d["scenes"]:
        key = sc["scene"].split(" ")[0]; lines.append(f"| {sc['start_s']:.0f}–{sc.get('end_s', d['seconds']):.0f} | {sc['scene'].split(' ', 1)[1]} | {BEATS.get(key, '')} |")
    lines += ["", "Every number is read from the archives: phase-1 reference records and trajectories, `holdout_predictions.json` / `holdout_evaluation.json`, `stage4_predictions.json`, `hypotheses.json`, `paper_sweeps_evaluation.json`, and the follow-up database and prediction files. Frames between stored quasi-static states are interpolated in the fracture scenes; the stress fields are stored states.", ""]
    open(os.path.join(outdir, "R01_README.md"), "w").write("# " + lines[0][3:] + "\n" + "\n".join(lines[1:]))
    readme = os.path.join(outdir, "README.md")
    if os.path.exists(readme):
        txt = open(readme).read()
        if "\n## R01" in txt: txt = txt[:txt.index("\n## R01")]
        open(readme, "w").write(txt.rstrip("\n") + "\n\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); ap.add_argument("--outdir", default=os.path.join(ROOT, "movies_hq")); ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    width, fps, crf, preset = (960, 8, 28, "ultrafast") if a.preview else (2560, 30, 17, "slow")
    os.makedirs(os.path.join(a.outdir, "R01_frames"), exist_ok=True)
    base = os.path.join(a.outdir, "R01_reasoning_trajectory" + ("_preview" if a.preview else ""))
    mv = Movie(width, fps, base + ".mp4", crf=crf, preset=preset); t0 = time.time()
    scenes = [("1", lambda: scene_language(mv)), ("2", lambda: scene_exploration(mv)), ("3", lambda: scene_hypotheses(mv)), ("4", lambda: scene_rule(mv)), ("5", lambda: scene_ligaments(mv)),
              ("6a", lambda: scene_holdout(mv)), ("6b", lambda: scene_alignment_index(mv)), ("6c", lambda: scene_sweep(mv)), ("7", lambda: scene_decision(mv)), ("8", lambda: scene_compose(mv)),
              ("9a", lambda: fracture_scene(mv, "9a mechanisms at 300 Å: one level against two levels", ["H3_1L_fine_r0", "H3_2L_r0"], 7.5, "s9a", vein_load_intro=True)),
              ("9b", lambda: fracture_scene(mv, "9b a contained crack is held at the veins and deflected", ["H3_2L_L60_r0"], 5.0, "s9b")),
              ("9c", lambda: fracture_scene(mv, "9c fiber-bundle veins fail strip by strip", ["H5_VF_E_r0"], 5.0, "s9c")),
              ("10", lambda: scene_outcomes(mv)), ("11", lambda: scene_principles(mv))]
    with mv.writer.saving(mv.fig, base + ".mp4", mv.dpi):
        for key, fn in scenes:
            if not a.only or key in a.only: fn()
    plt.close(mv.fig)
    # scene names of the reused v1 scenes keep their own numbering prefix; normalise it for the index
    ren = {"7 outcomes": "10 outcomes", "8 principles": "11 principles"}
    for sc in mv.scene_log:
        for k_, v_ in ren.items():
            if sc["scene"].startswith(k_): sc["scene"] = v_ + sc["scene"][len(k_):]
    meta = dict(file=os.path.basename(base) + ".mp4", resolution=f"{mv.W}x{mv.H}", fps=fps, seconds=round(mv.nframes / fps, 1), scenes=mv.scene_log, data=LOG,
                text_in_movie="numbers, unit symbols, single-symbol axis labels and three symbolic relations (force balance, the net-section rule, the alignment index)",
                conventions="white background; structures: dark atoms, grey bonds; stress fields: per-atom stress (inferno); fracture: energy change per atom (inferno, truncated), red rings = atoms that lost a bond, panel frame colour = curve colour, frames between stored states interpolated; model results only (screened REBO2, athermal quasi-static tension along x)",
                render_minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(base + ".json", "w"), indent=1)
    if not a.preview:
        compact(base); write_readme(meta, a.outdir)
    print("done", base, f"{mv.nframes / fps:.1f} s", f"{(time.time() - t0) / 60:.1f} min")
