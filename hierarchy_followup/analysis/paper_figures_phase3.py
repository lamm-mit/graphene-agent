"""Draft paper figures of the hierarchy follow-up, in the style of paper/make_figures.py (Arial, 7.2 in wide, panel
letters, dark atoms / grey bonds / red damaged atoms, no top-right spines).  Every number is read from this tree's
database; the phase-1 tree is not touched.  Outputs figures/hierarchy/paper/<name>.{svg,png,pdf} and numbers_phase3.json.

  fig_scale        hierarchy pays when the levels are separated: designs, curves, strength/work at 300 vs 120 A,
                   load sharing, compartment-by-compartment failure against run-through and single-avalanche failure
  fig_flaw         flaw tolerance: observed notch retention against the net-section rule for every cracked design,
                   cracked vs intact curves, crack paths (held / running / branching), work to failure under a crack
  fig_blocks       what the building blocks buy: post-peak curves, post-peak energy, strip-by-strip failure of fibre
                   bundles against snapping solid veins and sub-vein compartments

    python analysis/paper_figures_phase3.py
"""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from experiments import db
from analysis.fracture_viz import load_traj, select_event_frames, damaged_atoms_until
from analysis.assess_phase1 import curve_metrics
from atomistics.descriptors.columns import vein_bands

OUT = os.path.join(ROOT, "figures", "hierarchy", "paper"); os.makedirs(OUT, exist_ok=True)
EV = 16.0217663
DARK, RED, BLUE, ORANGE, GREEN, GREY, PURPLE, BROWN = "#222222", "#c0392b", "#1f77b4", "#ff7f0e", "#2ca02c", "0.55", "#9467bd", "#8c564b"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none"})
OWN = {r["name"]: r for r in db.own_records() if r.get("status") == "completed"}
NUM = {}
_TRAJ = {}


# --------------------------------------------------------------------------- helpers
def axin(fig, x, ytop, w, h, FW, FH, **kw):
    return fig.add_axes([x / FW, 1 - (ytop + h) / FH, w / FW, h / FH], **kw)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.2, lw=0.4); ax.set_axisbelow(True)


def letter(ax, s, dx=-0.02, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def traj(name):
    if name not in _TRAJ:
        _TRAJ[name] = load_traj(db.trajectory_path(OWN[name]))
    return _TRAJ[name]


def frame_for(T, key):
    for lab, f in select_event_frames(T):
        if key in lab.split(" = "): return f
    return 0 if key == "relaxed" else (len(T["eps_x"]) - 1 if key == "final" else int(np.argmax(T["sigma_xx"])))


def first_drop_frame(T, drop=0.15):
    s = T["sigma_xx"]; ip = int(np.argmax(s))
    for k in range(ip + 1, len(s)):
        if s[k - 1] - s[k] > drop * s[ip]: return k
    return min(ip + 2, len(s) - 1)


def snap(ax, name, k, crop=None, bands=True, s=None, sdmg=None, lw=None):
    """Atoms (dark), bonds (grey), atoms that lost a bond (red); load-parallel veins shaded behind."""
    rec = OWN[name]; T = traj(name)
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    for a in range(2): P[:, a] = np.mod(P[:, a], c[a, a])
    big = c[0, 0] > 200
    s = s or (0.12 if big else 0.35); sdmg = sdmg or (1.2 if big else 2.4); lw = lw or (0.1 if big else 0.22)
    if bands:
        for yc, hw, sy in (vein_bands(rec.get("design") or {}) or []):
            for y0 in (yc - hw, yc - hw - c[1, 1], yc - hw + c[1, 1]):
                ax.add_patch(Rectangle((-5, y0), c[0, 0] + 10, 2 * hw, facecolor="#e9eef5", edgecolor="none", zorder=0))
    if len(b):
        p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]; d = p1 - p0
        for a in range(2): d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
        ax.add_collection(LineCollection(np.stack([p0, p0 + d], 1), colors="0.45", linewidths=lw, zorder=1))
    ax.scatter(P[:, 0], P[:, 1], s=s, c="#2b2b2b", linewidths=0, zorder=2)
    dmg = damaged_atoms_until(rec, T["eps_x"][k])
    if len(dmg): ax.scatter(P[dmg, 0], P[dmg, 1], s=sdmg, c=RED, linewidths=0, zorder=3)
    if crop:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
    else:
        ax.set_xlim(-1.5, c[0, 0] + 1.5); ax.set_ylim(-1.5, c[1, 1] + 1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    return float(T["eps_x"][k]), float(T["sigma_xx"][k] * EV), int(T["n_broken_cum"][k])


def under(ax, s, color=DARK, dy=-0.03, fs=6.3):
    ax.text(0.5, dy, s, transform=ax.transAxes, ha="center", va="top", fontsize=fs, color=color, linespacing=1.15)


def two(names, key):
    v = [OWN[n]["metrics"][key] for n in names if n in OWN]
    return float(np.mean(v)), (float(np.ptp(v)) / 2 if len(v) > 1 else 0.0)


def curve(name):
    c = db.stress_strain(OWN[name]); return c["eps_x"], c["sigma_xx_Nm"]


def save(fig, name):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "paper_analysis", "current"))
        import check_overlaps as CO
        n = CO.audit(fig, name, verbose=True); print(f"overlap audit {name}: {n} conflicts")
    except Exception as e:
        print("overlap audit skipped:", repr(e)[:80])
    for ext in ("svg", "png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- figure 1: scale separation
def fig_scale():
    FW, FH = 7.2, 7.35
    fig = plt.figure(figsize=(FW, FH))
    # (a) designs at the same physical scale
    y0 = 0.22; H = 1.5
    items = [("H3_1L_fine_r0", "one level, fine pores", 0.35), ("H3_1L_coarse_r0", "one level, coarse bars", 2.0), ("H3_2L_r0", "two levels: 30 Å veins every 100 Å", 3.65), ("H1_veinsX_r0", "two levels at 120 Å\n(12 Å veins every 40 Å)", 5.45)]
    for k, (n, lab, x) in enumerate(items):
        w = H if n.startswith("H3") else H * 120.55 / 300.1
        ax = axin(fig, x, y0 + (H - w), w, w, FW, FH)
        snap(ax, n, 0)
        s, ds = two([n, n.replace("_r0", "_r1")], "strength_Nm"); W, dW = two([n, n.replace("_r0", "_r1")], "work_to_failure_J_m2")
        under(ax, f"{lab}\nσ = {s:.1f} N/m, W = {W:.2f} J/m²", dy=-0.04, fs=6.3)
        if k == 0: letter(ax, "a", dx=-0.06)
    # (b) curves at 300 A
    ax = axin(fig, 0.5, 2.45, 2.05, 1.65, FW, FH); style(ax); letter(ax, "b", dy=1.2)
    for n, col, lab in (("H3_1L_fine", BLUE, "one level, fine"), ("H3_1L_coarse", GREEN, "one level, coarse"), ("H3_2L", RED, "two levels")):
        for reg, ls in ((0, "-"), (1, ":")):
            e, s = curve(f"{n}_r{reg}"); ax.plot(e, s, ls, color=col, lw=1.0, label=lab if reg == 0 else None)
    ax.set_xlabel("engineering strain", labelpad=1); ax.set_ylabel("2D stress (N/m)", labelpad=1); ax.set_xlim(0, 0.2); ax.set_ylim(0, 21)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.5, columnspacing=1.0, handletextpad=0.5, fontsize=6.4)
    # (c) strength and work at 300 and 120 A
    ax = axin(fig, 3.0, 2.45, 2.25, 1.65, FW, FH); style(ax); letter(ax, "c", dy=1.2)
    sets = [("300 Å", [("H3_1L_fine", BLUE, "one level, fine"), ("H3_1L_coarse", GREEN, "one level, coarse"), ("H3_2L", RED, "two levels")]),
            ("120 Å", [("H1_ellipse", BLUE, "one level, fine"), ("H1_slit", ORANGE, "one level, slits"), ("H1_veinsX", RED, "two levels")])]
    xs = []; xt = []
    for g, (lab, items) in enumerate(sets):
        for m, (key, fac, ylab) in enumerate((("strength_Nm", 1.0, "strength (N/m)"), ("work_to_failure_J_m2", 10.0, "work to failure (J/m²) × 10"))):
            x0 = g * 2.3 + m * 1.0
            for i, (n, col, _) in enumerate(items):
                v, dv = two([f"{n}_r0", f"{n}_r1"], key); v, dv = v * fac, dv * fac
                ax.bar(x0 + i * 0.26, v, 0.24, color=col, yerr=dv, capsize=1.5, error_kw=dict(lw=0.6), zorder=3)
                ax.text(x0 + i * 0.26, v + dv + 0.6, f"{v:.0f}" if fac == 1 else f"{v/10:.1f}", ha="center", va="bottom", fontsize=5.6)
            xt.append((x0 + 0.26, ylab.split(" (")[0].replace("work to failure", "work") + ("" if fac == 1 else " ×10")))
        ax.text(g * 2.3 - 0.2, 39.5, lab + " cell", ha="left", va="top", fontsize=7.2, fontweight="bold")
    ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=6.5); ax.set_ylim(0, 40); ax.set_ylabel("N/m  |  J/m² × 10", labelpad=1)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=5), Line2D([], [], color=GREEN, lw=5), Line2D([], [], color=ORANGE, lw=5), Line2D([], [], color=RED, lw=5)],
              labels=["one level, fine", "one level, coarse", "one level, slits", "two levels"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2, handlelength=1.2, ncol=2, columnspacing=1.2, handletextpad=0.5)
    NUM.update(sigma_2L_300=two(["H3_2L_r0", "H3_2L_r1"], "strength_Nm")[0], sigma_fine_300=two(["H3_1L_fine_r0", "H3_1L_fine_r1"], "strength_Nm")[0], sigma_coarse_300=two(["H3_1L_coarse_r0", "H3_1L_coarse_r1"], "strength_Nm")[0],
               W_2L_300=two(["H3_2L_r0", "H3_2L_r1"], "work_to_failure_J_m2")[0], W_fine_300=two(["H3_1L_fine_r0", "H3_1L_fine_r1"], "work_to_failure_J_m2")[0], W_coarse_300=two(["H3_1L_coarse_r0", "H3_1L_coarse_r1"], "work_to_failure_J_m2")[0])
    # (d) load share before the peak in the two-level design
    ax = axin(fig, 5.5, 2.45, 1.5, 1.5, FW, FH); letter(ax, "d", dy=1.06)
    T = traj("H3_2L_r0"); ip = int(np.argmax(T["sigma_xx"])); k = int(np.searchsorted(T["eps_x"], 0.8 * T["eps_x"][ip]))
    P = T["positions"][k].astype(float); c = T["cells"][k]; v = T["peratom_virial"][k][:, 0].astype(float); share = v / v.mean()
    sc = ax.scatter(P[:, 0], P[:, 1], c=share, s=0.25, cmap="magma", vmin=0, vmax=2, linewidths=0)
    ax.set_xlim(0, c[0, 0]); ax.set_ylim(0, c[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    bands = vein_bands(OWN["H3_2L_r0"]["design"]); from atomistics.descriptors.columns import in_vein
    inv = np.array([in_vein(y, bands) for y in P[:, 1]]); NUM["vein_load_share"] = float(v[inv].sum() / v.sum())
    cax = axin(fig, 7.08, 2.45, 0.06, 1.5, FW, FH); cb = fig.colorbar(sc, cax=cax, orientation="vertical"); cb.set_label("load per atom / mean", fontsize=6.0, labelpad=1); cb.ax.tick_params(labelsize=5.6, length=2)
    under(ax, f"load before the peak (ε = {T['eps_x'][k]:.2f}): veins carry\n{100*NUM['vein_load_share']:.0f} % of the load on 30 % of the section", dy=-0.04, fs=6.0)
    # (e) how they fail
    y0 = 4.85; H = 1.55
    panels = [("H3_2L_r0", first_drop_frame(traj("H3_2L_r0")), "two levels, after the first drop:\ncracks confined to one compartment"),
              ("H3_2L_r0", frame_for(traj("H3_2L_r0"), "major_propagation"), "two levels, major propagation:\ncompartments fail one by one, veins last"),
              ("H3_1L_fine_r0", first_drop_frame(traj("H3_1L_fine_r0")), "one level, fine: cracks run\nthrough row after row"),
              ("H3_1L_coarse_r0", first_drop_frame(traj("H3_1L_coarse_r0")), "one level, coarse: all bars\nfail in one avalanche")]
    for k, (n, f, lab) in enumerate(panels):
        ax = axin(fig, 0.35 + k * 1.72, y0, H, H, FW, FH)
        e, s, nb = snap(ax, n, f)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m, {nb} bonds broken", dy=-0.04, fs=6.0)
        if k == 0: letter(ax, "e", dx=-0.06)
    save(fig, "fig_scale")


# --------------------------------------------------------------------------- figure 2: flaw tolerance
def fig_flaw():
    FW, FH = 7.2, 8.45
    fig = plt.figure(figsize=(FW, FH))
    preds = {}
    for f in ("hierarchy_followup_predictions.json", "hierarchy_T5_predictions.json"):
        preds.update({p["name"]: p for p in json.load(open(os.path.join(ROOT, "experiments", "predictions", f)))["predictions"]})
    # cracked designs: (cracked name, uncracked name, class)
    pairs = []
    for reg in (0, 1):
        for cl in (20, 40):
            pairs += [(f"H1_pristine_L{cl}_r{reg}", None, "pristine sheet"), (f"H1_ellipse_L{cl}_r{reg}", f"H1_ellipse_r{reg}", "one level"), (f"H1_slit_L{cl}_r{reg}", f"H1_slit_r{reg}", "one level, slits"),
                      (f"H1_veinsX_L{cl}_r{reg}", f"H1_veinsX_r{reg}", "two levels, solid veins")]
        pairs += [(f"H3_2L_L60_r{reg}", f"H3_2L_r{reg}", "two levels, solid veins"), (f"H3_1L_fine_L60_r{reg}", f"H3_1L_fine_r{reg}", "one level")]
    pairs += [("H3_1L_coarse_L60_r0", "H3_1L_coarse_r0", "one level"), ("H5_VF_E_L60_r0", "H5_VF_E_r0", "fibrous veins"), ("H5_VF_E_L60_r1", "H5_VF_E_r1", "fibrous veins"), ("H5_VF_R_L60_r0", "H5_VF_R_r0", "fibrous veins"),
              ("H5_VS_R_L60_r0", "H5_VS_R_r0", "two levels, solid veins"), ("H5_VE_R_L60_r0", "H5_VE_R_r0", "porous veins"), ("H5_VR_E_L60_r0", "H5_VR_E_r0", "porous veins"), ("H5_V3_E_L60_r0", "H5_V3_E_r0", "three levels"),
              ("H5_GR_0_L40_r0", "H5_GR_0_r0", "grids"), ("H5_GR_0_L40_r1", "H5_GR_0_r1", "grids"), ("H5_GFx_0_L40_r0", "H5_GFx_0_r0", "grids"), ("H5_GS_L40_r0", "H5_GS_r0", "one level, solid bars")]
    COL = {"pristine sheet": GREY, "one level": BLUE, "one level, slits": ORANGE, "one level, solid bars": "#17becf", "two levels, solid veins": RED, "fibrous veins": PURPLE, "porous veins": "#e377c2", "three levels": BROWN, "grids": GREEN}
    ax = axin(fig, 0.5, 0.25, 2.75, 2.2, FW, FH); style(ax); letter(ax, "a")
    rows = []
    for cr, unc, cls in pairs:
        if cr not in OWN: continue
        s_c = OWN[cr]["metrics"]["strength_Nm"]; s_u = 38.79 if unc is None else OWN[unc]["metrics"]["strength_Nm"]
        rule = preds[cr]["predicted"]["retention_rule"]; ret = s_c / s_u; big = OWN[cr]["descriptors"]["Lx"] > 200
        Wr = None if unc is None else OWN[cr]["metrics"]["work_to_failure_J_m2"] / OWN[unc]["metrics"]["work_to_failure_J_m2"]
        rows.append(dict(name=cr, cls=cls, rule=rule, ret=ret, big=big, Wr=Wr))
        ax.scatter([rule], [ret], s=24, facecolors=COL[cls] if big else "white", edgecolors=COL[cls], linewidths=1.1 if not big else 0.4, zorder=3)
    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color=DARK, lw=0.8); ax.text(0.625, 0.612, "crack costs its section", fontsize=6.3, ha="left", va="top", color=DARK, rotation=39)
    ax.fill_between([0.5, 1.0], [0.5, 1.0], [0.3, 0.3], color="0.94", zorder=0); ax.text(0.945, 0.375, "notch-sensitive", fontsize=6.5, ha="right", color=GREY)
    ax.set_xlim(0.6, 0.97); ax.set_ylim(0.35, 1.0); ax.set_xlabel("net-section rule: section left by the crack", labelpad=1); ax.set_ylabel("strength with crack / strength without", labelpad=1)
    hs = [Line2D([], [], marker="o", color=COL[c], lw=0, ms=4.5) for c in COL] + [Line2D([], [], marker="o", color=DARK, lw=0, ms=4.5), Line2D([], [], marker="o", markerfacecolor="white", color=DARK, lw=0, ms=4.5)]
    hs = [Line2D([], [], marker="o", markerfacecolor=COL[c], markeredgecolor=COL[c], lw=0, ms=4.5) for c in COL] + [Line2D([], [], marker="o", markerfacecolor=DARK, markeredgecolor=DARK, lw=0, ms=4.5), Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=DARK, lw=0, ms=4.5)]
    fig.legend(handles=hs, labels=list(COL) + ["filled: 300 Å cell", "open: 120 Å cell"], frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(0.45 / FW, 1 - 2.95 / FH), ncol=3, handletextpad=0.3, columnspacing=1.0)
    NUM["n_cracked_designs"] = len(rows)
    # (b, c) cracked vs intact curves
    for j, (x, title, items, xl) in enumerate(((3.75, "300 Å, 60 Å crack", [("H3_2L_r0", "H3_2L_L60_r0", RED, "two levels"), ("H3_1L_fine_r0", "H3_1L_fine_L60_r0", BLUE, "one level, fine")], 0.25),
                                              (5.7, "120 Å, 20 Å crack", [("H1_veinsX_r0", "H1_veinsX_L20_r0", RED, "two levels"), ("H1_ellipse_r0", "H1_ellipse_L20_r0", BLUE, "one level, fine")], 0.24))):
        ax = axin(fig, x, 0.25, 1.4, 2.2, FW, FH); style(ax); letter(ax, "bc"[j])
        for unc, cr, col, lab in items:
            e, s = curve(unc); ax.plot(e, s, "-", color=col, lw=1.0, label=lab + ", intact")
            e, s = curve(cr); ax.plot(e, s, "--", color=col, lw=1.0, label=lab + ", cracked")
        ax.set_xlim(0, xl); ax.set_ylim(0, 21); ax.set_xlabel("engineering strain", labelpad=1); ax.set_title(title, fontsize=7.5, pad=2)
        if j == 0: ax.set_ylabel("2D stress (N/m)", labelpad=1)
        if j == 1:
            fig.legend(handles=[Line2D([], [], color=RED, lw=1.0), Line2D([], [], color=RED, lw=1.0, ls="--"), Line2D([], [], color=BLUE, lw=1.0), Line2D([], [], color=BLUE, lw=1.0, ls="--")],
                       labels=["two levels, intact", "two levels, cracked", "one level (fine), intact", "one level (fine), cracked"], frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(4.55 / FW, 1 - 2.95 / FH), ncol=1, handlelength=1.8)
    # (d) crack paths
    y0 = 3.95; H = 1.12
    panels = [("H1_veinsX_L20_r0", frame_for(traj("H1_veinsX_L20_r0"), "peak_load"), "two levels, at the peak:\ncrack held, damage elsewhere"),
              ("H1_ellipse_L20_r0", frame_for(traj("H1_ellipse_L20_r0"), "peak_load"), "one level, at the peak:\ndamage at the crack tips"),
              ("H1_slit_L20_r0", first_drop_frame(traj("H1_slit_L20_r0")), "slits, after the drop:\nneighbouring strips fail"),
              ("H3_2L_L60_r0", frame_for(traj("H3_2L_L60_r0"), "peak_load"), "two levels, at the peak:\nheld at the veins"),
              ("H3_2L_L60_r0", frame_for(traj("H3_2L_L60_r0"), "post_peak"), "two levels, after:\nbranches across the veins"),
              ("H3_1L_fine_L60_r0", frame_for(traj("H3_1L_fine_L60_r0"), "post_peak"), "one level, after:\nstraight tearing")]
    for k, (n, f, lab) in enumerate(panels):
        ax = axin(fig, 0.3 + k * 1.16, y0, H, H, FW, FH)
        e, s, nb = snap(ax, n, f)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m", dy=-0.04, fs=5.5)
        if k == 0: letter(ax, "d", dx=-0.08)
    fig.text(0.3 / FW, 1 - (y0 - 0.06) / FH, "120 Å cells, 20 Å crack", fontsize=6.5, color=GREY, va="bottom"); fig.text((0.3 + 3 * 1.16) / FW, 1 - (y0 - 0.06) / FH, "300 Å cells, 60 Å crack (not to the same scale)", fontsize=6.5, color=GREY, va="bottom")
    # (e) work to failure under the crack
    ax = axin(fig, 0.5, 6.0, 6.55, 1.55, FW, FH); style(ax); letter(ax, "e")
    order = ["pristine sheet", "one level", "one level, slits", "one level, solid bars", "porous veins", "grids", "two levels, solid veins", "three levels", "fibrous veins"]
    for cr, unc, cls in pairs:
        if cr not in OWN: continue
    groups = {}
    for r in rows:
        if r["Wr"] is not None: groups.setdefault(r["cls"], []).append(r["Wr"])
    prist = [OWN[n]["metrics"]["work_to_failure_J_m2"] / 6.73 for n in ("H1_pristine_L20_r0", "H1_pristine_L20_r1", "H1_pristine_L40_r0", "H1_pristine_L40_r1")]
    groups["pristine sheet"] = prist
    xs = np.arange(len(order))
    for i, cls in enumerate(order):
        v = groups.get(cls, [])
        if not v: continue
        ax.bar(i, np.mean(v), 0.62, color=COL[cls], zorder=3); ax.scatter([i] * len(v), v, s=9, color=DARK, zorder=4, linewidths=0)
        ax.text(i, max(v) + 0.05, f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color=DARK, lw=0.6, ls=":"); ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=6.4, rotation=18, ha="right"); ax.set_ylim(0, 1.75)
    ax.set_ylabel("work to failure with crack\n/ without", labelpad=1); ax.text(-0.45, 1.03, "no loss", fontsize=6.3, va="bottom", ha="left", color=DARK)
    NUM["W_ratio_by_class"] = {k: float(np.mean(v)) for k, v in groups.items()}
    save(fig, "fig_flaw")


# --------------------------------------------------------------------------- figure 3: building blocks
def fig_blocks():
    FW, FH = 7.2, 5.6
    fig = plt.figure(figsize=(FW, FH))
    items = [("H3_1L_fine", BLUE, "one level, fine"), ("H3_1L_coarse", GREEN, "one level, coarse"), ("H3_2L", RED, "two levels, solid veins"), ("H5_VF_E", PURPLE, "two levels, fibrous veins"), ("H5_V3_E", BROWN, "three levels"), ("H5_GR_0", "#17becf", "mesh of meshes")]
    # (a) post-peak curves, normalised
    ax = axin(fig, 0.5, 0.25, 2.9, 2.15, FW, FH); style(ax); letter(ax, "a")
    for n, col, lab in items:
        for reg, ls in ((0, "-"), (1, ":")):
            if f"{n}_r{reg}" not in OWN: continue
            e, s = curve(f"{n}_r{reg}"); ip = int(np.argmax(s))
            ax.plot(e[ip:] - e[ip], s[ip:] / s[ip], ls, color=col, lw=1.0, label=lab if reg == 0 else None)
    ax.set_xlim(0, 0.2); ax.set_ylim(0, 1.05); ax.set_xlabel("strain beyond the peak", labelpad=1); ax.set_ylabel("stress / peak stress", labelpad=1)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=1.6)
    # (b) post-peak energy
    ax = axin(fig, 3.95, 0.25, 3.1, 2.15, FW, FH); style(ax); letter(ax, "b")
    for i, (n, col, lab) in enumerate(items):
        vals = [curve_metrics(OWN[f"{n}_r{reg}"])["post_peak_energy_J_m2"] for reg in (0, 1) if f"{n}_r{reg}" in OWN]
        ax.bar(i, np.mean(vals), 0.62, color=col, zorder=3); ax.scatter([i] * len(vals), vals, s=9, color=DARK, zorder=4, linewidths=0)
        ax.text(i, max(vals) + 0.03, f"{np.mean(vals):.2f}", ha="center", va="bottom", fontsize=6)
        NUM[f"post_peak_energy_{n}"] = float(np.mean(vals))
    ax.set_xticks(range(len(items))); ax.set_xticklabels([lab.replace(", ", ",\n").replace("three levels", "three\nlevels").replace("mesh of meshes", "mesh of\nmeshes") for _, _, lab in items], fontsize=6.2); ax.set_ylabel("energy absorbed after the peak (J/m²)", labelpad=1); ax.set_ylim(0, 1.45)
    # (c) how a vein fails: solid vein snapping, fibre bundle strip by strip, sub-vein compartments
    y0 = 3.05; H = 1.55; W = 2.1
    panels = [("H3_2L_r0", frame_for(traj("H3_2L_r0"), "major_propagation"), (60, 210, 80, 190), "solid vein: snaps once the compartments\non both sides have failed"),
              ("H5_VF_E_r0", first_drop_frame(traj("H5_VF_E_r0")), (60, 210, 80, 190), "fibre-bundle vein: strips fail one at a time,\nthe bundle keeps carrying load"),
              ("H5_V3_E_r0", frame_for(traj("H5_V3_E_r0"), "post_peak"), "auto", "three levels: sub-veins split the domain\ninto smaller compartments that fail separately")]
    for k, (n, f, crop, lab) in enumerate(panels):
        if crop == "auto":            # window around the densest damage at that frame
            T = traj(n); c = T["cells"][f]; dmg = damaged_atoms_until(OWN[n], T["eps_x"][f]); Pd = np.mod(T["positions"][f][dmg, :2].astype(float), [c[0, 0], c[1, 1]])
            best = None
            for x0 in np.arange(0, c[0, 0] - 150, 10):
                for y0_ in np.arange(0, c[1, 1] - 110, 10):
                    m = int(((Pd[:, 0] > x0) & (Pd[:, 0] < x0 + 150) & (Pd[:, 1] > y0_) & (Pd[:, 1] < y0_ + 110)).sum())
                    if best is None or m > best[0]: best = (m, x0, y0_)
            crop = (best[1], best[1] + 150, best[2], best[2] + 110)
        ax = axin(fig, 0.3 + k * 2.3, y0, W, H, FW, FH)
        e, s, nb = snap(ax, n, f, crop=crop, s=0.5, sdmg=2.2, lw=0.18)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m", dy=-0.04, fs=6.0)
        if k == 0: letter(ax, "c", dx=-0.05)
    fig.text(0.3 / FW, 1 - (y0 - 0.06) / FH, "150 × 110 Å windows of the 300 Å cells around one vein (shaded)", fontsize=6.5, color=GREY, va="bottom")
    save(fig, "fig_blocks")




# --------------------------------------------------------------------------- figure 4: stress, bond-strain and damage fields
R0_BOND = 1.4204   # relaxed C-C bond of REBO2+S (A)


def peratom_stress(T, k):
    """Per-atom in-plane stress (N/m) from the per-atom virial (xx, yy, xy), area per atom = cell area / N; returns the maximum principal stress."""
    c = T["cells"][k]; v = T["peratom_virial"][k].astype(float); N = v.shape[0]; A = c[0, 0] * c[1, 1]
    s = v / (A / N) * EV; sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
    return 0.5 * (sxx + syy) + np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)


def bond_strain(T, k):
    """Largest stretch of the bonds of each atom, (r - r0) / r0 with r0 the relaxed bond length (bonds broken earlier are absent)."""
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    d = P[b[:, 1], :2] - P[b[:, 0], :2]
    for a in range(2): d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
    e = (np.linalg.norm(d, axis=1) - R0_BOND) / R0_BOND
    out = np.zeros(len(P)); np.maximum.at(out, b[:, 0], e); np.maximum.at(out, b[:, 1], e)
    return out


def field(ax, name, k, mode, s=None, dmg=True):
    T = traj(name); rec = OWN[name]; P = T["positions"][k].astype(float); c = T["cells"][k]
    for a in range(2): P[:, a] = np.mod(P[:, a], c[a, a])
    if mode == "conc":       # stress concentration: maximum principal stress per atom / applied stress
        val = peratom_stress(T, k) / (T["sigma_xx"][k] * EV); cmap, vmin, vmx = "inferno", 0.0, 3.0
    elif mode == "bondstrain":
        val = bond_strain(T, k); cmap, vmin, vmx = "plasma", 0.0, 0.25
    else:
        val = T["peratom_energy"][k].astype(float) - T["peratom_energy"][0].astype(float); cmap, vmin, vmx = "magma", 0.0, 1.5
    big = c[0, 0] > 200; s = s or (0.22 if big else 1.3)
    sc = ax.scatter(P[:, 0], P[:, 1], c=val, s=s, cmap=cmap, vmin=vmin, vmax=vmx, linewidths=0, zorder=2)
    if dmg and mode == "energy":
        d = damaged_atoms_until(rec, T["eps_x"][k])
        if len(d): ax.scatter(P[d, 0], P[d, 1], s=s * 3.0, facecolors="none", edgecolors=RED, linewidths=0.3, zorder=3)
    ax.set_xlim(0, c[0, 0]); ax.set_ylim(0, c[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    return sc, float(T["eps_x"][k]), float(T["sigma_xx"][k] * EV), int(T["n_broken_cum"][k])


def fig_fields():
    FW, FH = 7.2, 8.35
    fig = plt.figure(figsize=(FW, FH))
    rows = [("H3_2L_L60_r0", "300 Å, two levels\n+ 60 Å crack"), ("H3_1L_fine_L60_r0", "300 Å, one level (fine)\n+ 60 Å crack"),
            ("H1_veinsX_L20_r0", "120 Å, two levels\n+ 20 Å crack"), ("H1_ellipse_L20_r0", "120 Å, one level (fine)\n+ 20 Å crack")]
    cols = [("conc", "stress concentration at ε = 0.08\n(max. principal stress / applied stress)"), ("bondstrain", "bond strain at the peak\n(largest bond stretch per atom)"),
            ("energy", "damage after the first drop\n(energy change per atom)"), ("energy", "damage at the end")]
    cb_label = {"conc": "$\\sigma_1$ / $\\sigma_\\mathrm{applied}$", "bondstrain": "bond strain ($r$ − $r_0$) / $r_0$", "energy": "energy change per atom (eV)"}
    H = 1.5; x0 = 0.95; y0 = 0.42; scs = {}
    for i, (n, lab) in enumerate(rows):
        T = traj(n); ip = int(np.argmax(T["sigma_xx"])); k8 = int(np.searchsorted(T["eps_x"], 0.08)); kd = first_drop_frame(T); kf = len(T["eps_x"]) - 1
        for j, ((mode, title), k) in enumerate(zip(cols, (k8, ip, kd, kf))):
            ax = axin(fig, x0 + j * (H + 0.08), y0 + i * (H + 0.36), H, H, FW, FH)
            sc, e, sig, nb = field(ax, n, k, mode); scs[mode] = sc
            under(ax, f"ε = {e:.3f}, σ = {sig:.1f} N/m" + ("" if mode != "energy" else f", {nb} bonds broken"), dy=-0.03, fs=5.8)
            if i == 0: ax.set_title(title, fontsize=6.6, pad=3)
            if j == 0:
                ax.text(-0.06, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=6.6, linespacing=1.2)
                letter(ax, "abcd"[i], dx=-0.62, dy=0.9)
    ycb = y0 + 4 * (H + 0.36) - 0.05
    for j, (mode, _) in enumerate(cols[:3]):
        cax = axin(fig, x0 + j * (H + 0.08) + 0.1, ycb, H - 0.2, 0.07, FW, FH); cb = fig.colorbar(scs[mode], cax=cax, orientation="horizontal")
        cb.set_label(cb_label[mode], fontsize=6.0, labelpad=1); cb.ax.tick_params(labelsize=5.6, length=2)
    fig.text(x0 / FW, 1 - (ycb + 0.45) / FH, "Full cells, load along x. Red rings: atoms that have lost a bond. Each pair of rows has the same crack-to-cell ratio; the 300 Å rows are shown at a quarter of the 120 Å magnification.", fontsize=6.0, color=GREY, va="top")
    save(fig, "fig_fields")


if __name__ == "__main__":
    fig_scale(); fig_flaw(); fig_blocks(); fig_fields()
    json.dump(NUM, open(os.path.join(OUT, "numbers_phase3.json"), "w"), indent=1)
    print("wrote", OUT)
