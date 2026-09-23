"""Mechanism overview figure for the main text: figures/fig_mechanisms.{svg,png,pdf}.
One row per mechanism, each with a clean schematic, an atomistic snapshot of the failure and one measurement:
  a  the rule: strength = weakest section x ligament strength (ligament strength by ligament class)
  b  where the rule breaks: tilting a slit array passes through three failure mechanisms (full angle sweep)
  c  hierarchy adds no premium: strength follows the load-path alignment index A
The slit schematics are drawn from the generator parameters of the simulated designs (same periods, slit
length, slit width and angle), so they are geometrically faithful.  Every number comes from the experiment
database.  Also written: figures/mech_numbers.{json,tex} (macros for caption and text) and the LaTeX snippet
figures/fig_mechanisms_snippet.tex (macro version) + fig_mechanisms_snippet_resolved.tex (numbers filled in).
The figure is checked with the overlap audit of check_overlaps.py (must report 0 conflicts)."""
from __future__ import annotations
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
import make_figures as MF            # database records, helpers, Arial rcParams
import check_overlaps as CO          # overlap audit
from analysis.fracture_viz import load_traj, select_event_frames, damaged_atoms_until
from experiments.campaign_stage7_holdouts import ligament_class

EV = 16.0217663
DARK, RED, BLUE, GREY3 = MF.DARK, MF.RED, MF.BLUE, "0.35"
SHEET, VEIN = "#d4d4d4", "#b3b3b3"
FW, FH = 7.2, 7.12                    # figure size in inches
NUMS = {}


def axin(fig, x, ytop, w, h, **kw):
    """Axes from a box given in inches, measured from the left and from the top of the figure."""
    return fig.add_axes([x / FW, 1 - (ytop + h) / FH, w / FW, h / FH], **kw)


def blank(ax, w, h):
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")


def load_arrows(ax, x0, x1, y, length=0.7, gap=0.12, sigma=True):
    for xa, xb in ((x0 - gap, x0 - gap - length), (x1 + gap, x1 + gap + length)):
        ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>", mutation_scale=7, color=DARK, lw=1.0, shrinkA=0, shrinkB=0, zorder=5))
    if sigma: ax.text(x1 + gap + length / 2, y + 0.16, "$\\sigma$", fontsize=7.5, ha="center", va="bottom", color=DARK)


# --------------------------------------------------------------------------- atomistic snapshots
def snap(ax, name, key):
    """Atoms (dark), bonds (grey) and atoms that have lost a bond (red) of one event-selected frame."""
    rec = MF.BY[name]; T = load_traj(os.path.join(MF.ROOT, rec["trajectory"]))
    k = None
    for lab, f in select_event_frames(T):
        if key in lab.split(" = "): k = f
    if k is None: k = len(T["eps_x"]) - 1 if key == "final" else int(np.argmax(T["sigma_xx"]))
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    for a in range(2): P[:, a] = np.mod(P[:, a], c[a, a])
    if len(b):
        p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]; d = p1 - p0
        for a in range(2): d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
        ax.add_collection(LineCollection(np.stack([p0, p0 + d], 1), colors="0.45", linewidths=0.22, zorder=1))
    ax.scatter(P[:, 0], P[:, 1], s=0.35, c="#2b2b2b", linewidths=0, zorder=2)
    dmg = damaged_atoms_until(rec, T["eps_x"][k])
    if len(dmg): ax.scatter(P[dmg, 0], P[dmg, 1], s=2.4, c=RED, linewidths=0, zorder=3)
    ax.set_xlim(-1.5, c[0, 0] + 1.5); ax.set_ylim(-1.5, c[1, 1] + 1.5); ax.set_aspect("equal"); ax.set_anchor("N"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    return float(T["eps_x"][k]), float(T["sigma_xx"][k] * EV)


def under(ax, s, color=DARK):
    ax.text(0.5, -0.04, s, transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=color)


# --------------------------------------------------------------------------- schematics
def slit_sheet(ax, x0, y0, w, params, nx=2.5, ny=4.5):
    """Window of nx by ny periods of the simulated slit lattice, drawn to scale (slits rotate about their
    centres on a fixed staggered lattice, as in the generator).  Returns the scale, the height and the slits."""
    px, py, L, sw, th = params["period_x"], params["period_y"], params["slit_len"], params["slit_w"], np.radians(params["angle_deg"])
    s = w / (nx * px); h = ny * py * s
    clip = Rectangle((x0, y0), w, h, fc=SHEET, ec="none", zorder=1); ax.add_patch(clip)
    u = np.array([np.cos(th), np.sin(th)]); slits = []
    for iy in range(-2, int(ny) + 3):
        for ix in range(-2, int(nx) + 3):
            c = np.array([x0 + (ix + 0.5 * (iy % 2) + 0.25) * px * s, y0 + (iy + 0.5) * py * s])
            r = Rectangle((c[0] - L * s / 2, c[1] - sw * s / 2), L * s, sw * s, angle=params["angle_deg"], rotation_point=(c[0], c[1]), fc="white", ec="none", zorder=2)
            r.set_clip_path(clip); r._background = True; ax.add_patch(r)
            slits.append(dict(ix=ix, iy=iy, c=c, a=c - u * L * s / 2, b=c + u * L * s / 2))
    return s, h, slits


def inside(p, x0, y0, w, h, m=0.0):
    return x0 + m <= p[0] <= x0 + w - m and y0 + m <= p[1] <= y0 + h - m


def mesh_sheet(ax, x0, y0, w, n=4, across=False, veins=False):
    """Single-level mesh with elliptical pores (along or across the load) or a two-level mesh with solid
    veins along the load and fine pores elongated along the load."""
    ax.add_patch(Rectangle((x0, y0), w, w, fc=SHEET, ec="none", zorder=1))
    if not veins:
        cell = w / n
        for i in range(n):
            for j in range(n):
                e = Ellipse((x0 + cell * (i + 0.5), y0 + cell * (j + 0.5)), cell * (0.34 if across else 0.74), cell * (0.74 if across else 0.34), fc="white", ec="none", zorder=2); ax.add_patch(e)
    else:
        rows = 7; t = w / rows
        for j in range(rows):
            if j in (1, 3, 5):
                v = Rectangle((x0, y0 + j * t), w, t, fc=VEIN, ec="none", zorder=1.5); v._background = True; ax.add_patch(v)
            else:
                m = 6; cw = w / m
                for i in range(m): ax.add_patch(Ellipse((x0 + cw * (i + 0.5), y0 + (j + 0.5) * t), cw * 0.72, t * 0.5, fc="white", ec="none", zorder=2))


# --------------------------------------------------------------------------- data
def ligament_data():
    """Net-section (ligament) strength sigma_max/(A_min/A) of the porous designs by ligament class of the
    round-1 predictor.  Tilted slit arrays are the subject of row b and are left out of this panel."""
    lab = {"straight": "straight,\nalong load", "coarse_round": "wide\n(≥ 18 Å)", "fine_round": "narrow\n(< 18 Å)", "random": "random\nnetwork"}
    out = {k: [] for k in lab}
    for r in MF.ALL:
        if r.get("campaign") not in ("discovery", "paper") or (r.get("porosity") or 0) <= 0.03: continue
        c = ligament_class(r["family"], r["params"], r["descriptors"])
        if c not in lab: continue
        if r["family"] == "slit_array" and abs(r["params"].get("angle_deg", 0)) > 1e-6: continue
        out[c].append((r["name"], MF.sig(r) / MF.msf(r)))
    return lab, out


def slit_data():
    P = json.load(open(os.path.join(MF.ROOT, "experiments", "predictions", "paper_sweeps_predictions.json")))
    preds = {p["name"]: p for p in P["predictions"]}; snet, _ = MF.sigma_net_straight()
    rows = []
    for r in MF.ALL:
        if r["family"] == "slit_array" and r["params"].get("orientation", "zigzag") == "zigzag" and r["params"].get("period_x") == 30 and 0.17 <= (r.get("porosity") or 0) <= 0.23:
            p = preds.get(r["name"])
            rows.append(dict(name=r["name"], theta=float(r["params"]["angle_deg"]), phi=r["porosity"], sigma=MF.sig(r), rule=p["predicted"]["strength_Nm_rule"] if p else MF.msf(r) * snet))
    best = {}
    for q in sorted(rows, key=lambda q: abs(q["phi"] - 0.2)): best.setdefault(q["theta"], q)
    return [best[t] for t in sorted(best)]


def mesh_data():
    def ordered(r): return not any(k in r["name"] for k in ("jitter", "sizedis")) and "disorder" not in (r.get("tags") or [])
    meshes = [r for r in MF.ALL if r["family"] in ("nanomesh_single", "nanomesh_hier") and 0.17 <= (r.get("porosity") or 0) <= 0.23 and ordered(r)]
    lev = lambda r: min(r.get("hierarchy_levels", 2), 3) if r["family"] == "nanomesh_hier" else 1
    return [(r["name"], lev(r), MF.Aidx(r), MF.sig(r)) for r in meshes]


# --------------------------------------------------------------------------- the figure
def fig_mechanisms():
    fig = plt.figure(figsize=(FW, FH))
    XS, WS = 0.08, 2.35            # schematic column
    XN, WN = 2.62, 1.62            # snapshot column
    XM, WM = 4.98, 2.12            # measurement column (axes box)
    TH = 0.27                      # height of a row title
    ya, ha = 0.05, 1.50            # row a: top of the title, height of the content
    yb, hb = ya + TH + ha + 0.36, 2.62
    yc, hc = yb + TH + hb + 0.22, 1.50

    def title(y, letter, s, tag, tagcol):
        fig.text(0.012, 1 - (y + 0.02) / FH, letter, fontsize=10.5, fontweight="bold", ha="left", va="top", color=DARK)
        fig.text(0.046, 1 - (y + 0.04) / FH, s, fontsize=8.6, fontweight="bold", ha="left", va="top", color=DARK)
        fig.text(0.988, 1 - (y + 0.05) / FH, tag, fontsize=7.2, ha="right", va="top", color=tagcol, style="italic")
        fig.add_artist(plt.Line2D([0.012, 0.988], [1 - (y + TH - 0.035) / FH] * 2, color="0.75", lw=0.6))

    # ------------------------------------------------------------------ row a: the rule
    title(ya, "a", "The rule: strength = weakest section × ligament strength", "established physics, sharpened", "0.45")
    y0 = ya + TH
    ax = axin(fig, XS, y0, WS, ha); U = 10.0; blank(ax, U, U * ha / WS)
    sx, sy, sw_, sh = 1.75, 1.15, 6.5, 3.9
    ax.add_patch(Rectangle((sx, sy), sw_, sh, fc=SHEET, ec="none", zorder=1))
    cols = np.linspace(sx, sx + sw_, 6)[:-1] + sw_ / 10; rws = np.linspace(sy, sy + sh, 4)[:-1] + sh / 6
    for cx, dia in zip(cols, (0.42, 0.66, 0.98, 0.66, 0.42)):
        for cy in rws: ax.add_patch(Ellipse((cx, cy), dia, dia, fc="white", ec="none", zorder=2))
    xm = cols[2]
    ax.plot([xm, xm], [sy - 0.3, sy + sh + 0.3], color=RED, lw=0.9, ls=(0, (3, 1.6)), zorder=4)
    edges = [sy] + [v for cy in rws for v in (cy - 0.49, cy + 0.49)] + [sy + sh]
    for a_, b_ in zip(edges[0::2], edges[1::2]): ax.plot([xm, xm], [a_, b_], color=RED, lw=2.6, solid_capstyle="butt", zorder=5)
    load_arrows(ax, sx, sx + sw_, sy + sh / 2)
    ax.text(xm, sy - 0.42, "weakest section $A_\\mathrm{min}$", fontsize=7, color=RED, ha="center", va="top")
    ax.text(xm, sy + sh + 0.42, "$\\sigma_\\mathrm{max} = (A_\\mathrm{min}/A)\\;\\sigma_\\mathrm{lig}$", fontsize=8.2, color=DARK, ha="center", va="bottom")
    axs = axin(fig, XN, y0 + 0.02, WN, ha - 0.2); e, s = snap(axs, "S2_graded_x", "post_peak")
    under(axs, "graded pores: crack in the weakest column")
    axm = axin(fig, XM, y0 + 0.06, WM, ha - 0.34); MF.style(axm)
    lab, data = ligament_data(); rng = np.random.default_rng(3); NUMS["lig"] = {}
    for i, (k, l) in enumerate(lab.items()):
        v = np.array([q[1] for q in data[k]]); xj = i + rng.uniform(-0.2, 0.2, len(v))
        axm.scatter(xj, v, s=9, color="0.42", edgecolors="white", linewidths=0.3, zorder=3)
        axm.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=DARK, lw=1.4, zorder=4, solid_capstyle="butt")
        axm.text(i + 0.36, np.median(v), f"{np.median(v):.0f}", fontsize=6.6, va="center", ha="left", color=DARK)
        NUMS["lig"][k] = dict(n=int(len(v)), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    axm.axhline(MF.S0, color="0.35", lw=0.7, ls=(0, (4, 2))); axm.text(3.62, MF.S0 + 0.8, "pristine sheet", fontsize=6.2, ha="right", va="bottom", color="0.3")
    axm.set_xticks(range(len(lab))); axm.set_xticklabels(list(lab.values()), fontsize=6.3); axm.tick_params(axis="x", length=0, pad=2)
    axm.set_xlim(-0.55, 3.75); axm.set_ylim(0, 46); axm.set_yticks([0, 10, 20, 30, 40]); axm.grid(axis="x", visible=False)
    axm.set_ylabel("ligament strength $\\sigma_\\mathrm{lig}$ (N/m)", fontsize=7.2)
    NUMS["lig_n"] = int(sum(len(v) for v in data.values()))
    wl = [(r["descriptors"]["ligament_width_mean_A"], MF.sig(r) / MF.msf(r)) for r in MF.ALL if r.get("campaign") in ("discovery", "paper") and ligament_class(r["family"], r["params"], r["descriptors"]) in ("coarse_round", "fine_round") and (r.get("porosity") or 0) > 0.03]
    NUMS["width_r"] = float(np.corrcoef(*zip(*wl))[0, 1]); NUMS["width_n"] = len(wl)

    # ------------------------------------------------------------------ row b: tilted slits
    title(yb, "b", "Where the rule breaks: a tilted slit array fails by three mechanisms", "new in this work", RED)
    y0 = yb + TH
    regimes = [("I", "tip linking", "S7_slit_20deg", "final", RED), ("II", "ligament rotation", "S2_slit_45deg", "peak_load", BLUE), ("III", "bridge bending", "S2_slit_90deg", "post_peak", GREY3)]
    gapx = 0.14; wmini = (XN + WN - XS - 2 * gapx) / 3; hs = 1.12
    for i, (rom, nm, run, key, col) in enumerate(regimes):
        x = XS + i * (wmini + gapx); par = MF.BY[run]["params"]
        ax = axin(fig, x, y0 + 0.2, wmini, hs); U = 10.0; blank(ax, U, U * hs / wmini)
        w = 7.0; x0 = (U - w) / 2; s, h, slits = slit_sheet(ax, x0, 0.25, w, par); yb0 = 0.25
        ax.text(U / 2, yb0 + h + 0.3, f"{rom}  {nm}", fontsize=7.4, fontweight="bold", color=col, ha="center", va="bottom")
        if i == 0: load_arrows(ax, x0, x0 + w, yb0 + h / 2, length=0.95, gap=0.15, sigma=False)
        by = {(q["ix"], q["iy"]): q for q in slits}
        if rom == "I":      # overlapping tips of adjacent rows: the sliver between them fails and the cracks link
            for q in slits:
                nb = by.get((q["ix"] + (0 if q["iy"] % 2 == 0 else 1), q["iy"] + 1))
                if nb is not None and inside(q["b"], x0, yb0, w, h, 0.05) and inside(nb["a"], x0, yb0, w, h, 0.05):
                    ax.plot([q["b"][0], nb["a"][0]], [q["b"][1], nb["a"][1]], color=RED, lw=1.5, solid_capstyle="round", zorder=4)
        if rom == "II":     # ribbons between the slit channels rotate towards the load axis
            c0 = np.array([x0 + w * 0.36, yb0 + h * 0.30]); ang = np.radians(par["angle_deg"]); R = 2.7
            ax.plot([c0[0], c0[0] + R * 1.25], [c0[1], c0[1]], color=BLUE, lw=0.9, ls=(0, (2.5, 1.5)), zorder=5)
            ax.plot([c0[0], c0[0] + R * 1.25 * np.cos(ang)], [c0[1], c0[1] + R * 1.25 * np.sin(ang)], color=BLUE, lw=0.9, ls=(0, (2.5, 1.5)), zorder=5)
            tt = np.linspace(ang * 0.93, ang * 0.22, 24); arc = np.stack([c0[0] + R * np.cos(tt), c0[1] + R * np.sin(tt)], 1)
            ax.plot(arc[:, 0], arc[:, 1], color=BLUE, lw=1.5, zorder=6)
            ax.add_patch(FancyArrowPatch(tuple(arc[-2]), tuple(arc[-1] + (arc[-1] - arc[-2]) * 2.0), arrowstyle="-|>", mutation_scale=8, color=BLUE, lw=0, shrinkA=0, shrinkB=0, zorder=6))
        if rom == "III":    # the load path zigzags through the short necks between collinear slits; the strips between them bend
            necks = []
            for q in slits:
                nb = by.get((q["ix"], q["iy"] + 2))
                if nb is None: continue
                mid = (q["b"] + nb["a"]) / 2
                if inside(mid, x0, yb0, w, h, 0.02): necks.append(mid)
            necks = np.array(necks); ymid = yb0 + h / 2
            cols_x = sorted(set(np.round(necks[:, 0], 3)))
            path = [np.array([x0, ymid])]
            for cx in cols_x:
                cand = necks[np.abs(necks[:, 0] - cx) < 1e-3]; path.append(cand[np.argmin(np.abs(cand[:, 1] - path[-1][1]))])
            path.append(np.array([x0 + w, path[-1][1]])); path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color=RED, lw=1.5, solid_joinstyle="round", zorder=5)
            ax.scatter(necks[:, 0], necks[:, 1], s=9, color=RED, linewidths=0, zorder=6)
        axs = axin(fig, x, y0 + 0.2 + hs + 0.1, wmini, hb - hs - 0.5); e, sg = snap(axs, run, key)
        under(axs, f"θ = {par['angle_deg']:.0f}°, {MF.sig(MF.BY[run]):.1f} N/m", color=col)
    axm = axin(fig, XM, y0 + 0.2, WM, hb - 0.55); MF.style(axm)
    rows = slit_data(); NUMS["slit"] = rows
    for (lo, hi, col, rom) in ((-4, 22.5, RED, "I"), (22.5, 52.5, BLUE, "II"), (52.5, 94, GREY3, "III")):
        MF._bg(axm.axvspan(lo, hi, color=col, alpha=0.08, lw=0))
        axm.text((max(lo, -4) + hi) / 2, 33.6, rom, fontsize=8, fontweight="bold", color=col, ha="center", va="top")
    th = sorted(set(q["theta"] for q in rows)); rule = [np.mean([q["rule"] for q in rows if q["theta"] == t]) for t in th]
    axm.plot(th, rule, "-", color="0.55", lw=1.0, zorder=2)
    for q in rows:
        col = RED if q["theta"] < 22.5 else (BLUE if q["theta"] < 52.5 else GREY3); low = q["phi"] < 0.19 and q["theta"] in (35.0, 45.0)
        axm.scatter(q["theta"], q["sigma"], s=24, facecolors="white" if low else col, edgecolors=col if low else "k", linewidths=0.9 if low else 0.4, zorder=4)
    axm.set_xlim(-4, 94); axm.set_ylim(0, 35); axm.set_xticks([0, 15, 30, 45, 60, 75, 90])
    axm.set_xlabel("slit angle to the load θ (°)", fontsize=7.2); axm.set_ylabel("2D strength $\\sigma_\\mathrm{max}$ (N/m)", fontsize=7.2)
    axm.text(61, 27.2, "net-section\nrule (row a)", fontsize=6.3, color="0.4", ha="left", va="bottom", linespacing=1.1)
    sget = lambda t: next(q["sigma"] for q in rows if q["theta"] == t)
    axm.text(20, sget(20) - 1.6, f"{sget(20):.1f}", fontsize=6.4, ha="center", va="top", color=RED)
    axm.text(2.5, 28.0, f"{sget(0):.1f}", fontsize=6.4, ha="left", va="bottom", color=RED)
    axm.text(45, sget(45) + 1.5, f"{sget(45):.1f}", fontsize=6.4, ha="center", va="bottom", color=BLUE)
    axm.text(86.8, sget(90) - 0.9, f"{sget(90):.1f}", fontsize=6.4, ha="right", va="center", color=GREY3)

    # ------------------------------------------------------------------ row c: alignment, not hierarchy
    title(yc, "c", "Hierarchy adds no premium: strength follows load-path alignment", "new in this work", RED)
    y0 = yc + TH
    ax = axin(fig, XS, y0, WS, hc); U = 10.0; blank(ax, U, U * hc / WS)
    wm = 2.3; ys = 2.0; xs = [1.0, 3.85, 6.7]
    labs = [("pores\nacross load", "one level\n$A < 0$", GREY3), ("pores\nalong load", "one level\n$A > 0$", BLUE), ("veins + pores\nalong load", "two levels\n$A > 0$", RED)]
    for x, (l1, l2, col), kw in zip(xs, labs, (dict(across=True), dict(), dict(veins=True))):
        mesh_sheet(ax, x, ys, wm, **kw)
        ax.text(x + wm / 2, ys + wm + 0.2, l1, fontsize=6.4, ha="center", va="bottom", color=DARK, linespacing=1.1)
        ax.text(x + wm / 2, ys - 0.2, l2, fontsize=6.4, ha="center", va="top", color=col, linespacing=1.15)
    load_arrows(ax, xs[0], xs[2] + wm, ys + wm / 2, length=0.55, gap=0.1, sigma=False)
    axs = axin(fig, XN, y0 + 0.02, WN, hc - 0.2); e, s = snap(axs, "S7_H2_veinsX_W12_ellipseX", "post_peak")
    under(axs, "two levels, aligned: fine-pore rows fail first")
    axm = axin(fig, XM, y0 + 0.06, WM, hc - 0.4); MF.style(axm)
    md = mesh_data(); A = np.array([q[2] for q in md]); S = np.array([q[3] for q in md]); Lv = np.array([q[1] for q in md])
    f1 = np.polyfit(A[Lv == 1], S[Lv == 1], 1); f2 = np.polyfit(A[Lv == 2], S[Lv == 2], 1); xx = np.linspace(-0.3, 0.68, 10)
    axm.plot(xx, np.polyval(f1, xx), "-", color=BLUE, lw=1.0, zorder=2); axm.plot(xx, np.polyval(f2, xx), "-", color=RED, lw=1.0, zorder=2)
    for lv, mk, col, lb in ((1, "o", BLUE, "one level"), (2, "s", RED, "two levels"), (3, "^", MF.PURPLE, "three levels")):
        axm.scatter(A[Lv == lv], S[Lv == lv], marker=mk, s=15, color=col, edgecolors="white", linewidths=0.35, zorder=3, label=lb)
    prem = S[Lv >= 2] - np.polyval(f1, A[Lv >= 2])
    axm.legend(loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
    axm.text(0.98, 0.04, "premium of the nested designs\n" + f"{prem.mean():+.1f}".replace("-", "−") + f" ± {prem.std():.1f} N/m", transform=axm.transAxes, fontsize=6.3, ha="right", va="bottom", color=RED, linespacing=1.15)
    axm.set_xlim(-0.32, 0.72); axm.set_ylim(4, 24); axm.set_xlabel("load-path alignment index $A$", fontsize=7.2); axm.set_ylabel("2D strength $\\sigma_\\mathrm{max}$ (N/m)", fontsize=7.2)
    NUMS["align"] = dict(n=int(len(md)), n1=int((Lv == 1).sum()), n2=int((Lv == 2).sum()), n3=int((Lv == 3).sum()), slope1=float(f1[0]), slope2=float(f2[0]), premium_mean=float(prem.mean()), premium_std=float(prem.std()), n_nested=int(len(prem)))
    return fig


CAPTION = r"""\caption{\textbf{What sets the strength: the rule and the two ways in which it fails.} Each row shows a schematic (left), an atomistic snapshot of the failure (middle; atoms that have lost a bond in red) and one measurement (right). (a) The rule: strength is the smallest solid cross-section perpendicular to the load, $A_\mathrm{min}/A$, times a ligament strength $\sigma_\mathrm{lig}$. In a sheet with graded pores the crack runs through the most porous column. Right: $\sigma_\mathrm{lig}=\sigma_\mathrm{max}/(A_\mathrm{min}/A)$ of the \mechNlig{} porous designs other than tilted slit arrays, grouped by the ligament classes of the pre-registered predictor (straight along the load; wide, at least 18\,\AA; narrow; random network); bars are medians (\mechSigStraight, \mechSigWide, \mechSigNarrow{} and \mechSigRandom\,\Nm), the dashed line is the pristine sheet. (b) Where the rule breaks: slit arrays at equal porosity tilted against the load, drawn to scale from the simulated parameters. I, tip linking: the tips of adjacent rows overlap, the slivers between them fail and the cracks link en echelon (red). II, ligament rotation: the ribbons between the slit channels rotate towards the load; at peak load the slits have closed. III, bridge bending: the slits cut every straight load path and the load zigzags through short bridges (red). Right: strength against slit angle for the \mechNslit{} simulated angles; the net-section rule of row a (grey line) is blind to all three mechanisms; open symbols have porosity 0.18 instead of 0.20. (c) Hierarchy adds no premium: elongating the pores across or along the load changes the load-path alignment index $A$ (Eq.~\ref{eq:A}), and a second level of solid veins along the load is another way of raising $A$. Right: strength against $A$ for the \alignN{} ordered meshes with one, two and three levels and the linear trends for one and two levels; at equal $A$ the nested designs lie $\premiumMean\pm\premiumStd$\,\Nm{} from the one-level trend. Model results (screened REBO2, athermal quasi-static tension).}"""

TEXT = r"""Figure~\ref{fig:mech} collects the mechanisms in one view. The baseline is a rule (Fig.~\ref{fig:mech}a): the strength of a porous sheet is its weakest cross-section times the strength of its ligaments, and the ligament strength depends on the kind of ligament, about \mechSigStraight\,\Nm{} for straight or wide ligaments against \mechSigNarrow\,\Nm{} for ligaments narrower than 18\,\AA, in which edge atoms make up a large share of the section (over the \mechWidthN{} meshes the ligament strength correlates with the mean ligament width, $r=\mechWidthR$). The two new results of this work are the two ways in which the rule fails. It fails for slit arrays tilted against the load (Fig.~\ref{fig:mech}b), because it counts material and not connectivity: up to $\mechSlitMinAngle^\circ$ the overlapping slit tips link en echelon and the strength falls from \mechSlitZero{} to \mechSlitMin\,\Nm; between 25 and $45^\circ$ the ribbons between the slits rotate towards the load and the strength recovers to \mechSlitFortyFive\,\Nm; beyond $60^\circ$ every straight load path is cut, the load zigzags through short bridges and the strength falls to \mechSlitNinety\,\Nm. And it is incomplete for meshes (Fig.~\ref{fig:mech}c), in which the strength at fixed porosity follows the alignment of the solid phase with the load, and a second hierarchical level earns no premium beyond the alignment it provides ($\premiumMean\pm\premiumStd$\,\Nm{} relative to the one-level trend)."""


def write_tex():
    """Macros for caption and text, and the LaTeX snippet in a macro version and with the numbers filled in."""
    sl = {q["theta"]: q["sigma"] for q in NUMS["slit"]}; tmin = min((t for t in sl if t <= 45), key=sl.get)   # minimum of the tip-linking regime
    mac = dict(mechNlig=f"{NUMS['lig_n']:d}", mechSigStraight=f"{NUMS['lig']['straight']['median']:.0f}", mechSigWide=f"{NUMS['lig']['coarse_round']['median']:.0f}",
               mechSigNarrow=f"{NUMS['lig']['fine_round']['median']:.0f}", mechSigRandom=f"{NUMS['lig']['random']['median']:.0f}", mechNslit=f"{len(sl):d}",
               mechSlitZero=f"{sl[0.0]:.1f}", mechSlitMin=f"{sl[tmin]:.1f}", mechSlitMinAngle=f"{tmin:.0f}", mechSlitFortyFive=f"{sl[45.0]:.1f}", mechSlitNinety=f"{sl[90.0]:.1f}", mechWidthR=f"{NUMS['width_r']:.2f}", mechWidthN=f"{NUMS['width_n']:d}")
    open(os.path.join(MF.OUT, "mech_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    fig_env = "\\begin{figure}[!tbp]\\centering\\includegraphics[width=\\textwidth]{fig_mechanisms.pdf}\n" + CAPTION + "\n\\label{fig:mech}\\end{figure}\n"
    head = ("% Mechanism overview figure: generated by paper/mechanism_figure.py, do not edit numbers by hand.\n"
            "% 1) upload figures/fig_mechanisms.pdf    2) paste the paragraph and the figure environment into main.tex\n")
    open(os.path.join(MF.OUT, "fig_mechanisms_snippet.tex"), "w").write(head + "% 3) this macro version also needs figures/mech_numbers.tex and  \\input{figures/mech_numbers.tex}  in the preamble\n\n% --- paragraph\n" + TEXT + "\n\n% --- figure\n" + fig_env)
    # resolved version: fill in every number macro from the generated macro files (unit macros such as \Nm stay)
    defs = dict(mac)
    for f in ("numbers.tex", "facts.tex"):
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", open(os.path.join(MF.OUT, f)).read(), re.M): defs.setdefault(m.group(1), m.group(2))
    def resolve(t):
        for k in sorted(defs, key=len, reverse=True):
            t = re.sub(r"\\" + k + r"(?![A-Za-z])(\{\})?", lambda _m, v=defs[k]: v, t)
        return t
    open(os.path.join(MF.OUT, "fig_mechanisms_snippet_resolved.tex"), "w").write(head + "\n% --- paragraph\n" + resolve(TEXT) + "\n\n% --- figure\n" + "\\begin{figure}[!tbp]\\centering\\includegraphics[width=\\textwidth]{fig_mechanisms.pdf}\n" + resolve(CAPTION) + "\n\\label{fig:mech}\\end{figure}\n")
    return mac


if __name__ == "__main__":
    fig = fig_mechanisms()
    fig._resolved = True                       # deliberate placement: audit only, no automatic moves
    conflicts = CO.audit(fig, "fig_mechanisms")
    for ext in ("svg", "png", "pdf"): fig.savefig(os.path.join(MF.OUT, f"fig_mechanisms.{ext}"), facecolor="white")
    json.dump(NUMS, open(os.path.join(MF.OUT, "mech_numbers.json"), "w"), indent=1)
    mac = write_tex()
    print("conflicts:", len(conflicts)); print(mac)
