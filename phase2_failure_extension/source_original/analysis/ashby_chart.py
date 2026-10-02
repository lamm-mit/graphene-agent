"""Ashby-style property charts of all simulated architectures (family envelopes, guidelines of constant
specific property, thumbnails of representative relaxed structures) and an atlas of every simulated
architecture.  Reads only the experiment database; writes figures/campaign/ashby_overview.* and
figures/campaign/architecture_atlas.* (pdf/png/svg + caption).

Label placement is checked automatically: family labels, guideline labels and thumbnail captions are
moved (from a list of candidate positions) until they overlap neither the leader lines of the thumbnails,
nor the guidelines, nor other labels, nor data points."""
from __future__ import annotations
import os, sys, textwrap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from ase.io import read
from analysis.campaign_analysis import load, M, FAMCOL, save

FAMNAME = {"pristine": "pristine", "vacancies": "vacancies", "precrack": "precracked", "ring_around_hole": "hole + rings",
           "slit_array": "slit array", "strut_lattice": "strut lattice", "graded_pores": "graded pores",
           "nanomesh_single": "nanomesh (single level)", "nanomesh_hier": "nanomesh (hierarchical)", "voronoi_network": "Voronoi network"}
FAMORDER = ["pristine", "vacancies", "precrack", "ring_around_hole", "slit_array", "strut_lattice", "graded_pores", "nanomesh_single", "nanomesh_hier", "voronoi_network"]
THUMBS = ["S1_pristine_zz", "S1_precrack_L20", "S1_vac_random_2pct", "S5_slit0_phi0.1", "S2_slit_90deg", "S7_slit_20deg",
          "S7_H2_veinsX_W12_ellipseX", "S2_hole20_rings2", "S2_voronoi_n25_reg0.6", "S2_strut_090", "S2_H1_p16", "S2_graded_x"]
RHO_GRAPHENE = 4.0 / (np.sqrt(3.0) * 2.460177 ** 2)  # atoms per A^2 of the relaxed REBO2 lattice
STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]


def rho_rel(r):
    d = r.get("descriptors") or {}
    if d.get("Lx") and d.get("Ly"):
        return r["n_atoms"] / (d["Lx"] * d["Ly"]) / RHO_GRAPHENE
    return 1.0 - (r.get("porosity") or 0.0)


def thumbnail(rec, px=240):
    """Render the relaxed structure (atoms as black dots, one periodic cell, square frame) into an RGB array."""
    atoms = read(os.path.join(ROOT, rec["run_dir"], "relaxed.extxyz"))
    pos = atoms.get_positions(); Lx, Ly = atoms.cell.lengths()[:2]
    x = pos[:, 0] % Lx; y = pos[:, 1] % Ly; L = max(Lx, Ly)
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim((Lx - L) / 2, (Lx + L) / 2); ax.set_ylim((Ly - L) / 2, (Ly + L) / 2); ax.set_aspect("equal")
    s = (px / L * 1.7 * 0.72) ** 2
    ax.scatter(x, y, s=s, c="k", linewidths=0, marker="o")
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return img


def short(name):
    return name.split("_", 1)[1] if "_" in name else name


# ----------------------------------------------------------------------------- collision-free label placement
def _seg_hits_bbox(p, q, bb, pad=2.0):
    """True if the segment p-q (display coords) intersects the bbox enlarged by pad (Liang-Barsky clipping)."""
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0: return False
        else:
            t = qk / pk
            if pk < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
    return t0 <= t1


def _conflicts(bb, segments, obstacles):
    """Number of line segments (p, q, pad) crossing and obstacle bboxes overlapping bb."""
    return sum(_seg_hits_bbox(p, q, bb, pad) for p, q, pad in segments) + sum(bb.overlaps(o) for o in obstacles)


def _ring_candidates(step=12, rings=7):
    out = [(0, 0)]
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for k in range(1, rings + 1):
        out += [(dx * step * k, dy * step * k) for dx, dy in dirs]
    return out


def _padded(bb, pad):
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def place_text(text, renderer, candidates, segments, obstacles, inside=None, pad=1.5):
    """Move `text` to the candidate offset (display pixels) with the fewest conflicts: bbox inside `inside`,
    crossing no segment (p, q, pad) and overlapping no obstacle bbox.  Earlier candidates win ties.
    Returns (final bbox, number of remaining conflicts)."""
    tr = text.get_transform(); inv = tr.inverted(); base = tr.transform(text.get_position())
    best = None
    for dx, dy in candidates:
        text.set_position(inv.transform((base[0] + dx, base[1] + dy)))
        bb = _padded(text.get_window_extent(renderer), pad)
        if inside is not None and not (bb.x0 >= inside.x0 and bb.x1 <= inside.x1 and bb.y0 >= inside.y0 and bb.y1 <= inside.y1):
            continue
        c = _conflicts(bb, segments, obstacles)
        if best is None or c < best[0]:
            best = (c, (dx, dy), bb)
        if c == 0: break
    if best is None:
        text.set_position(inv.transform(base)); return _padded(text.get_window_extent(renderer), pad), 99
    text.set_position(inv.transform((base[0] + best[1][0], base[1] + best[1][1])))
    return best[2], best[0]


def envelope(ax, xs, ys, color, label, lw=13, label_pos=None):
    P = np.c_[np.log10(xs), np.log10(ys)]
    poly = P
    if len(P) >= 3 and np.ptp(P[:, 0]) > 1e-6 and np.ptp(P[:, 1]) > 1e-6:
        try:
            poly = P[ConvexHull(P).vertices]
        except Exception:
            poly = P
    X, Y = 10 ** poly[:, 0], 10 ** poly[:, 1]
    if len(poly) >= 3:
        ax.fill(X, Y, color=color, alpha=0.10, lw=0, zorder=1)
        ax.plot(np.r_[X, X[0]], np.r_[Y, Y[0]], color=color, alpha=0.16, lw=lw, solid_joinstyle="round", solid_capstyle="round", zorder=1)
    else:
        ax.plot(X, Y, color=color, alpha=0.16, lw=lw, solid_capstyle="round", zorder=1)
    if not label: return None, (X, Y)
    if label_pos is not None:
        cx, cy, ha = label_pos
    else:
        cx, cy, ha = 10 ** P[:, 0].mean(), 10 ** (P[:, 1].max() + 0.02), "center"
    return ax.text(cx, cy, label, color=color, fontsize=7.5, ha=ha, va="bottom", zorder=4, fontweight="bold", path_effects=STROKE), (X, Y)


def ashby_panel(ax, recs, xf, yf, xlabel, ylabel, letter="", label_families=True):
    """Scatter + family envelopes.  Returns (family label texts as (family, text), data points, envelope polygons as (family, X, Y))."""
    fams = [f for f in FAMORDER if any(r["family"] == f for r in recs)]
    labels, points, polys = [], [], []
    for f in fams:
        rs = [r for r in recs if r["family"] == f]
        X = np.array([xf(r) for r in rs]); Y = np.array([yf(r) for r in rs]); ok = np.isfinite(X) & np.isfinite(Y) & (X > 0) & (Y > 0)
        if ok.sum() == 0: continue
        if isinstance(label_families, dict):
            t, poly = envelope(ax, X[ok], Y[ok], FAMCOL.get(f, "0.3"), FAMNAME.get(f, f) if f in label_families else "", label_pos=label_families.get(f))
        else:
            t, poly = envelope(ax, X[ok], Y[ok], FAMCOL.get(f, "0.3"), FAMNAME.get(f, f) if label_families else "")
        if t is not None: labels.append((f, t))
        polys.append((f, poly[0], poly[1]))
        ax.scatter(X[ok], Y[ok], s=14, color=FAMCOL.get(f, "0.3"), edgecolors="white", linewidths=0.4, zorder=3, label=FAMNAME.get(f, f))
        points += list(zip(X[ok], Y[ok]))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    if letter:
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    return labels, points, polys


def guide_lines(ax, ks, x0, x1, fmt, fontsize=6):
    """Dashed guidelines y = k x (slope 1 in log-log) with labels at their left end inside the axes.
    Returns (label texts, segments in data coords)."""
    y0, y1 = ax.get_ylim(); texts, segs = [], []
    for k in ks:
        xx = np.array([x0, x1]); ax.plot(xx, k * xx, "--", color="0.55", lw=0.7, zorder=2)
        segs.append(((x0, k * x0), (x1, k * x1)))
        xl = max(x0 * 1.02, 1.03 * y0 / k)
        if k * xl > y1: continue
        p0 = ax.transData.transform((xl, k * xl)); p1 = ax.transData.transform((xl * 1.3, k * xl * 1.3))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        texts.append(ax.text(xl, k * xl * 1.04, fmt(k), fontsize=fontsize, color="0.3", rotation=ang, rotation_mode="anchor", ha="left", va="bottom", zorder=4, path_effects=STROKE))
    return texts, segs


def rho_axis(ax):
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))


def add_thumbnails(ax, recs, images, xf, yf):
    """Thumbnails around the axes with straight leader lines to their data points; slots are assigned by
    angular order to minimise crossings.  Returns (leader segments in display coords, caption texts with slots)."""
    slots = np.array([[0.02, 1.15], [0.26, 1.15], [0.50, 1.15], [0.74, 1.15], [0.98, 1.15],
                      [-0.25, 0.16], [-0.25, 0.50], [-0.25, 0.84],
                      [1.25, 0.10], [1.25, 0.37], [1.25, 0.63], [1.25, 0.90]])
    sel = [r for r in recs if r["name"] in images][: len(slots)]
    pts = np.array([ax.transAxes.inverted().transform(ax.transData.transform((xf(r), yf(r)))) for r in sel])
    ang_p = np.arctan2(pts[:, 1] - 0.5, pts[:, 0] - 0.5); ang_s = np.arctan2(slots[:, 1] - 0.5, slots[:, 0] - 0.5)
    D = np.abs((ang_p[:, None] - ang_s[None, :] + np.pi) % (2 * np.pi) - np.pi)
    ri, ci = linear_sum_assignment(D)
    segments, captions = [], []
    for i, j in zip(ri, ci):
        r = sel[i]; col = FAMCOL.get(r["family"], "0.3")
        ab = AnnotationBbox(OffsetImage(images[r["name"]], zoom=0.33), (xf(r), yf(r)), xybox=tuple(slots[j]), xycoords="data",
                            boxcoords="axes fraction", frameon=True, pad=0.12, bboxprops=dict(edgecolor=col, lw=1.4),
                            arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=0, alpha=0.65), zorder=6, annotation_clip=False)
        ax.add_artist(ab)
        segments.append((ax.transData.transform((xf(r), yf(r))), ax.transAxes.transform(tuple(slots[j]))))
        t = ax.text(slots[j][0], slots[j][1] - 0.085, short(r["name"]), transform=ax.transAxes, ha="center", va="top", fontsize=6, color=col, clip_on=False)
        captions.append((t, slots[j]))
    return segments, captions


def layout_panel_a(fig, ax, fam_labels, polys, guide_texts, guide_segs_data, leader_segs, captions, points):
    """Collision-free placement of all text in panel (a): thumbnail captions avoid leader lines; guideline labels
    avoid leader lines, the other guidelines, every envelope outline, data points and other labels (either side of
    their own line); family labels avoid leader lines, guidelines, the outlines of the other families, data points
    and other labels.  Returns the list of (label, remaining conflicts) that could not be fully resolved."""
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    leaders = [(p, q, 2.0) for p, q in leader_segs]
    guides = [(ax.transData.transform(p), ax.transData.transform(q), 2.0) for p, q in guide_segs_data]
    outlines = {}
    for f, X, Y in polys:
        P = [ax.transData.transform((x, y)) for x, y in zip(X, Y)]
        outlines[f] = [(P[i], P[(i + 1) % len(P)], 7.0) for i in range(len(P))] if len(P) > 1 else []
    all_outlines = [sg for v in outlines.values() for sg in v]
    pt_boxes = []
    for x, y in points:
        px, py = ax.transData.transform((x, y)); pt_boxes.append(Bbox.from_extents(px - 4, py - 4, px + 4, py + 4))
    cap_boxes = []
    for t, slot in captions:
        bb = t.get_window_extent(renderer)
        if any(_seg_hits_bbox(p, q, bb) for p, q in leader_segs):
            t.set_position((slot[0], slot[1] + 0.085)); t.set_va("bottom"); bb = t.get_window_extent(renderer)
        cap_boxes.append(bb)
    unresolved = []
    gbb = []
    for i, t in enumerate(guide_texts):
        ang = np.radians(t.get_rotation()); d = np.array([np.cos(ang), np.sin(ang)]); nrm = np.array([-np.sin(ang), np.cos(ang)])
        h = t.get_window_extent(renderer).height
        cands = [tuple(d * sft + nrm * side) for side in (0.0, -(h + 8.0)) for sft in (0, 15, 30, 50, 70, 100, 130, 160, 200, 250, 300, 350, 400, 450)]
        own = guides[i]
        segs = leaders + [g for g in guides if g is not own] + all_outlines
        bb, c = place_text(t, renderer, cands, segs, pt_boxes + gbb + cap_boxes, inside=axbb)
        gbb.append(bb)
        if c: unresolved.append((t.get_text(), c))
    placed = []
    for f, t in fam_labels:
        others = [sg for g, v in outlines.items() if g != f for sg in v]
        segs = leaders + guides + others
        bb, c = place_text(t, renderer, _ring_candidates(), segs, gbb + placed + pt_boxes, inside=axbb)
        if c:
            bb, c = place_text(t, renderer, _ring_candidates(step=14, rings=9), segs, gbb + placed, inside=axbb)
        placed.append(bb)
        if c: unresolved.append((t.get_text(), c))
    return unresolved


def main():
    recs = [r for r in load() if r.get("metrics") and np.isfinite(M(r, "strength_Nm"))]
    by = {r["name"]: r for r in recs}
    images = {n: thumbnail(by[n]) for n in THUMBS if n in by}
    missing = [n for n in THUMBS if n not in by]
    if missing: print("thumbnail designs not in database:", missing)
    s = lambda r: M(r, "strength_Nm"); Y = lambda r: M(r, "modulus_2d_Nm"); W = lambda r: M(r, "work_to_failure_J_m2"); ef = lambda r: M(r, "failure_strain")

    fig = plt.figure(figsize=(11, 13.5))
    axa = fig.add_axes([0.22, 0.44, 0.56, 0.42])
    axa.set_xscale("log"); axa.set_yscale("log"); axa.set_xlim(0.48, 1.08); axa.set_ylim(3, 48)
    LAB = {"pristine": (0.955, 36.5, "right"), "vacancies": (0.93, 25.5, "right"), "precrack": (1.03, 14.6, "left"), "slit_array": (0.70, 27.5, "center"),
           "strut_lattice": (0.55, 13.2, "center"), "nanomesh_single": (0.60, 8.6, "left"), "voronoi_network": (0.83, 5.6, "left"),
           "nanomesh_hier": (0.735, 20.5, "right"), "ring_around_hole": (0.93, 21.5, "left"), "graded_pores": (0.86, 9.6, "left")}
    fam_labels, points, polys = ashby_panel(axa, recs, rho_rel, s, "areal density relative to pristine graphene, $\\bar\\rho$", "2D strength $\\sigma_\\mathrm{max}$ (N/m)", label_families=LAB)
    axa.set_xlim(0.48, 1.08); axa.set_ylim(3, 48); rho_axis(axa)
    guide_texts, guide_segs = guide_lines(axa, (38.8, 20, 10, 5), 0.48, 1.08, lambda k: f"$\\sigma/\\bar\\rho$ = {k:g} N/m" + (" (pristine zigzag)" if k == 38.8 else ""))
    axa.text(0.02, 0.98, "a", transform=axa.transAxes, fontsize=12, fontweight="bold", va="top")
    leader_segs, captions = add_thumbnails(axa, recs, images, rho_rel, s)
    # one legend for all panels, between the rows (outside every axes, so it cannot collide with lines)
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMCOL.get(f, "0.3"), markersize=5, label=FAMNAME.get(f, f)) for f in FAMORDER if any(r["family"] == f for r in recs)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.375), ncol=5, fontsize=7, frameon=False, columnspacing=1.6, handletextpad=0.4)

    axb = fig.add_axes([0.08, 0.05, 0.24, 0.28]); axc = fig.add_axes([0.42, 0.05, 0.24, 0.28]); axd = fig.add_axes([0.75, 0.05, 0.24, 0.28])
    axb.set_xscale("log"); axb.set_yscale("log"); axb.set_xlim(0.48, 1.08); axb.set_ylim(1.5, 400)
    ashby_panel(axb, recs, rho_rel, Y, "relative areal density $\\bar\\rho$", "2D modulus $Y$ (N/m)", letter="b", label_families=False)
    axb.set_xlim(0.48, 1.08); axb.set_ylim(1.5, 400); rho_axis(axb)
    gtb, _ = guide_lines(axb, (245, 100, 30), 0.48, 1.08, lambda k: f"$Y/\\bar\\rho$ = {k:g} N/m", fontsize=5.5)
    ashby_panel(axc, recs, W, s, "work to failure $W$ (J/m$^2$, proxy)", "2D strength $\\sigma_\\mathrm{max}$ (N/m)", letter="c", label_families=False)
    ashby_panel(axd, recs, ef, s, "failure strain $\\varepsilon_f$", "2D strength $\\sigma_\\mathrm{max}$ (N/m)", letter="d", label_families=False)
    for ax in (axc, axd):
        ax.set_ylim(3, 48)
    axd.set_xlim(0.08, 0.36); axd.xaxis.set_major_locator(FixedLocator([0.1, 0.15, 0.2, 0.3])); axd.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axd.xaxis.set_minor_formatter(NullFormatter())
    axc.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5])); axc.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axc.xaxis.set_minor_formatter(NullFormatter())
    for ax in (axa, axb, axc, axd):
        ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50, 100, 200, 300])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())

    bad = layout_panel_a(fig, axa, fam_labels, polys, guide_texts, guide_segs, leader_segs, captions, points)
    # panel b guideline labels: keep them clear of the data points
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    pb = []
    for r in recs:
        px, py = axb.transData.transform((rho_rel(r), Y(r))); pb.append(Bbox.from_extents(px - 4, py - 4, px + 4, py + 4))
    for t in gtb:
        ang = np.radians(t.get_rotation()); d = np.array([np.cos(ang), np.sin(ang)])
        place_text(t, renderer, [tuple(d * s_) for s_ in (0, 10, 20, 35, 50, 70, 90)], [], pb, inside=axb.get_window_extent(renderer))
    if bad: print("NOTE: labels with remaining conflicts (label, count):", bad)
    else: print("label layout: no text crosses a leader line, guideline or foreign envelope outline")
    n = len(recs)
    save(fig, "ashby_overview", f"Ashby-style property charts of all {n} simulated architectures of the discovery campaign (model results: screened REBO2, athermal quasi-static uniaxial tension; 2D quantities). (a) 2D strength versus areal density relative to pristine graphene (log-log). Shaded envelopes are the convex hulls of each generator family, dashed guidelines are lines of constant specific strength sigma/rho (the top line is the pristine zigzag sheet), and the thumbnails show the relaxed structures of twelve representative designs (frame colour = family; leader lines point to their data points). (b) 2D modulus versus relative areal density with guidelines of constant specific modulus. (c) 2D strength versus work to failure (area under the stress-strain curve to loss of load-bearing capacity; a proxy, not a fracture toughness). (d) 2D strength versus failure strain. Every point is one simulation; Stage 6 seed replicates are included.")

    # ---- atlas of all simulated architectures
    order = sorted(recs, key=lambda r: (FAMORDER.index(r["family"]) if r["family"] in FAMORDER else 99, -M(r, "strength_Nm")))
    cols = 10; rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 1.32 * rows + 0.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.035, wspace=0.08, hspace=0.62)
    for k, ax in enumerate(axes.flat):
        if k >= len(order):
            ax.axis("off"); continue
        r = order[k]; col = FAMCOL.get(r["family"], "0.3")
        img = images[r["name"]] if r["name"] in images else thumbnail(r, px=200)
        ax.imshow(img, interpolation="antialiased"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.6)
        ax.set_title("\n".join(textwrap.wrap(r["name"], 20)), fontsize=4.9, pad=1.5, color=col, fontweight="bold")
        ax.set_xlabel(f"$\\sigma$={M(r, 'strength_Nm'):.1f}  $W$={M(r, 'work_to_failure_J_m2'):.2f}  $\\varepsilon_f$={M(r, 'failure_strain'):.2f}\n$\\phi$={r.get('porosity') or 0:.2f}  N={r['n_atoms']}  {str(M(r, 'fracture_mode')).split(' ')[0]}", fontsize=4.6, labelpad=1.2)
    handles = [Patch(facecolor="white", edgecolor=FAMCOL.get(f, "0.3"), linewidth=1.8, label=FAMNAME.get(f, f)) for f in FAMORDER if any(r["family"] == f for r in recs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, frameon=False, handlelength=1.2, columnspacing=1.2)
    save(fig, "architecture_atlas", f"Atlas of all {n} simulated architectures of the discovery campaign: relaxed structure of one periodic cell (atoms as black dots), grouped by generator family (frame colour, legend at the bottom) and ordered by decreasing 2D strength within each family. Below each structure: 2D strength sigma (N/m), work to failure W (J/m$^2$, proxy), failure strain, porosity phi, number of atoms N and fracture-mode class (abrupt = single avalanche, stepwise = few events, progressive = multiple events with load retained). Model results (screened REBO2, athermal quasi-static tension).")
    print("ashby overview and atlas written:", n, "structures")


if __name__ == "__main__":
    main()
