"""Overlap audit for the paper figures: every text artist (annotations, labels, legends, tick labels, titles)
is checked against every data artist (scatter markers, line segments, bars/patches, images) and against the
other texts of the same figure.  Run:  python check_overlaps.py [1 2 ...]   -> prints one line per conflict."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.transforms import Bbox
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.legend import Legend
import make_figures as MF


def seg_hits_bbox(p, q, bb, pad=0.0):
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    dx, dy = q[0] - p[0], q[1] - p[1]; t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0: return False
        else:
            t = qk / pk
            if pk < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
    return t0 <= t1


def shrink(bb, f=0.12):
    """Shrink a text bbox slightly so that touching (not overlapping) elements are not reported."""
    w, h = bb.width, bb.height
    return Bbox.from_extents(bb.x0 + f * w, bb.y0 + f * h, bb.x1 - f * w, bb.y1 - f * h)


def audit(fig, name, verbose=True):
    fig.canvas.draw(); R = fig.canvas.get_renderer()
    texts = []
    from matplotlib.text import Annotation, Text as _T
    def add_text(t, kind, ax=None):
        if t is None or not t.get_visible() or getattr(t, "_overlay", False): return
        s = t.get_text() if hasattr(t, "get_text") else ""
        if hasattr(t, "get_text") and not str(s).strip(): return
        bb = _T.get_window_extent(t, R) if isinstance(t, Annotation) else t.get_window_extent(R)
        if bb.width <= 0 or bb.height <= 0: return
        axi = fig.axes.index(ax) if ax in fig.axes else -1
        texts.append(dict(artist=t, bbox=bb, kind=kind, ax=ax, label=f"{kind}@ax{axi}:" + (str(s)[:32] if hasattr(t, "get_text") else kind) + f" [{bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}]"))
    for t in fig.texts: add_text(t, "figtext")
    for lg in fig.legends: texts.append(dict(artist=lg, bbox=lg.get_window_extent(R), kind="figlegend", ax=None, label="fig legend"))
    for ax in fig.axes:
        for t in ax.texts: add_text(t, "text", ax)
        add_text(ax.title, "title", ax); add_text(ax.xaxis.label, "xlabel", ax); add_text(ax.yaxis.label, "ylabel", ax)
        if ax.axison:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            for tk in ax.xaxis.get_major_ticks():
                if min(xl) <= tk.get_loc() <= max(xl): add_text(tk.label1, "tick", ax)
            for tk in ax.yaxis.get_major_ticks():
                if min(yl) <= tk.get_loc() <= max(yl): add_text(tk.label1, "tick", ax)
        lgs = ([ax.get_legend()] if ax.get_legend() else []) + [ar for ar in ax.artists if isinstance(ar, Legend) and ar is not ax.get_legend()]
        for lg in lgs:
            bb = lg.get_window_extent(R); texts.append(dict(artist=lg, bbox=bb, kind="legend", ax=ax, label=f"legend@ax{fig.axes.index(ax)} [{bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}]"))
    # data artists
    data = []
    for ax in fig.axes:
        axbb = ax.get_window_extent(R)
        for c in ax.collections:
            if isinstance(c, LineCollection):
                for seg in c.get_segments():
                    P = ax.transData.transform(seg)
                    for i in range(len(P) - 1): data.append(("lineseg", (P[i], P[i + 1]), ax))
            elif isinstance(c, PathCollection):
                off = c.get_offsets(); tr = c.get_offset_transform()
                if len(off) == 0: continue
                P = tr.transform(off); sizes = c.get_sizes();
                for k, p in enumerate(P):
                    s = sizes[k % len(sizes)] if len(sizes) else 10; r = 0.5 * np.sqrt(s) * fig.dpi / 72.0
                    if axbb.contains(p[0], p[1]) or True:
                        data.append(("marker", Bbox.from_extents(p[0] - r, p[1] - r, p[0] + r, p[1] + r), ax))
        for ln in ax.lines:
            if not ln.get_visible(): continue
            xy = ln.get_xydata()
            if len(xy) < 1: continue
            P = ax.transData.transform(xy)
            P = P[np.isfinite(P).all(1)]
            if ln.get_linestyle() != "None" and ln.get_linewidth() > 0:
                for i in range(len(P) - 1):
                    seg = (P[i], P[i + 1])
                    if ln.get_clip_on():
                        x0, y0, x1, y1 = axbb.x0, axbb.y0, axbb.x1, axbb.y1; pp, qq = seg; dx, dy = qq[0] - pp[0], qq[1] - pp[1]; t0, t1 = 0.0, 1.0; ok = True
                        for pk, qk in ((-dx, pp[0] - x0), (dx, x1 - pp[0]), (-dy, pp[1] - y0), (dy, y1 - pp[1])):
                            if pk == 0:
                                if qk < 0: ok = False
                            else:
                                tt = qk / pk
                                if pk < 0: t0 = max(t0, tt)
                                else: t1 = min(t1, tt)
                        if not ok or t0 > t1: continue
                        seg = (np.array([pp[0] + t0 * dx, pp[1] + t0 * dy]), np.array([pp[0] + t1 * dx, pp[1] + t1 * dy]))
                    data.append(("line", seg, ax))
            if ln.get_marker() not in (None, "None", ""):
                r = 0.5 * ln.get_markersize() * fig.dpi / 72.0
                for p in P: data.append(("marker", Bbox.from_extents(p[0] - r, p[1] - r, p[0] + r, p[1] + r), ax))
        for pt in ax.patches:
            if isinstance(pt, (Rectangle, Polygon, FancyBboxPatch)) and pt.get_visible() and not getattr(pt, "_background", False):
                bb = pt.get_window_extent(R)
                if bb.width > 0 and bb.height > 0 and pt.get_alpha() not in (0,) and not (pt.get_facecolor()[3] == 0 and pt.get_edgecolor()[3] == 0):
                    # skip axis-span shading (very light, intended as background)
                    fc = pt.get_facecolor()
                    if pt.get_alpha() is not None and pt.get_alpha() <= 0.15: continue
                    if fc[3] <= 0.15 and pt.get_edgecolor()[3] <= 0.15: continue
                    data.append(("patch", bb, ax))
        for im in ax.images:
            data.append(("image", im.get_window_extent(R), ax))
    conflicts = []
    for t in texts:
        ax = t["ax"]
        if ax is None or t["kind"] in ("figtext", "figlegend"): continue
        kids = set(getattr(ax, "child_axes", []))
        for other in fig.axes:
            if other is ax or other in kids or ax in getattr(other, "child_axes", []): continue
            if shrink(t["bbox"]).overlaps(other.get_window_extent(R)):
                conflicts.append((t["label"], f"intrudes into ax{fig.axes.index(other)}")); break
    for i, t in enumerate(texts):
        bb = shrink(t["bbox"])
        for kind, geo, ax in data:
            if kind in ("line", "lineseg"):
                if seg_hits_bbox(geo[0], geo[1], bb): conflicts.append((t["label"], f"{kind}@ax{fig.axes.index(ax)} ({geo[0][0]:.0f},{geo[0][1]:.0f})-({geo[1][0]:.0f},{geo[1][1]:.0f})")); break
            else:
                if bb.overlaps(geo): conflicts.append((t["label"], f"{kind}@ax{fig.axes.index(ax)} [{geo.x0:.0f},{geo.y0:.0f},{geo.x1:.0f},{geo.y1:.0f}]")); break
        for j, u in enumerate(texts):
            if j <= i: continue
            if bb.overlaps(shrink(u["bbox"])) and not (t["kind"] == "tick" and u["kind"] == "tick"):
                conflicts.append((t["label"], f"text:{u['label']}"))
    # deduplicate
    seen = set(); out = []
    for c in conflicts:
        if c not in seen: seen.add(c); out.append(c)
    if verbose:
        print(f"[{name}] {len(out)} conflict(s)")
        for c in out: print("   ", c)
    return out


if __name__ == "__main__":
    only = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    orig_save = MF.save
    results = {}
    def save_and_audit(fig, name):
        MF.resolve(fig)
        results[name] = audit(fig, name)
        orig_save(fig, name)
    MF.save = save_and_audit
    for k in only:
        getattr(MF, f"fig{k}")()
    print("TOTAL conflicts:", sum(len(v) for v in results.values()))
    if set(only) >= {"1", "2", "3", "4", "5", "6"}: MF.export_numbers(); print("numbers.tex / sweeps_table.tex written")
