"""Paper figures for the six claims (rule of mixtures, load paths, alignment cliff, hierarchy/alignment,
costs of disorder, pre-registered predictions).  Every number is read from the experiment database
(campaigns "discovery" and "paper") and exported to figures/numbers.tex for the manuscript.
Outputs: paper/figures/<name>.{svg,png,pdf} + numbers.json + numbers.tex."""
from __future__ import annotations
import os, sys, json, math, glob, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse, Circle, Patch
from scipy import stats
from ase.io import read
from experiments import db
from analysis.campaign_analysis import FAMCOL, M, curve
from analysis.ashby_chart import FAMNAME, FAMORDER, rho_rel, thumbnail
from atomistics.structures import design_space as D
from experiments.campaign_paper_sweeps import alignment_index, slit_geometry, sigma_net_straight

OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
DARK, RED, BLUE, ORANGE, GREEN, GREY, PURPLE = "#222222", "#c0392b", "#1f77b4", "#ff7f0e", "#2ca02c", "0.55", "#9467bd"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none"})
NUM = {}



# --------------------------------------------------------------------------- automatic collision-free text placement
from matplotlib.legend import Legend
from matplotlib.text import Annotation
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.transforms import Bbox


def _bg(p):
    """Mark a patch (axvspan/axhspan shading) as background: texts may sit on it."""
    p._background = True; return p


def _fx(t):
    """Mark a text as fixed: the resolver never moves it."""
    t._fixed = True; return t


def _seg_hits(p, q, bb, pad=1.0):
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


def _shrink(bb, f=0.0, pad=3.0):
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _obstacles(fig, R):
    obs = []
    for ax in fig.axes:
        for c in ax.collections:
            if isinstance(c, LineCollection):
                for seg in c.get_segments():
                    P = ax.transData.transform(seg); obs += [("seg", (P[i], P[i + 1])) for i in range(len(P) - 1)]
            elif isinstance(c, PathCollection):
                off = c.get_offsets()
                if len(off) == 0: continue
                P = c.get_offset_transform().transform(off); sizes = c.get_sizes()
                for k, pt in enumerate(P):
                    sz = sizes[k % len(sizes)] if len(sizes) else 10; r = 0.5 * np.sqrt(sz) * fig.dpi / 72.0
                    obs.append(("box", Bbox.from_extents(pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r)))
        axbb = ax.get_window_extent(R)
        def _clip(p, q):
            x0, y0, x1, y1 = axbb.x0, axbb.y0, axbb.x1, axbb.y1; dx, dy = q[0] - p[0], q[1] - p[1]; t0, t1 = 0.0, 1.0
            for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
                if pk == 0:
                    if qk < 0: return None
                else:
                    t = qk / pk
                    if pk < 0: t0 = max(t0, t)
                    else: t1 = min(t1, t)
            if t0 > t1: return None
            return (np.array([p[0] + t0 * dx, p[1] + t0 * dy]), np.array([p[0] + t1 * dx, p[1] + t1 * dy]))
        for ln in ax.lines:
            if not ln.get_visible(): continue
            P = ax.transData.transform(ln.get_xydata()); P = P[np.isfinite(P).all(1)]
            if ln.get_linestyle() != "None" and ln.get_linewidth() > 0:
                for i in range(len(P) - 1):
                    cs = _clip(P[i], P[i + 1]) if ln.get_clip_on() else (P[i], P[i + 1])
                    if cs is not None: obs.append(("seg", cs))
            if ln.get_marker() not in (None, "None", ""):
                r = 0.5 * ln.get_markersize() * fig.dpi / 72.0
                obs += [("box", Bbox.from_extents(pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r)) for pt in P]
        for pt in ax.patches:
            if getattr(pt, "_background", False) or not pt.get_visible(): continue
            bb = pt.get_window_extent(R)
            if bb.width > 0 and bb.height > 0: obs.append(("box", bb))
        for im in ax.images:
            obs.append(("box", im.get_window_extent(R)))
    return obs


def _foreign_axes(fig, R, ax):
    """Bboxes of the axes a text of `ax` must not intrude into (all axes except ax, its parent and its children)."""
    kids = set(getattr(ax, "child_axes", [])); out = []
    for other in fig.axes:
        if other is ax or other in kids or ax in getattr(other, "child_axes", []): continue
        out.append(other.get_window_extent(R))
    return out


def _all_texts(fig, R):
    items = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        items += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            items += [tk.label1 for tk in ax.xaxis.get_major_ticks() if min(xl) <= tk.get_loc() <= max(xl)]
            items += [tk.label1 for tk in ax.yaxis.get_major_ticks() if min(yl) <= tk.get_loc() <= max(yl)]
        if ax.get_legend(): items.append(ax.get_legend())
        items += [a for a in ax.artists if isinstance(a, Legend) and a is not ax.get_legend()]
    out = []
    for t in items:
        if not t.get_visible() or getattr(t, "_overlay", False): continue
        if hasattr(t, "get_text") and not str(t.get_text()).strip(): continue
        bb = _tbbox(t, R)
        if bb.width > 0 and bb.height > 0: out.append((t, bb))
    return out


def _tbbox(t, R):
    """Text-only bbox (for annotations the default extent includes the arrow)."""
    from matplotlib.text import Text as _T
    return _T.get_window_extent(t, R) if isinstance(t, Annotation) else t.get_window_extent(R)


def _nconf(bb, obs, others):
    bb = _shrink(bb); n = 0
    for kind, g in obs:
        n += _seg_hits(g[0], g[1], bb) if kind == "seg" else bb.overlaps(g)
    for ob in others: n += bb.overlaps(_shrink(ob))
    return n


def _ring(step=8, rings=7):
    out = [(0, 0)]
    for k in range(1, rings + 1):
        out += [(dx * step * k, dy * step * k) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    return out


CORNERS = [(0.03, 0.97, "left", "top"), (0.97, 0.97, "right", "top"), (0.03, 0.03, "left", "bottom"), (0.97, 0.03, "right", "bottom"),
           (0.5, 0.97, "center", "top"), (0.97, 0.5, "right", "center"), (0.03, 0.5, "left", "center"), (0.5, 0.03, "center", "bottom")]


def resolve(fig, step=6, rings=5):
    if getattr(fig, "_resolved", False): return fig
    fig._resolved = True
    """Move annotations, data-coordinate labels, axes-coordinate notes and legends to conflict-free positions.
    Policy: Annotation and data-coordinate Text -> ring of pixel offsets around the original position (inside the axes);
    axes-coordinate Text (unless fixed) -> corner/edge candidates; Legend without an explicit anchor -> loc candidates."""
    fig.canvas.draw(); R = fig.canvas.get_renderer()
    obs = _obstacles(fig, R); texts = _all_texts(fig, R)
    bboxes = {id(t): bb for t, bb in texts}; byid = {id(t): t for t, _ in texts}
    def others_of(t): return [bb for k, bb in bboxes.items() if k != id(t)]
    order = sorted(texts, key=lambda tb: (0 if isinstance(tb[0], Legend) else 1, -tb[1].width * tb[1].height))
    for t, _ in order:
        if getattr(t, "_fixed", False): continue
        if isinstance(t, Legend):
            if t._bbox_to_anchor is not None or t.axes is None: continue
            best = None
            for loc in [t._loc] + [1, 2, 3, 4, 7, 6, 9, 8]:
                t._set_loc(loc); bb = t.get_window_extent(R); c = _nconf(bb, obs, others_of(t))
                if best is None or c < best[0]: best = (c, loc)
                if c == 0: break
            t._set_loc(best[1]); bboxes[id(t)] = t.get_window_extent(R); continue
        ax = t.axes
        if ax is None: continue
        if isinstance(t, Annotation):
            tr = t.get_transform(); inv = tr.inverted(); base = tr.transform(t.xyann); axbb = ax.get_window_extent(R); best = None
            for dx, dy in _ring(step, rings):
                t.xyann = tuple(inv.transform((base[0] + dx, base[1] + dy))); bb = _tbbox(t, R)
                inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
                c = _nconf(bb, obs, others_of(t)) + (0 if inside else 3)
                if best is None or c < best[0]: best = (c, t.xyann)
                if c == 0: break
            t.xyann = best[1]; bboxes[id(t)] = _tbbox(t, R); continue
        tr = t.get_transform()
        if tr == ax.transAxes:
            x0, y0 = t.get_position(); ha0, va0 = t.get_ha(), t.get_va(); best = None
            for (x, y, ha, va) in [(x0, y0, ha0, va0)] + CORNERS:
                t.set_position((x, y)); t.set_ha(ha); t.set_va(va); bb = t.get_window_extent(R); c = _nconf(bb, obs, others_of(t))
                if best is None or c < best[0]: best = (c, (x, y, ha, va))
                if c == 0: break
            x, y, ha, va = best[1]; t.set_position((x, y)); t.set_ha(ha); t.set_va(va); bboxes[id(t)] = t.get_window_extent(R); continue
        if tr == ax.transData:
            inv = tr.inverted(); base = tr.transform(t.get_position()); axbb = ax.get_window_extent(R); best = None
            for dx, dy in _ring(step, rings):
                t.set_position(tuple(inv.transform((base[0] + dx, base[1] + dy)))); bb = t.get_window_extent(R)
                inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
                c = _nconf(bb, obs, others_of(t)) + (0 if inside else 3)
                if best is None or c < best[0]: best = (c, t.get_position())
                if c == 0: break
            t.set_position(best[1]); bboxes[id(t)] = t.get_window_extent(R)
    return fig


def save(fig, name):
    resolve(fig)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.2, lw=0.4); ax.set_axisbelow(True)


def letter(ax, s, dx=-0.02, dy=1.05):
    _fx(ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right"))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), float(stats.spearmanr(x[ok], y[ok]).correlation)


# --------------------------------------------------------------------------- data
ALL = [r for r in db.all_records() if r.get("status") == "completed" and r.get("metrics") and np.isfinite(M(r, "strength_Nm"))]
DISC = [r for r in ALL if r.get("campaign") == "discovery"]
PAPER = [r for r in ALL if r.get("campaign") == "paper"]
BY = {r["name"]: r for r in ALL}
S0 = M(BY["S1_pristine_zz"], "strength_Nm"); S0AC = M(BY["S1_pristine_ac"], "strength_Nm"); Y0 = M(BY["S1_pristine_zz"], "modulus_2d_Nm")
sig = lambda r: M(r, "strength_Nm"); Ymod = lambda r: M(r, "modulus_2d_Nm"); W = lambda r: M(r, "work_to_failure_J_m2")
msf = lambda r: r["descriptors"]["min_solid_fraction_across_x"]
Aidx = lambda r: alignment_index(r["descriptors"])
NUM.update(n_discovery=len(DISC), n_paper=len(PAPER), sigma0_zz=S0, sigma0_ac=S0AC, Y0=Y0)


def famcol(r): return FAMCOL.get(r["family"], "0.3")


def family_legend(ax, fams, loc="lower right", ncol=1, **kw):
    h = [Line2D([], [], marker="o", ls="", color=FAMCOL.get(f, "0.3"), markersize=4, label=FAMNAME.get(f, f)) for f in FAMORDER if f in fams]
    return ax.legend(handles=h, loc=loc, ncol=ncol, frameon=True, framealpha=0.9, edgecolor="none", handletextpad=0.3, **kw)


# --------------------------------------------------------------------------- figure 1: rule of mixtures
def fig1():
    recs = DISC; rho = np.array([rho_rel(r) for r in recs]); s = np.array([sig(r) for r in recs]); y = np.array([Ymod(r) for r in recs])
    band = (rho >= 0.77) & (rho <= 0.83)
    ss, ys = s[band] / rho[band], y[band] / rho[band]
    yr = y / rho; exc = band & (yr < 110); lig = band & ~exc     # exceptions: inclined slit arrays whose ligaments rotate or bend
    NUM.update(band_n=int(band.sum()), band_spec_sigma_min=float(ss.min()), band_spec_sigma_max=float(ss.max()), band_spec_sigma_ratio=float(ss.max() / ss.min()),
               band_spec_Y_min_lig=float(yr[lig].min()), band_spec_Y_max_lig=float(yr[lig].max()), band_spec_Y_ratio_lig=float(yr[lig].max() / yr[lig].min()),
               band_Y_exceptions=[(recs[i]["name"], round(float(y[i]), 1), round(float(yr[i]), 1)) for i in np.where(exc)[0]], band_n_exceptions=int(exc.sum()))
    por = np.array([(r.get("porosity") or 0) > 0.03 for r in recs])
    NUM.update(r_sigma_rho=pearson(rho[por], s[por])[0], r_Y_rho=pearson(rho[por], y[por])[0], rs_sigma_rho=pearson(rho[por], s[por])[1], rs_Y_rho=pearson(rho[por], y[por])[1])
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.75)); plt.subplots_adjust(wspace=0.42, left=0.07, right=0.99, top=0.92, bottom=0.3)
    for ax, val, y0, y0b, lab in [(axs[0], s, S0, S0AC, "2D strength $\\sigma_\\mathrm{max}$ (N/m)"), (axs[1], y, Y0, None, "2D modulus $Y$ (N/m)")]:
        style(ax); xx = np.linspace(0.45, 1.02, 50)
        ax.plot(xx, y0 * xx, "-", color="0.25", lw=0.9)
        if y0b: ax.plot(xx, y0b * xx, "--", color="0.45", lw=0.8)
        _bg(ax.axvspan(0.77, 0.83, color="0.85", alpha=0.5, lw=0))
        for r, rr, v in zip(recs, rho, val):
            ax.scatter(rr, v, s=13, color=famcol(r), edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_xlabel("relative areal density $\\bar\\rho$"); ax.set_ylabel(lab); ax.set_xlim(0.45, 1.04)
    axs[0].set_ylim(0, 42); axs[1].set_ylim(0, 300)
    def along(ax, k, x0, text, dy=1.06):
        p0 = ax.transData.transform((x0, k * x0)); p1 = ax.transData.transform((x0 * 1.2, k * x0 * 1.2))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        _fx(ax.text(x0, k * x0 * dy, text, fontsize=5.8, color="0.3", rotation=ang, rotation_mode="anchor", ha="left", va="bottom"))
    pass
    axs[0].annotate(f"at $\\bar\\rho$ = 0.8:\n$\\sigma/\\bar\\rho$ from {ss.min():.1f} to {ss.max():.1f} N/m\n(factor {ss.max() / ss.min():.1f})", xy=(0.8, 34.0), xytext=(0.47, 41.5), va="top", fontsize=6.8, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    axs[1].annotate(f"at $\\bar\\rho$ = 0.8:\n$Y/\\bar\\rho$ within {NUM['band_spec_Y_min_lig']:.0f}–{NUM['band_spec_Y_max_lig']:.0f} N/m\nfor all but {NUM['band_n_exceptions']} designs", xy=(0.8, 232), xytext=(0.47, 296), va="top", fontsize=6.8, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    for nm, yy, _ in NUM["band_Y_exceptions"]:
        r = BY[nm]; ang = r["params"].get("angle_deg", 0)
        axs[1].annotate(f"slits at {ang:.0f}°" + (" (ligaments rotate)" if ang == 45 else ""), xy=(rho_rel(r), yy), xytext=(0.5, 20) if ang == 45 else (0.61, 58), fontsize=6.2, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    ax = axs[2]; style(ax)
    ex = y / (rho * Y0); es = s / (rho * S0)
    for r, a, b in zip(recs, ex, es):
        ax.scatter(a, b, s=13, color=famcol(r), edgecolors="white", linewidths=0.3, zorder=3)
    ax.plot([0, 1.2], [0, 1.2], ":", color="0.5", lw=0.7); ax.scatter([1], [1], s=40, marker="*", color="k", zorder=4)
    ax.text(0.97, 1.03, "pristine", ha="right", va="bottom", fontsize=6.5)
    ax.set_xlabel("modulus efficiency $Y/(\\bar\\rho\\,Y_0)$"); ax.set_ylabel("strength efficiency $\\sigma_\\mathrm{max}/(\\bar\\rho\\,\\sigma_0)$")
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.15)
    NUM.update(eff_Y_p10=float(np.percentile(ex[por], 10)), eff_Y_p90=float(np.percentile(ex[por], 90)), eff_s_p10=float(np.percentile(es[por], 10)), eff_s_p90=float(np.percentile(es[por], 90)))
    ax.text(0.97, 0.04, f"porous designs, 10–90% range\n$Y$: {NUM['eff_Y_p10']:.2f}–{NUM['eff_Y_p90']:.2f}   $\\sigma$: {NUM['eff_s_p10']:.2f}–{NUM['eff_s_p90']:.2f}", transform=ax.transAxes, fontsize=6.3, va="bottom", ha="right")
    h = [Line2D([], [], marker="o", ls="", color=FAMCOL.get(f, "0.3"), markersize=4, label=FAMNAME.get(f, f)) for f in FAMORDER if f in set(r["family"] for r in recs)]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    for a, l in zip(axs, "abc"): letter(a, l)
    save(fig, "fig1_rule_of_mixtures")


# --------------------------------------------------------------------------- figure 2: load paths, not mass
def fig2():
    recs = [r for r in DISC if (r.get("porosity") or 0) > 0.03]
    rho = np.array([rho_rel(r) for r in recs]); s = np.array([sig(r) for r in recs]); m = np.array([msf(r) for r in recs]); A = np.array([Aidx(r) for r in recs])
    r_rho, rs_rho = pearson(rho, s); r_msf, rs_msf = pearson(m, s)
    # combined load-path predictor: minSF x sigma_net(class) from the Stage 7 predictor classes
    from experiments.campaign_stage7_holdouts import ligament_class
    snet = {}
    for r in recs:
        c = ligament_class(r["family"], r["params"], r["descriptors"]); snet.setdefault(c, []).append(sig(r) / msf(r))
    snet_med = {c: float(np.median(v)) for c, v in snet.items()}
    pred = np.array([msf(r) * snet_med[ligament_class(r["family"], r["params"], r["descriptors"])] for r in recs])
    r_pred, rs_pred = pearson(pred, s)
    NUM.update(lp_r_rho=r_rho, lp_rs_rho=rs_rho, lp_r_msf=r_msf, lp_rs_msf=rs_msf, lp_r_pred=r_pred, lp_rs_pred=rs_pred, lp_n=len(recs), snet_classes={k: round(v, 1) for k, v in snet_med.items()})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.25])); plt.subplots_adjust(wspace=0.62, left=0.065, right=0.99, top=0.9, bottom=0.2)
    for ax, x, xl, rr, rs in [(axs[0], rho, "relative areal density $\\bar\\rho$ (mass)", r_rho, rs_rho), (axs[1], m, "minimum load-bearing section $A_\\mathrm{min}/A$", r_msf, rs_msf)]:
        style(ax)
        for r, xx, yy in zip(recs, x, s): ax.scatter(xx, yy, s=13, color=famcol(r), edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_xlabel(xl); ax.set_ylabel("2D strength $\\sigma_\\mathrm{max}$ (N/m)"); ax.set_ylim(0, 30)
        ax.text(0.03, 0.97, f"Pearson r = {rr:.2f}\nSpearman ρ = {rs:.2f}", transform=ax.transAxes, va="top", fontsize=6.8)
    # guide lines: median net-section strength of the ligament classes of the predictor (straight and wide ligaments coincide)
    sw = 0.5 * (snet_med["straight"] + snet_med["coarse_round"])
    for sn, lab, ls in [(sw, f"{sw:.0f} N/m, straight or wide (≥ 18 Å) ligaments", "--"), (snet_med["random"], f"{snet_med['random']:.0f} N/m, random networks", "-."), (snet_med["fine_round"], f"{snet_med['fine_round']:.0f} N/m, narrow (< 18 Å) ligaments", ":")]:
        xx = np.linspace(0.1, 1.0, 10); axs[1].plot(xx, sn * xx, ls, color="0.45", lw=0.7, label=lab)
    axs[1].set_xlim(0.1, 1.05); axs[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=False, fontsize=5.2, title="$\\sigma = (A_\\mathrm{min}/A)\\,\\sigma_\\mathrm{lig}$, class medians:", title_fontsize=5.4, handlelength=1.4, labelspacing=0.15, handletextpad=0.3, borderaxespad=0.1)
    ax = axs[2]; style(ax)
    best = {}
    for r in DISC:
        if r["family"] == "pristine": continue
        e = sig(r) / (rho_rel(r) * S0)
        if r["family"] not in best or e > best[r["family"]][0]: best[r["family"]] = (e, r["name"])
    fams = sorted(best, key=lambda f: best[f][0])
    for i, f in enumerate(fams):
        e, n = best[f]
        ax.barh(i, e, color=FAMCOL.get(f, "0.3"), height=0.7)
        ax.text(e + 0.015, i, f"{e:.2f}", va="center", fontsize=6.2)
        tt = _fx(ax.text(0.012, i, n.split("_", 1)[1], va="center", ha="left", fontsize=5.0, color="white")); tt._overlay = True
    ax.axvline(0.8, color=RED, lw=0.8, ls="--"); _fx(ax.text(0.81, -0.75, "80% of pristine", color=RED, fontsize=6.2, va="bottom", ha="left"))
    SHORT = {"nanomesh_single": "single-level mesh", "nanomesh_hier": "hierarchical mesh", "ring_around_hole": "hole + rings", "voronoi_network": "Voronoi network", "precrack": "precracked"}
    ax.set_yticks(range(len(fams))); ax.set_yticklabels([SHORT.get(f, FAMNAME.get(f, f)) for f in fams], fontsize=6.2); ax.set_xlim(0, 1.05); ax.set_ylim(-1.1, len(fams) - 0.4)
    ax.set_xlabel("best specific strength in the family, $\\sigma_\\mathrm{max}/(\\bar\\rho\\,\\sigma_0)$")
    NUM.update(best_eff={f: {"eff": round(best[f][0], 3), "name": best[f][1]} for f in best})
    for a, l in zip(axs, "abc"): letter(a, l)
    save(fig, "fig2_load_paths")


# --------------------------------------------------------------------------- figure 3: alignment cliff (slit angle sweep)
def fig3():
    P = json.load(open(os.path.join(ROOT, "experiments", "predictions", "paper_sweeps_predictions.json")))
    preds = {p["name"]: p for p in P["predictions"]}
    snet, _ = sigma_net_straight()
    slits = [r for r in ALL if r["family"] == "slit_array" and r["params"].get("orientation", "zigzag") == "zigzag" and r["params"].get("period_x") == 30]
    rows = []
    for r in slits:
        th = r["params"]["angle_deg"]; g = slit_geometry(r["params"], r["design"] if "design" in r else dict(Lx=r["descriptors"]["Lx"], Ly=r["descriptors"]["Ly"]))
        rows.append(dict(name=r["name"], theta=th, phi=r.get("porosity"), sigma=sig(r), msf=msf(r), O=g["O_A"], t=g["t_A"], mode=M(r, "fracture_mode"), W=W(r), ef=M(r, "failure_strain"), observed=True, params=r["params"]))
    for n, p in preds.items():
        if p["family"] == "slit_array" and n not in BY:
            rows.append(dict(name=n, theta=p["params"]["angle_deg"], phi=p["porosity"], sigma=np.nan, msf=p["minSF"], O=p["slit_tip_overlap_A"], t=p["slit_row_clearance_A"], observed=False, params=p["params"]))
    for row in rows:
        p = preds.get(row["name"])
        if p: row["rule"], row["mech"] = p["predicted"]["strength_Nm_rule"], p["predicted"]["strength_Nm_mechanism"]
        else:
            row["rule"] = row["msf"] * snet
            row["mech"] = row["rule"] * (max(0.38, 1 - 0.62 * min(1, row["theta"] / 20)) if (row["O"] > 0 and row["theta"] >= 5) else 0.9)
    sweep = sorted([r for r in rows if 0.17 <= (r["phi"] or 0) <= 0.23], key=lambda r: r["theta"])
    ctrl = sorted([r for r in rows if abs(r["theta"] - 20) < 1e-6], key=lambda r: r["phi"])
    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.42, 1], width_ratios=[1.35, 1, 0.9], wspace=0.45, hspace=0.35, left=0.07, right=0.99, top=0.97, bottom=0.15)
    # top strip: the geometry at five angles (generated from the matched parameters, no simulation needed)
    strip = fig.add_subplot(gs[0, :]); strip.axis("off")
    show = []
    for r in sorted(sweep, key=lambda r: abs((r["phi"] or 0) - 0.2)):
        if r["theta"] in (0, 10, 20, 30, 45, 90) and r["theta"] not in [x["theta"] for x in show]: show.append(r)
    show.sort(key=lambda r: r["theta"])
    for k, r in enumerate(show):
        a = D.generate("slit_array", **r["params"]); pos = a.get_positions(); Lx, Ly = a.cell.lengths()[:2]
        ins = strip.inset_axes([k / len(show) + 0.01, 0.0, 1 / len(show) - 0.02, 0.86]); ins.scatter(pos[:, 0] % Lx, pos[:, 1] % Ly, s=0.25, c="k", linewidths=0); ins.set_xlim(0, Lx); ins.set_ylim(0, Ly); ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values(): sp.set_edgecolor(RED if r["O"] > 0 and r["theta"] > 0 else "0.5"); sp.set_linewidth(0.8)
        ins.set_title(f"{r['theta']:.0f}°" + ("" if r["observed"] else " (pending)") + f"\nO = {r['O']:+.0f} Å", fontsize=5.8, pad=1.5, color=RED if r["O"] > 0 and r["theta"] > 0 else DARK, linespacing=1.1)
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    th = [r["theta"] for r in sweep]
    ax.plot(th, [r["rule"] for r in sweep], "-", color="0.55", lw=1.0, label="net-section rule")
    ax.plot(th, [r["mech"] for r in sweep], "--", color=RED, lw=1.0, label="mechanism rule (pre-registered)")
    obs = [r for r in sweep if r["observed"]]
    ax.scatter([r["theta"] for r in obs], [r["sigma"] for r in obs], s=30, color=ORANGE, edgecolors="k", linewidths=0.5, zorder=4, label="simulation")
    pend = [r for r in sweep if not r["observed"]]
    if pend: ax.scatter([r["theta"] for r in pend], [r["mech"] for r in pend], s=22, facecolors="none", edgecolors=RED, linewidths=0.7, zorder=4, label="pending")
    seen = set()
    for r in sorted(obs, key=lambda r: abs((r["phi"] or 0) - 0.2)):
        if r["theta"] in seen or r["theta"] not in (0, 20, 45, 60, 90): continue
        seen.add(r["theta"]); ax.text(r["theta"], r["sigma"] + 1.0, f"{r['sigma']:.1f}", fontsize=5.8, ha="center", va="bottom")
    ov = [r["theta"] for r in sweep if r["O"] > 0]
    if ov: _bg(ax.axvspan(min(ov) - 2.5, max(ov) + 2.5, color=RED, alpha=0.07, lw=0))
    ax.set_xlabel("slit angle to the load θ (°)"); ax.set_ylabel("2D strength $\\sigma_\\mathrm{max}$ (N/m)"); ax.set_xlim(-3, 93); ax.set_ylim(0, 35)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.legend(loc="upper right", frameon=False, fontsize=5.5, ncol=1, handlelength=1.6, borderaxespad=0.2)
    ax.text(0.02, 0.03, "shaded: tips of adjacent rows overlap (O > 0)", transform=ax.transAxes, fontsize=6.0, color=RED)
    ax2 = fig.add_subplot(gs[1, 1]); style(ax2)
    ax2.plot(th, [r["O"] for r in sweep], "o-", color=RED, ms=3.5, lw=0.9, label="tip overlap $O = L\\cos\\theta - p_x/2$")
    ax2.plot(th, [r["t"] for r in sweep], "s-", color=BLUE, ms=3.5, lw=0.9, label="row clearance $t = p_y - L\\sin\\theta$")
    ax2.axhline(0, color="0.3", lw=0.6); ax2.set_xlabel("slit angle θ (°)"); ax2.set_ylabel("length (Å)"); ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax2.legend(loc="upper right", frameon=False, fontsize=5.8); ax2.set_xlim(-3, 93); ax2.set_ylim(-20, 24)
    ax3 = fig.add_subplot(gs[1, 2]); style(ax3)
    xs = np.arange(len(ctrl)); w = 0.26
    ax3.bar(xs - w, [r["rule"] for r in ctrl], w, color="0.6", label="rule")
    ax3.bar(xs, [r["mech"] for r in ctrl], w, color=RED, alpha=0.7, label="mechanism")
    ax3.bar(xs + w, [r["sigma"] if r["observed"] else 0 for r in ctrl], w, color=ORANGE, edgecolor="k", linewidth=0.5, label="simulation")
    for i, r in enumerate(ctrl):
        if not r["observed"]: ax3.text(i + w, 1.0, "pending", rotation=90, fontsize=5.5, ha="center", va="bottom", color="0.4")
        _fx(ax3.text(i, 33.5, f"O = {r['O']:+.0f} Å", ha="center", va="top", fontsize=5.8, color=RED if r["O"] > 0 else BLUE))
    ax3.set_xticks(xs); ax3.set_xticklabels([f"φ = {r['phi']:.2f}" for r in ctrl], fontsize=6.5)
    ax3.set_ylabel("2D strength at θ = 20° (N/m)"); ax3.set_ylim(0, 35); ax3.legend(loc="upper right", frameon=False, fontsize=5.8, bbox_to_anchor=(1.02, 0.9))
    for a, l in zip((ax, ax2, ax3), "abc"): letter(a, l)
    save(fig, "fig3_alignment_cliff")
    clean = lambda r: {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items() if k != "params"}
    NUM.update(slit_sweep=[clean(r) for r in sweep], slit_controls=[clean(r) for r in ctrl])


# --------------------------------------------------------------------------- figure 4: hierarchy is not a free lunch (alignment index)
def fig4():
    def ordered(r):
        tg = r.get("tags") or []; p = r["params"]
        return not any(k in r["name"] for k in ("jitter", "sizedis")) and "disorder" not in tg
    meshes = [r for r in ALL if r["family"] in ("nanomesh_single", "nanomesh_hier") and 0.17 <= (r.get("porosity") or 0) <= 0.23 and ordered(r)]
    H1 = [r for r in meshes if r["family"] == "nanomesh_single"]; H2 = [r for r in meshes if r["family"] == "nanomesh_hier" and r.get("hierarchy_levels", 2) == 2]; H3 = [r for r in meshes if r["family"] == "nanomesh_hier" and r.get("hierarchy_levels", 2) >= 3]
    A1, s1 = np.array([Aidx(r) for r in H1]), np.array([sig(r) for r in H1]); A2, s2 = np.array([Aidx(r) for r in H2]), np.array([sig(r) for r in H2])
    fit1 = np.polyfit(A1, s1, 1); fit2 = np.polyfit(A2, s2, 1) if len(H2) > 2 else fit1
    Aall = np.array([Aidx(r) for r in meshes]); sall = np.array([sig(r) for r in meshes]); fitall = np.polyfit(Aall, sall, 1)
    resid = sall - np.polyval(fitall, Aall)
    r_all, rs_all = pearson(Aall, sall); r1, _ = pearson(A1, s1); r2, _ = pearson(A2, s2)
    prem = [(r, sig(r) - np.polyval(fit1, Aidx(r))) for r in H2 + H3]
    # bootstrap (designs resampled within level): 95 % interval of the mean premium and of the slope difference
    rng = np.random.default_rng(0); iH1 = [i for i, r in enumerate(meshes) if r in H1]; iH2 = [i for i, r in enumerate(meshes) if r in H2]; iN = [i for i, r in enumerate(meshes) if r in H2 + H3]
    bp, bd = [], []
    for _ in range(5000):
        j1 = rng.choice(iH1, len(iH1)); j2 = rng.choice(iH2, len(iH2)); jn = rng.choice(iN, len(iN)); g1 = np.polyfit(Aall[j1], sall[j1], 1); g2 = np.polyfit(Aall[j2], sall[j2], 1)
        bp.append(float(np.mean(sall[jn] - np.polyval(g1, Aall[jn])))); bd.append(float(g2[0] - g1[0]))
    bp, bd = np.array(bp), np.array(bd)
    NUM.update(premium_ci=[float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))], premium_p_pos=float((bp > 0).mean()), slope_diff=float(fit2[0] - fit1[0]), slope_diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])
    NUM.update(align_n=len(meshes), align_r=r_all, align_rs=rs_all, align_r_H1=r1, align_r_H2=r2, align_slope_all=float(fitall[0]), align_slope_H1=float(fit1[0]), align_slope_H2=float(fit2[0]),
               align_resid_std=float(resid.std()), premium_mean=float(np.mean([p for _, p in prem])), premium_std=float(np.std([p for _, p in prem])), premium=[(r["name"], round(p, 2)) for r, p in prem])
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.25], wspace=0.42, hspace=0.55, left=0.07, right=0.99, top=0.95, bottom=0.1)
    # (a) definition: three example structures with A
    axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
    ex = [("S2_slit_90deg", "slits ⟂ load"), ("S2_H1_square_D40", "square mesh"), ("S2_slit_0deg", "slits ∥ load")]
    for k, (nm, lab) in enumerate(ex):
        ins = axa.inset_axes([k * 0.34, 0.36, 0.30, 0.52]); ins.imshow(thumbnail(BY[nm], px=180)); ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values(): sp.set_edgecolor(famcol(BY[nm])); sp.set_linewidth(0.8)
        ins.set_title(f"{lab}\nA = {Aidx(BY[nm]):+.2f}", fontsize=6.2, pad=1.5)
    _fx(axa.text(0.5, 0.02, "$A = a\\cos 2\\varphi$: $a$ = anisotropy of the structure\ntensor of the solid phase, $\\varphi$ = its ligament\ndirection relative to the load", transform=axa.transAxes, ha="center", va="bottom", fontsize=5.9, linespacing=1.25))
    letter(axa, "a", dx=0.0)
    # (b) strength vs A
    axb = fig.add_subplot(gs[0, 1:]); style(axb)
    mk = {1: "o", 2: "s", 3: "^"}
    for r in meshes:
        lev = r.get("hierarchy_levels", 1) if r["family"] == "nanomesh_hier" else 1
        el = r["params"].get("aspect", 1.0) > 1.2
        col = {1: BLUE, 2: RED, 3: PURPLE}[min(lev, 3)]
        axb.scatter(Aidx(r), sig(r), marker=mk[min(lev, 3)], s=28 if el else 20, color=col, edgecolors="k" if el else "white", linewidths=0.5, zorder=3)
    xx = np.linspace(-0.3, 0.7, 20)
    axb.plot(xx, np.polyval(fit1, xx), "-", color=BLUE, lw=0.9, label=f"single level fit (r = {r1:.2f})")
    axb.plot(xx, np.polyval(fit2, xx), "-", color=RED, lw=0.9, label=f"two levels fit (r = {r2:.2f})")
    axb.plot(xx, np.polyval(fitall, xx), ":", color="0.3", lw=0.8, label=f"all meshes (r = {r_all:.2f})")
    ann = {"S2_ellipse_90deg": (4, -9), "S2_ellipse_0deg": (5, 2), "S7_H2_veinsX_W12_ellipseX": (4, -24), "S4_H2_D40_W8_veins_y": (5, -8)}
    for r in meshes:
        if r["name"] in ann:
            axb.annotate(r["name"].split("_", 1)[1], (Aidx(r), sig(r)), xytext=ann[r["name"]], textcoords="offset points", fontsize=5.2, color="0.3", ha="right" if ann[r["name"]][0] < 0 else "left", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5, shrinkA=0, shrinkB=3))
    h = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=4, label="1 level"), Line2D([], [], marker="s", ls="", color=RED, markersize=4, label="2 levels"), Line2D([], [], marker="^", ls="", color=PURPLE, markersize=4, label="3 levels"),
         Line2D([], [], marker="o", ls="", color="w", markeredgecolor="k", markersize=4, label="elongated pores")]
    leg1 = axb.legend(handles=h, loc="lower right", frameon=False, ncol=2, columnspacing=0.8, fontsize=6.2); axb.add_artist(leg1)
    axb.legend(loc="upper left", frameon=False, fontsize=6.2)
    axb.set_xlabel("load-path alignment index A"); axb.set_ylabel("2D strength (N/m)"); axb.set_ylim(4, 24); axb.set_xlim(-0.3, 0.75)
    axb.set_xlabel(f"load-path alignment index A  (n = {len(meshes)} meshes at porosity 0.17–0.23)")
    letter(axb, "b")
    # (c) hierarchy premium
    axc = fig.add_subplot(gs[1, :2]); style(axc)
    prem_sorted = sorted(prem, key=lambda t: Aidx(t[0]))
    names = [r["name"].split("_", 1)[1] for r, _ in prem_sorted]; vals = [p for _, p in prem_sorted]
    cols = [PURPLE if (r.get("hierarchy_levels", 2) >= 3) else RED for r, _ in prem_sorted]
    axc.bar(range(len(vals)), vals, color=cols, edgecolor="none")
    _bg(axc.axhspan(-resid.std(), resid.std(), color="0.85", alpha=0.6, lw=0)); axc.axhline(0, color="0.2", lw=0.7)
    axc.set_xticks(range(len(vals))); axc.set_xticklabels(names, rotation=60, ha="right", fontsize=5.6)
    axc.set_ylabel("hierarchy premium (N/m)\n$\\sigma$ − single-level trend at the same A")
    axc.text(0.97, 0.96, f"nested designs sorted by A\n(left: across, right: along the load)\nmean premium {np.mean(vals):+.1f} ± {np.std(vals):.1f} N/m\n95 % bootstrap interval {NUM['premium_ci'][0]:+.1f} to {NUM['premium_ci'][1]:+.1f} N/m\ngrey band: single-level scatter ±{resid.std():.1f}", transform=axc.transAxes, va="top", ha="right", fontsize=6.0, linespacing=1.2)
    letter(axc, "c")
    # (d) work to failure vs A
    axd = fig.add_subplot(gs[1, 2]); style(axd)
    for r in meshes:
        lev = r.get("hierarchy_levels", 1) if r["family"] == "nanomesh_hier" else 1
        el = r["params"].get("aspect", 1.0) > 1.2
        axd.scatter(Aidx(r), W(r), marker=mk[min(lev, 3)], s=26 if el else 18, color={1: BLUE, 2: RED, 3: PURPLE}[min(lev, 3)], edgecolors="k" if el else "white", linewidths=0.5, zorder=3)
    rw, _ = pearson(Aall, [W(r) for r in meshes]); NUM.update(align_r_W=rw)
    axd.set_xlabel("alignment index A"); axd.set_ylabel("work to failure (J/m$^2$, proxy)"); axd.text(0.03, 0.96, f"r = {rw:.2f}", transform=axd.transAxes, va="top", fontsize=6.8)
    letter(axd, "d")
    save(fig, "fig4_hierarchy_alignment")


# --------------------------------------------------------------------------- figure 5: disorder, gradients, halos are costs
def fig5():
    def vals(names, key):
        return [(n, M(BY[n], key)) for n in names if n in BY]
    ref_names = ["S2_H1_p16", "S1_mesh_p16_phi0.2", "S6_mesh_p16_offset1", "S6_mesh_p16_offset2", "S6_mesh_p16_offset3"]
    groups = [
        ("positional disorder\n(jitter, Å)", {"0": ref_names, "1": ["S2_mesh_jitter1.0"], "2": ["S2_mesh_jitter2.0", "S6_mesh_jitter2.0_s2", "S6_mesh_jitter2.0_s3"], "3": ["S2_mesh_jitter3.0"]}, ref_names),
        ("size dispersion\n(rel. std)", {"0": ref_names, "0.15": ["S2_mesh_sizedis0.15", "S6_mesh_sizedis0.15_s2", "S6_mesh_sizedis0.15_s3"], "0.3": ["S2_mesh_sizedis0.3"]}, ref_names),
        ("Voronoi network\n(regularity)", {"0": ["S2_voronoi_n25_reg0.0", "S6_voronoi_n25_reg0.0_s2", "S6_voronoi_n25_reg0.0_s3", "S2_voronoi_n50_reg0.0"], "0.3": ["S5_voronoi_n25_reg0.3"], "0.6": ["S2_voronoi_n25_reg0.6", "S6_voronoi_n25_reg0.6_s2", "S6_voronoi_n25_reg0.6_s3"], "0.8": ["S5_voronoi_n25_reg0.8", "S7_voronoi_n35_reg0.8_s7"], "1": ["S2_voronoi_n25_reg1.0"]}, ref_names),
        ("pore-size\ngradient", {"uniform": ref_names, "along\nload": ["S2_graded_x"], "radial": ["S2_graded_radial"], "weak\nband": ["S7_graded_x_weak"]}, ref_names),
        ("flaw halo\n(hole + rings)", {"hole 20 Å": ["S1_hole_d20"], "+2 rings": ["S2_hole20_rings2"], "+far\npores": ["S2_hole20_background"], "hole 30 Å": ["S4_hole_d30"], "+3 rings": ["S7_hole30_rings3"]}, None),
    ]
    fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[1, 0.9, 1.2, 1.05, 1.25])); plt.subplots_adjust(wspace=0.5, hspace=0.45, left=0.07, right=0.99, top=0.92, bottom=0.17)
    summary = {}
    for j, (title, levels, ref) in enumerate(groups):
        for i, (key, ylab) in enumerate([("strength_Nm", "2D strength (N/m)"), ("work_to_failure_J_m2", "work to failure (J/m$^2$)")]):
            ax = axs[i, j]; style(ax)
            xs = list(levels.keys())
            for k, lev in enumerate(xs):
                v = [x[1] for x in vals(levels[lev], key)]
                if not v: continue
                col = "0.35" if (ref and levels[lev] is ref) or (ref is None and k in (0, 3)) else (RED if key == "strength_Nm" else BLUE)
                ax.scatter([k] * len(v), v, s=14, color=col, alpha=0.75, zorder=3, edgecolors="none")
                ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=col, lw=1.3, zorder=4)
                summary[f"{title.split(chr(10))[0]}|{lev}|{key}"] = (float(np.mean(v)), float(np.std(v)), len(v))
            if ref:
                rv = [x[1] for x in vals(ref, key)]; _bg(ax.axhspan(np.mean(rv) - np.std(rv), np.mean(rv) + np.std(rv), color="0.85", alpha=0.6, lw=0)); ax.axhline(np.mean(rv), color="0.3", lw=0.6, ls="--")
            ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.replace("\n", " ") for x in xs] if j >= 3 else xs, fontsize=6.0, rotation=35 if j >= 3 else 0, ha="right" if j >= 3 else "center")
            if i == 0: ax.set_title(title, fontsize=7)
            if j == 0: ax.set_ylabel(ylab)
            if key == "strength_Nm": ax.set_ylim(0, 22)
            else: ax.set_ylim(0, 2.6)
    NUM.update(costs=summary)
    fig.text(0.5, 0.01, "grey: ordered fine mesh at the same porosity (registry replicates, mean ± s.d.); flaw-halo panels: the bare hole is the reference (dark)", ha="center", fontsize=6.5)
    for a, l in zip(axs[0], "abcde"): letter(a, l)
    save(fig, "fig5_costs")


# --------------------------------------------------------------------------- figure 6: pre-registered predictions
CLASS = {"S7_slit_20deg": ("new mechanism: en-echelon coalescence", RED), "S7_H2_veinsX_W12_ellipseX": ("new: alignment effects add across scales", RED),
         "S7_slit_0deg_armchair": ("missing input: lattice orientation", ORANGE), "S7_hole30_rings3": ("weakest section, not the average", ORANGE),
         "S7_graded_x_weak": ("weakest section, not the average", ORANGE), "S7_voronoi_n35_reg0.8_s7": ("seed scatter of random networks", "0.5")}


def fig6():
    E = json.load(open(os.path.join(ROOT, "experiments", "holdouts", "holdout_evaluation.json")))
    P7 = json.load(open(os.path.join(ROOT, "experiments", "predictions", "holdout_predictions.json")))
    PP = json.load(open(os.path.join(ROOT, "experiments", "predictions", "paper_sweeps_predictions.json")))
    EPp = os.path.join(ROOT, "experiments", "holdouts", "paper_sweeps_evaluation.json")
    EP = json.load(open(EPp)) if os.path.exists(EPp) else None
    rows = E["rows"]
    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55], width_ratios=[1, 1.15], wspace=0.75, hspace=0.55, left=0.075, right=0.99, top=0.96, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    srt = sorted(rows, key=lambda r: (r["observed"]["strength_Nm"] - r["predicted"]["strength_Nm"]) / r["predicted"]["strength_Nm"])
    num = {r["name"]: k + 1 for k, r in enumerate(srt)}
    for r in rows:
        p, o = r["predicted"]["strength_Nm"], r["observed"]["strength_Nm"]; lab, col = CLASS.get(r["name"], ("rule held (within 15%)", GREEN))
        ax.scatter(p, o, s=30, color=col, edgecolors="k", linewidths=0.4, zorder=3)
        ax.annotate(str(num[r["name"]]), (p, o), xytext=(4, 3), textcoords="offset points", fontsize=5.6, color="0.2", ha="left")
    ax.plot([5, 26], [5, 26], "--", color="0.5", lw=0.7); ax.fill_between([5, 26], [5 * 0.85, 26 * 0.85], [5 * 1.15, 26 * 1.15], color="0.9", lw=0)
    ax.set_xlabel("predicted strength (N/m), written before simulation"); ax.set_ylabel("simulated strength (N/m)"); ax.set_xlim(5, 26); ax.set_ylim(5, 26); ax.set_xticks([5, 10, 15, 20, 25]); ax.set_yticks([5, 10, 15, 20, 25])
    h = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", markersize=4, label=l) for l, c in [("rule held (≤ 15%)", GREEN), ("new mechanism", RED), ("physics missing in predictor", ORANGE), ("seed scatter", "0.5")]]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=1.0, title="round 1: twelve holdout designs", title_fontsize=7)
    letter(ax, "a", dy=1.22)
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    for i, r in enumerate(srt):
        e = (r["observed"]["strength_Nm"] - r["predicted"]["strength_Nm"]) / r["predicted"]["strength_Nm"] * 100
        lab, col = CLASS.get(r["name"], ("", GREEN)); ax2.barh(i, e, color=col, height=0.7)
        if lab: ax2.text(max(18, e + 3), i, lab, va="center", ha="left", fontsize=5.2, color="0.3")
    ax2.set_yticks(range(len(srt))); ax2.set_yticklabels([f"{i + 1}  {r['name'].split('_', 1)[1]}" for i, r in enumerate(srt)], fontsize=5.4)
    _bg(ax2.axvspan(-15, 15, color="0.9", lw=0)); ax2.axvline(0, color="0.3", lw=0.6); ax2.set_xlabel("(simulated − predicted) / predicted (%)"); ax2.set_xlim(-70, 100)
    med = E["summary"]["strength_Nm"]["median_abs_rel_error"] * 100
    ax2.text(0.02, 0.97, f"median |error| {med:.0f}%", transform=ax2.transAxes, ha="left", va="top", fontsize=6.5)
    letter(ax2, "b", dy=1.22)
    # round 2: the 18 pre-registered sweeps, two rules
    ax3 = fig.add_subplot(gs[1, 0]); style(ax3)
    ax4 = fig.add_subplot(gs[1, 1]); style(ax4)
    if EP:
        er = EP["rows"]; order = sorted(er, key=lambda x: (x["series"], x["name"]))
        colof = lambda x: ORANGE if x["name"].startswith("P1") else (RED if x["name"].startswith("P2") else (BLUE if "H1" in x["name"] else PURPLE))
        for x in order:
            c = colof(x)
            ax3.plot([x["rule"], x["mech"]], [x["simulated"], x["simulated"]], "-", color="0.7", lw=0.6, zorder=2)
            ax3.scatter(x["rule"], x["simulated"], s=22, facecolors="white", edgecolors=c, linewidths=0.8, zorder=3)
            ax3.scatter(x["mech"], x["simulated"], s=26, color=c, edgecolors="k", linewidths=0.4, zorder=4)
        ax3.plot([3, 30], [3, 30], "--", color="0.5", lw=0.7); ax3.fill_between([3, 30], [3 * 0.85, 30 * 0.85], [3 * 1.15, 30 * 1.15], color="0.9", lw=0)
        num2 = {x["name"]: k + 1 for k, x in enumerate(order)}
        for x in order:
            if x["name"] in ("P1_slit_60deg", "P1_slit_75deg", "P2_slit_20deg_phi0.10", "P1_slit_05deg", "P2_slit_20deg_phi0.30"):
                ax3.annotate(str(num2[x["name"]]), (x["mech"], x["simulated"]), xytext=(4, -7), textcoords="offset points", fontsize=5.6, color="0.2")
        ax3.set_xlim(3, 30); ax3.set_ylim(3, 30); ax3.set_xlabel("predicted strength (N/m): hollow = net-section rule, filled = mechanism rule\nnumbers refer to the rows of panel d"); ax3.set_ylabel("simulated strength (N/m)")
        h2 = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", markersize=4, label=l) for l, c in [("angle sweep", ORANGE), ("overlap controls", RED), ("alignment, single level", BLUE), ("alignment, two levels", PURPLE)]]
        ax3.legend(handles=h2, loc="upper left", frameon=False, fontsize=5.8, handletextpad=0.2)
        ax3.set_title("round 2: eighteen pre-registered sweeps", fontsize=7.5)
        # (d) errors of both rules
        for i, x in enumerate(order):
            ax4.barh(i + 0.18, x["err_rule"] * 100, height=0.34, color="0.75")
            ax4.barh(i - 0.18, x["err_mech"] * 100, height=0.34, color=colof(x))
        ax4.set_yticks(range(len(order))); ax4.set_yticklabels([f"{k + 1}  {x['name'].split('_', 1)[1]}" for k, x in enumerate(order)], fontsize=4.9)
        _bg(ax4.axvspan(-15, 15, color="0.9", lw=0)); ax4.axvline(0, color="0.3", lw=0.6); ax4.set_xlabel("(predicted − simulated) / simulated (%)"); ax4.set_xlim(-60, 240)
        sm = EP["summary"]
        ax4.text(0.98, 0.40, f"median |error|, slit designs:\nrule {sm['slit_median_abs_err_rule']*100:.0f}%, mechanism {sm['slit_median_abs_err_mech']*100:.0f}%\nalignment designs:\ninterpolation in A {sm['align_median_abs_err_interp']*100:.0f}%", transform=ax4.transAxes, ha="right", va="center", fontsize=5.6)
        h3 = [Patch(color="0.75", label="net-section rule"), Patch(color=ORANGE, label="mechanism / interpolation rule")]
        ax4.legend(handles=h3, loc="lower right", frameon=False, fontsize=5.8)
    letter(ax3, "c"); letter(ax4, "d")
    # (e) timeline of the two pre-registrations
    ax5 = fig.add_subplot(gs[2, :]); ax5.axis("off")
    runs7 = sorted([r for r in DISC if r.get("stage") == "stage7_holdouts"], key=lambda r: r["run_id"])
    def fmt(rid): return datetime.datetime.strptime(rid, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    ev1 = [("predictions written", P7["written"][:16] + f"\nSHA-256 {P7['sha256_of_content'][:12]}…", RED), ("first of 12 simulations", fmt(runs7[0]["run_id"][4:19]), BLUE), ("last simulation", fmt(runs7[-1]["run_id"][4:19]), BLUE), ("evaluation", E["evaluated"][:16], GREEN)]
    runsP = sorted(PAPER, key=lambda r: r["run_id"])
    ev2 = [("predictions written (two rules)", PP["written"][:16] + f"\nSHA-256 {PP['sha256_of_content'][:12]}…", RED)]
    if runsP: ev2 += [("first of 18 simulations", fmt(runsP[0]["run_id"][4:19]), BLUE), ("last simulation", fmt(runsP[-1]["run_id"][4:19]), BLUE)]
    if EP: ev2 += [("evaluation", EP["summary"]["evaluated"][:16], GREEN)]
    for yy, (title, ev) in enumerate([("round 1", ev1), ("round 2", ev2)]):
        y0 = 0.78 - yy * 0.55
        _fx(ax5.text(0.0, y0, title, fontsize=6.8, fontweight="bold", transform=ax5.transAxes, va="center", ha="left"))
        ax5.plot([0.08, 0.98], [y0, y0], color="0.5", lw=1.0, transform=ax5.transAxes)
        for k, (lab, when, col) in enumerate(ev):
            x = 0.14 + k * (0.8 / max(1, len(ev) - 1))
            ax5.scatter([x], [y0], s=30, color=col, edgecolors="k", linewidths=0.4, transform=ax5.transAxes, zorder=3, clip_on=False)
            _fx(ax5.text(x, y0 - 0.1, lab + "\n" + when, ha="center", va="top", fontsize=5.4, transform=ax5.transAxes))
    letter(ax5, "e", dx=0.0, dy=0.9)
    save(fig, "fig6_preregistration")
    NUM.update(holdout_median_err=med, holdout_mode_acc=E["summary"].get("fracture_mode_class_accuracy"), pred7_written=P7["written"], pred7_sha=P7["sha256_of_content"], first_holdout_run=fmt(runs7[0]["run_id"][4:19]), evaluated7=E["evaluated"],
               predP_written=PP["written"], predP_sha=PP["sha256_of_content"], n_paper_recorded=len(runsP), first_paper_run=fmt(runsP[0]["run_id"][4:19]) if runsP else "", last_paper_run=fmt(runsP[-1]["run_id"][4:19]) if runsP else "",
               sweep_err=(EP["summary"] if EP else {}))


# --------------------------------------------------------------------------- numbers export
def export_numbers():
    json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=lambda o: o if isinstance(o, (int, float, str)) else str(o))
    def macro(name, val, fmt="{:.2f}"):
        s = fmt.format(val) if isinstance(val, (int, float)) else str(val)
        return f"\\newcommand{{\\{name}}}{{{s}}}\n"
    lines = [macro("nDiscovery", NUM["n_discovery"], "{:d}"), macro("nPaperRuns", NUM["n_paper"], "{:d}"), macro("sigmaZeroZZ", NUM["sigma0_zz"], "{:.1f}"), macro("YZero", NUM["Y0"], "{:.0f}"),
             macro("bandN", NUM["band_n"], "{:d}"), macro("bandSigMin", NUM["band_spec_sigma_min"], "{:.1f}"), macro("bandSigMax", NUM["band_spec_sigma_max"], "{:.1f}"), macro("bandSigRatio", NUM["band_spec_sigma_ratio"], "{:.1f}"),
             macro("bandYMin", NUM["band_spec_Y_min_lig"], "{:.0f}"), macro("bandYMax", NUM["band_spec_Y_max_lig"], "{:.0f}"), macro("bandYRatio", NUM["band_spec_Y_ratio_lig"], "{:.1f}"), macro("bandYExceptions", NUM["band_n_exceptions"], "{:d}"), macro("bandYExcList", "; ".join(f"{n.split(chr(95), 1)[1].replace(chr(95), chr(92) + chr(95))}: {v:.1f} N/m" for n, v, _ in NUM["band_Y_exceptions"])),
             macro("rSigRho", NUM["r_sigma_rho"]), macro("rYRho", NUM["r_Y_rho"]), macro("rsSigRho", NUM["rs_sigma_rho"]), macro("rsYRho", NUM["rs_Y_rho"]),
             macro("effYlo", NUM["eff_Y_p10"]), macro("effYhi", NUM["eff_Y_p90"]), macro("effSlo", NUM["eff_s_p10"]), macro("effShi", NUM["eff_s_p90"]),
             macro("lpRrho", NUM["lp_r_rho"]), macro("lpRmsf", NUM["lp_r_msf"]), macro("lpRpred", NUM["lp_r_pred"]), macro("lpRSrho", NUM["lp_rs_rho"]), macro("lpRSmsf", NUM["lp_rs_msf"]), macro("lpRSpred", NUM["lp_rs_pred"]), macro("lpN", NUM["lp_n"], "{:d}"),
             macro("bestSlitEff", NUM["best_eff"].get("slit_array", {}).get("eff", float("nan"))), macro("bestSlitName", NUM["best_eff"].get("slit_array", {}).get("name", "").replace("_", "\\_")),
             macro("alignN", NUM["align_n"], "{:d}"), macro("alignR", NUM["align_r"]), macro("alignRS", NUM["align_rs"]), macro("alignRHone", NUM["align_r_H1"]), macro("alignRHtwo", NUM["align_r_H2"]), macro("alignSlope", NUM["align_slope_all"], "{:.1f}"),
             macro("alignSlopeHone", NUM["align_slope_H1"], "{:.1f}"), macro("alignSlopeHtwo", NUM["align_slope_H2"], "{:.1f}"), macro("alignResidStd", NUM["align_resid_std"], "{:.1f}"), macro("premiumMean", NUM["premium_mean"], "{:+.1f}"), macro("premiumStd", NUM["premium_std"], "{:.1f}"), macro("alignRW", NUM["align_r_W"]),
             macro("premiumCIlo", NUM["premium_ci"][0], "{:+.1f}"), macro("premiumCIhi", NUM["premium_ci"][1], "{:+.1f}"), macro("premiumPpos", 100 * NUM["premium_p_pos"], "{:.1f}"),
             macro("slopeDiff", NUM["slope_diff"], "{:+.1f}"), macro("slopeDiffLo", NUM["slope_diff_ci"][0], "{:+.1f}"), macro("slopeDiffHi", NUM["slope_diff_ci"][1], "{:+.1f}"),
             macro("snetStraight", NUM["snet_classes"]["straight"], "{:.0f}"), macro("snetWide", NUM["snet_classes"]["coarse_round"], "{:.0f}"), macro("snetNarrow", NUM["snet_classes"]["fine_round"], "{:.0f}"), macro("snetRandom", NUM["snet_classes"]["random"], "{:.0f}"),
             macro("holdoutMedianErr", NUM["holdout_median_err"], "{:.0f}"), macro("predSevenWritten", NUM["pred7_written"]), macro("predSevenSha", NUM["pred7_sha"][:16]), macro("firstHoldoutRun", NUM["first_holdout_run"]), macro("evaluatedSeven", NUM["evaluated7"][:16]),
             macro("predPaperWritten", NUM["predP_written"]), macro("predPaperSha", NUM["predP_sha"][:16]), macro("nPaperRecorded", NUM["n_paper_recorded"], "{:d}"),
             macro("firstPaperRun", NUM.get("first_paper_run", "")), macro("lastPaperRun", NUM.get("last_paper_run", "")),
             macro("sweepErrRule", 100 * NUM.get("sweep_err", {}).get("slit_median_abs_err_rule", float("nan")), "{:.0f}"), macro("sweepErrMech", 100 * NUM.get("sweep_err", {}).get("slit_median_abs_err_mech", float("nan")), "{:.0f}"),
             macro("sweepErrAlign", 100 * NUM.get("sweep_err", {}).get("align_median_abs_err_interp", float("nan")), "{:.0f}"), macro("sweepEvaluated", NUM.get("sweep_err", {}).get("evaluated", "")[:16]),
             macro("ctrlRetained", 100 * next((r["sigma"] for r in NUM.get("slit_controls", []) if r["observed"] and r["phi"] < 0.15), float("nan")) / M(BY["S5_slit0_phi0.1"], "strength_Nm"), "{:.0f}"),
             macro("slitSweepList", ", ".join(f"{r['theta']:.0f}$^\\circ$: {r['sigma']:.1f}" for r in sorted({r['theta']: r for r in sorted(NUM.get('slit_sweep', []), key=lambda r: abs((r['phi'] or 0) - 0.2), reverse=True)}.values(), key=lambda r: r['theta']) if r["observed"]))]
    for f in FAMORDER:
        if f in NUM["best_eff"]:
            key = "".join(w.capitalize() for w in f.split("_"))
            lines.append(macro(f"eff{key}", NUM["best_eff"][f]["eff"]))
    open(os.path.join(OUT, "numbers.tex"), "w").write("".join(lines))
    # LaTeX table of the pre-registered sweeps
    PP = json.load(open(os.path.join(ROOT, "experiments", "predictions", "paper_sweeps_predictions.json")))
    rows = ["\\begin{tabular}{llrrrrrrrl}", "\\toprule", "design & series & $\\phi$ & $A_\\mathrm{min}/A$ & $A$ & $O$ (\\AA) & rule & mech. & simulated & mode\\\\ \\midrule"]
    for p_ in PP["predictions"]:
        nm = p_["name"]; r = BY.get(nm); o = p_["slit_tip_overlap_A"]
        simv = f"{sig(r):.1f}" if r else "--"; mode = (str(M(r, "fracture_mode")).split(" ")[0] if r else "--")
        rows.append(f"{nm.replace('_', chr(92) + '_')} & {p_['series'].replace('_', chr(92) + '_')} & {p_['porosity']:.2f} & {p_['minSF']:.2f} & {p_['alignment_A']:+.2f} & " + (f"{o:+.1f}" if o == o else "--") + f" & {p_['predicted']['strength_Nm_rule']:.1f} & {p_['predicted']['strength_Nm_mechanism']:.1f} & {simv} & {mode}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "sweeps_table.tex"), "w").write("\n".join(rows))


if __name__ == "__main__":
    only = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k in only:
        globals()[f"fig{k}"]()
    export_numbers()
    print("figures written:", sorted(os.listdir(OUT)))
