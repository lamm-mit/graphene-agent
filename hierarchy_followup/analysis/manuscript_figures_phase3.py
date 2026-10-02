"""Manuscript figures, SI figures, LaTeX tables, plotted data and number macros of the hierarchy follow-up.

The follow-up is recorded in this tree as "phase 3" (campaign "hierarchy", stage "hierarchy_followup"); in the manuscript
it is the later human-AI follow-up at larger scales (Sec. "Hierarchy effects at larger scales").  Every number is read
from this tree's database and prediction files; the phase-1/2 tree is read only through reference copies.

Writes (editable code = this file; plotted data = data/*.json next to the figures):
  figures/hierarchy/manuscript/<name>.{pdf,png,svg}, data/<name>.json, hierarchy_numbers.tex, tables/*.tex
and copies the PDFs, data, macros and tables into the manuscript folder
  ../generated/hierarchy_exports/figures/hierarchy/           (main-text figures, data, tables)
  ../generated/hierarchy_exports/figures/si/hierarchy/        (SI figures)
  ../generated/hierarchy_exports/figures/hierarchy_numbers.tex

    python analysis/manuscript_figures_phase3.py
"""
from __future__ import annotations
import sys, os, json, shutil, types, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PAPER = os.path.join(os.path.dirname(ROOT), "generated/hierarchy_exports")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from analysis import paper_figures_phase3 as PF
from analysis.paper_figures_phase3 import (axin, style, letter, snap, under, two, curve, traj, frame_for, first_drop_frame, field, OWN, EV,
                                           DARK, RED, BLUE, ORANGE, GREEN, GREY, PURPLE, BROWN)
from analysis.assess_phase1 import curve_metrics
from analysis.hierarchy_followup_analysis import load, rows_T1, rows_T2, rows_T3, rows_T4, rows_T5, verdict_T1, verdict_T2, verdict_T3, verdict_T4, verdict_T5, crack_path
from atomistics.descriptors.columns import vein_bands, in_vein
from experiments import db

OUT = os.path.join(ROOT, "figures", "hierarchy", "manuscript"); DATA_DIR = os.path.join(OUT, "data"); TAB_DIR = os.path.join(OUT, "tables")
P_MAIN = os.path.join(PAPER, "figures", "hierarchy"); P_SI = os.path.join(PAPER, "figures", "si", "hierarchy")
for d in (OUT, DATA_DIR, TAB_DIR, P_MAIN, P_SI, os.path.join(P_MAIN, "data"), os.path.join(P_MAIN, "tables")):
    os.makedirs(d, exist_ok=True)
CYAN, PINK = "#17becf", "#e377c2"
COL = {"pristine sheet": GREY, "one level": BLUE, "one level, slits": ORANGE, "one level, solid bars": CYAN, "two levels, solid veins": RED,
       "fibrous veins": PURPLE, "porous veins": PINK, "three levels": BROWN, "grids": GREEN}
NUM, DATA, AUDIT = {}, {}, {}
if os.path.exists(os.path.join(OUT, "data", "numbers.json")):      # subset runs keep the numbers of the figures not regenerated
    NUM.update(json.load(open(os.path.join(OUT, "data", "numbers.json"))))
own, ref, preds = load()
R = json.load(open(os.path.join(ROOT, "reference", "phase1_reference_numbers.json")))
FACTS = json.load(open(os.path.join(ROOT, "results", "facts_phase3.json")))
S0, W0 = R["references"]["S1_pristine_zz"]["strength_Nm"], R["references"]["S1_pristine_zz"]["work_to_failure_J_m2"]   # phase-1 pristine zigzag sheet

# overlap audit of the manuscript (binds to the phase-1 database through make_figures; stub that import)
try:
    sys.modules.setdefault("make_figures", types.ModuleType("make_figures"))
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "paper_analysis", "current"))
    import check_overlaps as CO
except Exception as e:      # pragma: no cover
    CO = None; print("overlap audit unavailable:", repr(e)[:100])


# --------------------------------------------------------------------------------------------------- helpers
def snap(ax, name, k, crop=None, bands=True, s=None, sdmg=None, lw=None):
    """Atoms (dark), bonds (grey), atoms that lost a bond (red), load-parallel veins shaded; everything clipped to the
    window so that the overlap audit sees only what is drawn."""
    rec = OWN[name]; T = traj(name)
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    for a in range(2): P[:, a] = np.mod(P[:, a], c[a, a])
    big = c[0, 0] > 200
    s = s or (0.12 if big else 0.35); sdmg = sdmg or (1.2 if big else 2.4); lw = lw or (0.1 if big else 0.22)
    x0, x1, y0, y1 = crop if crop else (-1.5, c[0, 0] + 1.5, -1.5, c[1, 1] + 1.5)
    if bands:
        for yc, hw, sy in (vein_bands(rec.get("design") or {}) or []):
            for yb in (yc - hw, yc - hw - c[1, 1], yc - hw + c[1, 1]):
                lo, hi = max(yb, y0), min(yb + 2 * hw, y1)
                if hi > lo: ax.add_patch(Rectangle((x0, lo), x1 - x0, hi - lo, facecolor="#e9eef5", edgecolor="none", zorder=0))
    m = 1.6
    inside = (P[:, 0] > x0 - m) & (P[:, 0] < x1 + m) & (P[:, 1] > y0 - m) & (P[:, 1] < y1 + m)
    if len(b):
        p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]; d = p1 - p0
        for a in range(2): d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
        keep = inside[b[:, 0]] & inside[b[:, 1]]
        ax.add_collection(LineCollection(np.stack([p0[keep], p0[keep] + d[keep]], 1), colors="0.45", linewidths=lw, zorder=1, rasterized=True))
    ax.scatter(P[inside, 0], P[inside, 1], s=s, c="#2b2b2b", linewidths=0, zorder=2, rasterized=True)
    dmg = np.asarray(PF.damaged_atoms_until(rec, T["eps_x"][k]), int)
    if len(dmg):
        dm = dmg[inside[dmg]]
        if len(dm): ax.scatter(P[dm, 0], P[dm, 1], s=sdmg, c=RED, linewidths=0, zorder=3, rasterized=True)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    return float(T["eps_x"][k]), float(T["sigma_xx"][k] * EV), int(T["n_broken_cum"][k])


def limited(name):
    """True if the run ended on a loading limit (post-peak or total strain) with load still carried."""
    t = str(own[name].get("termination", ""))
    return ("limit" in t) or ("max strain" in t)


def term_code(name):
    t = str(own[name].get("termination", ""))
    return "L" if "post-peak" in t else ("M" if "max strain" in t else "F")


def cm(name):
    return curve_metrics(own[name])


def save(fig, name, si=False):
    n = None
    if CO is not None:
        try:
            n = CO.audit(fig, name, verbose=True); print(f"overlap audit {name}: {n} conflicts")
        except Exception as e:
            print("overlap audit failed:", repr(e)[:100])
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    shutil.copy(os.path.join(OUT, name + ".pdf"), os.path.join(P_SI if si else P_MAIN, name + ".pdf"))
    AUDIT[name] = n


def tex_id(s):
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def f1(v): return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}"
def f2(v): return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"
def f3(v): return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def pair(fn, a, b):
    return f"{fn(a)}/{fn(b)}" if b is not None else fn(a)


def reg_names(base):
    return [n for n in (base + "_r0", base + "_r1") if n in own]


# --------------------------------------------------------------------------------------------------- figure 1: scale separation (main text)
def fig_hier_scale():
    FW, FH = 7.2, 7.35
    fig = plt.figure(figsize=(FW, FH)); D = {}
    y0 = 0.22; H = 1.5
    items = [("H3_1L_fine", "one level, fine pores", 0.35), ("H3_1L_coarse", "one level, coarse bars", 2.0), ("H3_2L", "two levels, 300 Å", 3.65), ("H1_veinsX", "two levels, 120 Å", 5.45)]
    for k, (n, lab, x) in enumerate(items):
        w = H if n.startswith("H3") else H * 120.55 / 300.1
        ax = axin(fig, x, y0 + (H - w), w, w, FW, FH)
        snap(ax, n + "_r0", 0)
        s, ds = two(reg_names(n), "strength_Nm"); W, dW = two(reg_names(n), "work_to_failure_J_m2")
        under(ax, f"{lab}\nσ = {s:.1f} N/m, W = {W:.2f} J/m²", dy=-0.04, fs=6.3)
        D[n] = dict(strength_mean=s, strength_halfrange=ds, W_mean=W, W_halfrange=dW)
        if k == 0: letter(ax, "a", dx=-0.06)
    # (b) curves at 300 A (registry 0 solid, registry 1 dotted)
    ax = axin(fig, 0.5, 2.45, 2.05, 1.65, FW, FH); style(ax); letter(ax, "b", dy=1.2); D["curves"] = {}
    for n, col, lab in (("H3_1L_fine", BLUE, "one level, fine"), ("H3_1L_coarse", GREEN, "one level, coarse"), ("H3_2L", RED, "two levels")):
        for reg, ls in ((0, "-"), (1, ":")):
            e, s = curve(f"{n}_r{reg}"); ax.plot(e, s, ls, color=col, lw=1.0, label=lab if reg == 0 else None)
            D["curves"][f"{n}_r{reg}"] = dict(strain=[float(v) for v in e], stress_Nm=[float(v) for v in s])
    ax.set_xlabel("engineering strain", labelpad=1); ax.set_ylabel("2D stress (N/m)", labelpad=1); ax.set_xlim(0, 0.2); ax.set_ylim(0, 21)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.5, columnspacing=1.0, handletextpad=0.5, fontsize=6.4)
    # (c) strength and work at 300 and 120 A
    ax = axin(fig, 3.0, 2.45, 2.25, 1.65, FW, FH); style(ax); letter(ax, "c", dy=1.2); D["bars"] = []
    sets = [("300 Å", [("H3_1L_fine", BLUE), ("H3_1L_coarse", GREEN), ("H3_2L", RED)]), ("120 Å", [("H1_ellipse", BLUE), ("H1_slit", ORANGE), ("H1_veinsX", RED)])]
    xt = []
    for g, (lab, its) in enumerate(sets):
        for m, (key, fac, ylab) in enumerate((("strength_Nm", 1.0, "strength"), ("work_to_failure_J_m2", 10.0, "work ×10"))):
            x0 = g * 2.3 + m * 1.0
            for i, (n, col) in enumerate(its):
                v, dv = two(reg_names(n), key); v, dv = v * fac, dv * fac
                ax.bar(x0 + i * 0.26, v, 0.24, color=col, yerr=dv, capsize=1.5, error_kw=dict(lw=0.6), zorder=3)
                ax.text(x0 + i * 0.26, v + dv + 0.6, f"{v:.0f}" if fac == 1 else f"{v/10:.1f}", ha="center", va="bottom", fontsize=5.6)
                D["bars"].append(dict(cell=lab, design=n, quantity=key, mean=v / fac, halfrange=dv / fac, limited=[limited(x) for x in reg_names(n)]))
            xt.append((x0 + 0.26, ylab))
        ax.text(g * 2.3 - 0.2, 39.5, lab, ha="left", va="top", fontsize=7.2, fontweight="bold")
    ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=6.5); ax.set_ylim(0, 40); ax.set_ylabel("strength (N/m); work × 10 (J/m²)", labelpad=1)
    ax.legend(handles=[Line2D([], [], color=c, lw=5) for c in (BLUE, GREEN, ORANGE, RED)], labels=["one level, fine", "one level, coarse", "one level, slits", "two levels"],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2, handlelength=1.2, ncol=2, columnspacing=1.2, handletextpad=0.5)
    # (d) load share before the peak in the two-level design
    ax = axin(fig, 5.5, 2.45, 1.5, 1.5, FW, FH); letter(ax, "d", dy=1.06)
    T = traj("H3_2L_r0"); ip = int(np.argmax(T["sigma_xx"])); k = int(np.searchsorted(T["eps_x"], 0.8 * T["eps_x"][ip]))
    P = T["positions"][k].astype(float); c = T["cells"][k]; v = T["peratom_virial"][k][:, 0].astype(float); share = v / v.mean()
    sc = ax.scatter(P[:, 0], P[:, 1], c=share, s=0.25, cmap="magma", vmin=0, vmax=2, linewidths=0, rasterized=True)
    ax.set_xlim(0, c[0, 0]); ax.set_ylim(0, c[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    bands = vein_bands(own["H3_2L_r0"]["design"]); inv = np.array([in_vein(y, bands) for y in P[:, 1]])
    NUM["VeinLoadShare"] = 100 * float(v[inv].sum() / v.sum()); NUM["VeinSectionShare"] = 100 * float(inv.mean()); NUM["LoadMapStrain"] = float(T["eps_x"][k])
    NUM["VeinLoadPerAtom"] = float(share[inv].mean()); NUM["FineLoadPerAtom"] = float(share[~inv].mean())
    cax = axin(fig, 7.08, 2.45, 0.06, 1.5, FW, FH); cb = fig.colorbar(sc, cax=cax, orientation="vertical"); cb.set_label("load per atom / mean", fontsize=6.0, labelpad=1); cb.ax.tick_params(labelsize=5.6, length=2)
    under(ax, f"ε = {T['eps_x'][k]:.2f}, before damage", dy=-0.04, fs=6.0)
    D["load_map"] = dict(strain=float(T["eps_x"][k]), vein_load_share=NUM["VeinLoadShare"] / 100, vein_atom_share=NUM["VeinSectionShare"] / 100, x=[float(u) for u in P[:, 0]], y=[float(u) for u in P[:, 1]], load_share=[float(u) for u in share])
    # (e) how they fail
    y0 = 4.85; H = 1.55
    panels = [("H3_2L_r0", first_drop_frame(traj("H3_2L_r0")), "two levels, first drop"), ("H3_2L_r0", frame_for(traj("H3_2L_r0"), "major_propagation"), "two levels, major propagation"),
              ("H3_1L_fine_r0", first_drop_frame(traj("H3_1L_fine_r0")), "one level, fine, first drop"), ("H3_1L_coarse_r0", first_drop_frame(traj("H3_1L_coarse_r0")), "one level, coarse, first drop")]
    D["snapshots"] = []
    for k, (n, f, lab) in enumerate(panels):
        ax = axin(fig, 0.35 + k * 1.72, y0, H, H, FW, FH)
        e, s, nb = snap(ax, n, f)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m, {nb} bonds broken", dy=-0.04, fs=6.0)
        D["snapshots"].append(dict(run=n, frame=int(f), label=lab, strain=e, stress_Nm=s, bonds_broken=nb))
        if k == 0: letter(ax, "e", dx=-0.06)
    DATA["fig_hier_scale"] = D
    save(fig, "fig_hier_scale")


# --------------------------------------------------------------------------------------------------- figure 2: flaw tolerance (main text)
def cracked_pairs():
    pairs = []
    for reg in (0, 1):
        for cl in (20, 40):
            pairs += [(f"H1_pristine_L{cl}_r{reg}", None, "pristine sheet"), (f"H1_ellipse_L{cl}_r{reg}", f"H1_ellipse_r{reg}", "one level"), (f"H1_slit_L{cl}_r{reg}", f"H1_slit_r{reg}", "one level, slits"),
                      (f"H1_veinsX_L{cl}_r{reg}", f"H1_veinsX_r{reg}", "two levels, solid veins")]
        pairs += [(f"H3_2L_L60_r{reg}", f"H3_2L_r{reg}", "two levels, solid veins"), (f"H3_1L_fine_L60_r{reg}", f"H3_1L_fine_r{reg}", "one level")]
    pairs += [("H3_1L_coarse_L60_r0", "H3_1L_coarse_r0", "one level"), ("H5_VF_E_L60_r0", "H5_VF_E_r0", "fibrous veins"), ("H5_VF_E_L60_r1", "H5_VF_E_r1", "fibrous veins"), ("H5_VF_R_L60_r0", "H5_VF_R_r0", "fibrous veins"),
              ("H5_VS_R_L60_r0", "H5_VS_R_r0", "two levels, solid veins"), ("H5_VE_R_L60_r0", "H5_VE_R_r0", "porous veins"), ("H5_VR_E_L60_r0", "H5_VR_E_r0", "porous veins"), ("H5_V3_E_L60_r0", "H5_V3_E_r0", "three levels"),
              ("H5_GR_0_L40_r0", "H5_GR_0_r0", "grids"), ("H5_GR_0_L40_r1", "H5_GR_0_r1", "grids"), ("H5_GFx_0_L40_r0", "H5_GFx_0_r0", "grids"), ("H5_GS_L40_r0", "H5_GS_r0", "one level, solid bars")]
    rows = []
    for cr, unc, cls in pairs:
        if cr not in own: continue
        s_c = own[cr]["metrics"]["strength_Nm"]; s_u = S0 if unc is None else own[unc]["metrics"]["strength_Nm"]
        W_c = own[cr]["metrics"]["work_to_failure_J_m2"]; W_u = W0 if unc is None else own[unc]["metrics"]["work_to_failure_J_m2"]
        rule = preds[cr]["predicted"]["retention_rule"]; big = own[cr]["descriptors"]["Lx"] > 200
        rows.append(dict(name=cr, uncracked=unc, cls=cls, rule=rule, ret=s_c / s_u, s_cracked=s_c, s_uncracked=s_u, big=big, Wr=W_c / W_u, W_cracked=W_c, W_uncracked=W_u,
                         limited=limited(cr) or (unc is not None and limited(unc)), pristine_reference=unc is None))
    return rows


def fig_hier_flaw():
    FW, FH = 7.2, 8.45
    fig = plt.figure(figsize=(FW, FH)); D = {}
    rows = cracked_pairs()
    ax = axin(fig, 0.5, 0.25, 2.75, 2.2, FW, FH); style(ax); letter(ax, "a")
    for r in rows:
        ax.scatter([r["rule"]], [r["ret"]], s=24, facecolors=COL[r["cls"]] if r["big"] else "white", edgecolors=COL[r["cls"]], linewidths=1.1 if not r["big"] else 0.4, zorder=3)
    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color=DARK, lw=0.8)
    ax.fill_between([0.5, 1.0], [0.5, 1.0], [0.3, 0.3], color="0.94", zorder=0); ax.text(0.945, 0.375, "notch-sensitive", fontsize=6.5, ha="right", color=GREY)
    ax.set_xlim(0.6, 0.97); ax.set_ylim(0.35, 1.0); ax.set_xlabel("net-section rule: section left by the crack", labelpad=1); ax.set_ylabel("strength with crack / strength without", labelpad=1)
    hs = [Line2D([], [], marker="o", markerfacecolor=COL[c], markeredgecolor=COL[c], lw=0, ms=4.5) for c in COL] + [Line2D([], [], marker="o", markerfacecolor=DARK, markeredgecolor=DARK, lw=0, ms=4.5), Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=DARK, lw=0, ms=4.5)]
    fig.legend(handles=hs, labels=list(COL) + ["filled: 300 Å cell", "open: 120 Å cell"], frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(0.45 / FW, 1 - 2.95 / FH), ncol=3, handletextpad=0.3, columnspacing=1.0)
    D["retention"] = [{k: r[k] for k in ("name", "uncracked", "cls", "rule", "ret", "s_cracked", "s_uncracked", "big", "Wr", "W_cracked", "W_uncracked", "limited")} for r in rows]
    NUM["NCracked"] = len(rows)
    # (b, c) cracked vs intact curves
    D["curves"] = {}
    for j, (x, title, items, xl) in enumerate(((3.75, "300 Å, 60 Å crack", [("H3_2L_r0", "H3_2L_L60_r0", RED, "two levels"), ("H3_1L_fine_r0", "H3_1L_fine_L60_r0", BLUE, "one level, fine")], 0.25),
                                              (5.7, "120 Å, 20 Å crack", [("H1_veinsX_r0", "H1_veinsX_L20_r0", RED, "two levels"), ("H1_ellipse_r0", "H1_ellipse_L20_r0", BLUE, "one level, fine")], 0.24))):
        ax = axin(fig, x, 0.25, 1.4, 2.2, FW, FH); style(ax); letter(ax, "bc"[j])
        for unc, cr, col, lab in items:
            for nm, ls in ((unc, "-"), (cr, "--")):
                e, s = curve(nm); ax.plot(e, s, ls, color=col, lw=1.0); D["curves"][nm] = dict(strain=[float(v) for v in e], stress_Nm=[float(v) for v in s])
        ax.set_xlim(0, xl); ax.set_ylim(0, 21); ax.set_xlabel("engineering strain", labelpad=1); ax.set_title(title, fontsize=7.5, pad=2)
        if j == 0: ax.set_ylabel("2D stress (N/m)", labelpad=1)
        if j == 1:
            fig.legend(handles=[Line2D([], [], color=RED, lw=1.0), Line2D([], [], color=RED, lw=1.0, ls="--"), Line2D([], [], color=BLUE, lw=1.0), Line2D([], [], color=BLUE, lw=1.0, ls="--")],
                       labels=["two levels, intact", "two levels, cracked", "one level (fine), intact", "one level (fine), cracked"], frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(4.55 / FW, 1 - 2.95 / FH), ncol=1, handlelength=1.8)
    # (d) crack paths
    y0 = 3.95; H = 1.12
    panels = [("H1_veinsX_L20_r0", frame_for(traj("H1_veinsX_L20_r0"), "peak_load"), "two levels, peak"), ("H1_ellipse_L20_r0", frame_for(traj("H1_ellipse_L20_r0"), "peak_load"), "one level, peak"),
              ("H1_slit_L20_r0", first_drop_frame(traj("H1_slit_L20_r0")), "slits, after the first drop"), ("H3_2L_L60_r0", frame_for(traj("H3_2L_L60_r0"), "peak_load"), "two levels, peak"),
              ("H3_2L_L60_r0", frame_for(traj("H3_2L_L60_r0"), "post_peak"), "two levels, after the peak"), ("H3_1L_fine_L60_r0", frame_for(traj("H3_1L_fine_L60_r0"), "post_peak"), "one level, after the peak")]
    D["snapshots"] = []
    for k, (n, f, lab) in enumerate(panels):
        ax = axin(fig, 0.3 + k * 1.16, y0, H, H, FW, FH)
        e, s, nb = snap(ax, n, f)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m", dy=-0.04, fs=5.5)
        D["snapshots"].append(dict(run=n, frame=int(f), label=lab, strain=e, stress_Nm=s, bonds_broken=nb))
        if k == 0: letter(ax, "d", dx=-0.08)
    fig.text(0.3 / FW, 1 - (y0 - 0.06) / FH, "120 Å cells, 20 Å crack", fontsize=6.5, color=GREY, va="bottom"); fig.text((0.3 + 3 * 1.16) / FW, 1 - (y0 - 0.06) / FH, "300 Å cells, 60 Å crack", fontsize=6.5, color=GREY, va="bottom")
    # (e) work to failure under the crack, by class; open symbols: one run of the pair ended on a loading limit
    ax = axin(fig, 0.5, 6.0, 6.55, 1.55, FW, FH); style(ax); letter(ax, "e")
    order = ["pristine sheet", "one level", "one level, slits", "one level, solid bars", "porous veins", "grids", "two levels, solid veins", "three levels", "fibrous veins"]
    groups = {}
    for r in rows: groups.setdefault(r["cls"], []).append(r)
    D["W_ratio_by_class"] = {}
    for i, cls in enumerate(order):
        rr = groups.get(cls, [])
        if not rr: continue
        v = [r["Wr"] for r in rr]; ax.bar(i, np.mean(v), 0.62, color=COL[cls], zorder=3)
        for r in rr:
            ax.scatter([i], [r["Wr"]], s=11, facecolors="white" if r["limited"] else DARK, edgecolors=DARK, linewidths=0.5, zorder=4)
        ax.text(i, max(v) + 0.05, f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=6)
        D["W_ratio_by_class"][cls] = dict(mean=float(np.mean(v)), values=[dict(name=r["name"], Wr=r["Wr"], limited=r["limited"]) for r in rr])
        NUM["WrClass" + "".join(w.capitalize() for w in cls.replace(",", "").split())] = float(np.mean(v))
    ax.axhline(1.0, color=DARK, lw=0.6, ls=":"); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6.4, rotation=18, ha="right"); ax.set_ylim(0, 1.75)
    ax.set_ylabel("work to failure with crack\n/ without", labelpad=1); ax.text(-0.45, 1.03, "no loss", fontsize=6.3, va="bottom", ha="left", color=DARK)
    DATA["fig_hier_flaw"] = D
    save(fig, "fig_hier_flaw")


# --------------------------------------------------------------------------------------------------- SI: design atlases
DESC = {"H1_pristine_L20": "pristine sheet, 20 Å crack", "H1_ellipse": "one level: elongated pores, period 16 Å", "H1_veinsX": "two levels: 12 Å veins / 40 Å,\nelongated pores (period 12 Å)",
        "H1_slit": "one level: load-parallel slits\n(25 × 4 Å; 30 × 16 Å lattice)", "H4_veinsX_W16": "two levels: 16 Å veins / 40 Å,\nround pores", "H4_veinsX_W20": "two levels: 20 Å veins / 40 Å,\nround pores",
        "H2_seamX_s20": "20 Å crack; pore rows along the\nload every 20 Å (pitch 8 Å)", "H2_seamX_s30": "20 Å crack; pore rows along the\nload every 30 Å", "H2_seamX_s40": "20 Å crack; pore rows along the\nload every 40 Å",
        "H2_weakseamX_s20": "20 Å crack; rows every 20 Å,\npitch 6.5 Å (weaker seam)", "H2_seamY_s20": "control: 20 Å crack; pore rows\nacross the load", "H2_random_s20": "control: 20 Å crack; the same\npores placed at random",
        "H3_1L_fine": "one level: elongated pores,\nperiod 12 Å", "H3_1L_coarse": "one level: 44 Å square holes\n/ 100 Å (56 Å bars)", "H3_2L": "two levels: 30 Å veins / 100 Å,\nelongated pores (period 12 Å)",
        "H5_VF_E": "fibrous veins (slit bundles)\n+ elongated pores", "H5_VF_R": "fibrous veins + round pores", "H5_VE_R": "veins of aligned-pore mesh\n+ round pores", "H5_VR_E": "veins of round-pore mesh\n+ elongated pores",
        "H5_VS_R": "solid veins + round pores", "H5_GR_0": "square grid whose bars are\na round-pore mesh", "H5_GFx_0": "fiber-bar frame: slit-bundle\nbars along the load", "H5_V3_E": "three levels: veins / 100 Å,\n8 Å sub-veins / 33 Å, elongated pores",
        "H5_GS": "solid square grid, 65 Å bars\n(porosity 0.12, not mass-matched)"}
TEST = {"H1": "T1", "H2": "T2", "H3": "T3", "H4": "T4", "H5": "T5"}


SHORT = {"H1_pristine": "pristine sheet", "H1_ellipse": "one level, elongated pores", "H1_veinsX": "two levels, solid veins", "H1_slit": "one level, slits", "H3_2L": "two levels, solid veins", "H3_1L_fine": "one level, fine pores",
         "H3_1L_coarse": "one level, coarse bars", "H5_VS_R": "solid veins + round pores", "H5_VF_E": "fibrous veins + elongated pores", "H5_VF_R": "fibrous veins + round pores", "H5_VE_R": "aligned-mesh veins + round pores",
         "H5_VR_E": "round-mesh veins + elongated pores", "H5_V3_E": "three levels", "H5_GR_0": "mesh-of-meshes grid", "H5_GFx_0": "fiber-bar frame", "H5_GS": "solid-bar grid"}


def atlas(names, fname, ncol, cracked=False):
    nrow = int(np.ceil(len(names) / ncol)); W = 1.02; gap = 0.16; lab = 0.58 if cracked else 0.5
    FW = 0.12 + ncol * (W + gap); FH = 0.1 + nrow * (W + lab)
    fig = plt.figure(figsize=(FW, FH)); D = []
    for k, base in enumerate(names):
        i, j = divmod(k, ncol); ax = axin(fig, 0.1 + j * (W + gap), 0.05 + i * (W + lab), W, W, FW, FH)
        n0 = base + "_r0"; snap(ax, n0, 0, bands=False)
        rr = reg_names(base); s = [own[n]["metrics"]["strength_Nm"] for n in rr]; Wv = [own[n]["metrics"]["work_to_failure_J_m2"] for n in rr]
        L = own[n0]["descriptors"]["Lx"]; phi = own[n0]["descriptors"]["porosity"]; N = own[n0]["descriptors"]["n_atoms"]
        if cracked:
            key = base.split("_L")[0]; cl = base.split("_L")[-1]; desc = SHORT[key]
            rr_c = [x for x in cracked_pairs() if x["name"][:-3] == base]
            line3 = f"{cl} Å crack; {L:.0f} Å, φ = {phi:.2f}\nσ = {'/'.join(f'{x["s_cracked"]:.1f}' for x in rr_c)} N/m\nretention {'/'.join(f'{x["ret"]:.2f}' for x in rr_c)} (rule {'/'.join(f'{x["rule"]:.2f}' for x in rr_c)})"
        else:
            key = "H1_pristine_L20" if base.startswith("H1_pristine") else base; desc = DESC.get(key, key)
            line3 = f"{L:.0f} Å, {N:,} atoms, φ = {phi:.2f}\nσ = {'/'.join(f'{v:.1f}' for v in s)} N/m, W = {'/'.join(f'{v:.2f}' for v in Wv)} J/m²"
        under(ax, f"{base}\n{desc}\n{line3}", dy=-0.03, fs=4.7)
        D.append(dict(design=base, runs=rr, strength_Nm=s, W_J_m2=Wv, cell_A=L, porosity=phi, n_atoms=N, description=desc.replace("\n", " ")))
    DATA[fname] = D
    save(fig, fname, si=True)


def figS_hier_designs():
    atlas(["H1_pristine_L20", "H1_ellipse", "H1_veinsX", "H1_slit", "H4_veinsX_W16", "H4_veinsX_W20", "H2_seamX_s20", "H2_seamX_s30", "H2_seamX_s40", "H2_weakseamX_s20", "H2_seamY_s20", "H2_random_s20",
           "H3_1L_fine", "H3_1L_coarse", "H3_2L", "H5_VS_R", "H5_VF_E", "H5_VF_R", "H5_VE_R", "H5_VR_E", "H5_V3_E", "H5_GR_0", "H5_GFx_0", "H5_GS"], "figS_hier_designs", 6)


def figS_hier_cracked():
    atlas(["H1_pristine_L20", "H1_pristine_L40", "H1_ellipse_L20", "H1_ellipse_L40", "H1_veinsX_L20", "H1_veinsX_L40", "H1_slit_L20", "H1_slit_L40", "H3_2L_L60", "H3_1L_fine_L60", "H3_1L_coarse_L60",
           "H5_VS_R_L60", "H5_VF_E_L60", "H5_VF_R_L60", "H5_VE_R_L60", "H5_VR_E_L60", "H5_V3_E_L60", "H5_GR_0_L40", "H5_GFx_0_L40", "H5_GS_L40"], "figS_hier_cracked", 5, cracked=True)


# --------------------------------------------------------------------------------------------------- SI: the column rule
def cls_of(name):
    b = name.split("_r")[0].split("_L")[0] if not name.startswith("H2") else name
    if b.startswith("H1_pristine") or b.startswith("H2"): return "pristine sheet"
    if b in ("H1_ellipse", "H3_1L_fine", "H3_1L_coarse"): return "one level"
    if b == "H1_slit": return "one level, slits"
    if b in ("H1_veinsX", "H4_veinsX_W16", "H4_veinsX_W20", "H3_2L", "H5_VS_R"): return "two levels, solid veins"
    if b in ("H5_VF_E", "H5_VF_R"): return "fibrous veins"
    if b in ("H5_VE_R", "H5_VR_E"): return "porous veins"
    if b == "H5_V3_E": return "three levels"
    if b in ("H5_GR_0", "H5_GFx_0"): return "grids"
    if b == "H5_GS": return "one level, solid bars"
    return "one level"


def figS_hier_rule():
    FW, FH = 7.2, 2.55
    fig = plt.figure(figsize=(FW, FH)); D = {"uncracked": [], "cracked": []}
    # (a) uncracked designs: observed strength against the column class rule
    ax = axin(fig, 0.45, 0.2, 2.0, 2.0, FW, FH); style(ax); letter(ax, "a")
    errs = []
    for n, r in sorted(own.items()):
        if "_L" in n or n.startswith("H2") or n.startswith("test"): continue
        p = preds[n]["predicted"]; rule = p.get("strength_Nm_class_rule", p.get("strength_Nm_rule")) if n.startswith("H1") else p.get("strength_Nm_rule")
        if n.startswith("H1"): rule = p.get("strength_Nm_class_rule")
        obs = r["metrics"]["strength_Nm"]; big = r["descriptors"]["Lx"] > 200; cls = cls_of(n)
        ax.scatter([rule], [obs], s=22, facecolors=COL[cls] if big else "white", edgecolors=COL[cls], linewidths=1.0 if not big else 0.4, zorder=3)
        errs.append(abs(obs - rule) / obs); D["uncracked"].append(dict(name=n, rule_Nm=rule, observed_Nm=obs, cls=cls, cell_A=float(r["descriptors"]["Lx"])))
    ax.plot([8, 28], [8, 28], "-", color=DARK, lw=0.7); ax.set_xlim(8, 28); ax.set_ylim(8, 28); ax.set_xlabel("column class rule (N/m)", labelpad=1); ax.set_ylabel("simulated strength (N/m)", labelpad=1)
    NUM["RuleMedErrUncracked"] = 100 * float(np.median(errs)); NUM["RuleMaxErrUncracked"] = 100 * float(np.max(errs)); NUM["NRuleUncracked"] = len(errs)
    # (b) the three 300 A designs: simulated, column rule and phase-1 alignment trend
    ax = axin(fig, 2.95, 0.2, 1.75, 2.0, FW, FH); style(ax); letter(ax, "b"); D["T3"] = []
    for i, (key, lab, col) in enumerate((("H3_1L_fine", "one level,\nfine", BLUE), ("H3_1L_coarse", "one level,\ncoarse", GREEN), ("H3_2L", "two\nlevels", RED))):
        p = preds[key + "_r0"]["predicted"]; obs = [own[n]["metrics"]["strength_Nm"] for n in reg_names(key)]
        ax.bar(i - 0.27, p["strength_Nm_rule"], 0.25, color="0.75", zorder=3); ax.bar(i, np.mean(obs), 0.25, color=col, zorder=3)
        ax.bar(i + 0.27, p["strength_Nm_trend"], 0.25, facecolor="white", edgecolor=col, hatch="////", lw=0.6, zorder=3)
        for x, v in ((i - 0.27, p["strength_Nm_rule"]), (i, np.mean(obs)), (i + 0.27, p["strength_Nm_trend"])): ax.text(x, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=5.4)
        D["T3"].append(dict(design=key, rule=p["strength_Nm_rule"], trend=p["strength_Nm_trend"], A=p["alignment_A"], observed=obs))
        pre = {"H3_1L_fine": "Fine", "H3_1L_coarse": "Coarse", "H3_2L": "TwoL"}[key]; NUM[f"Trend{pre}"] = p["strength_Nm_trend"]; NUM[f"Apred{pre}"] = p["alignment_A"]
    ax.set_xticks(range(3)); ax.set_xticklabels(["one level,\nfine", "one level,\ncoarse", "two\nlevels"], fontsize=6.3); ax.set_ylim(0, 26); ax.set_ylabel("strength (N/m)", labelpad=1)
    ax.legend(handles=[Line2D([], [], color="0.75", lw=5), Line2D([], [], color=DARK, lw=5), Line2D([], [], color="white", lw=0, marker="s", markeredgecolor=DARK, markersize=6, markerfacecolor="white")],
              labels=["column rule", "simulated", "alignment trend"], frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    # (c) cracked designs: observed cracked strength against net-section scaling of the intact strength
    ax = axin(fig, 5.2, 0.2, 1.85, 2.0, FW, FH); style(ax); letter(ax, "c")
    errs = []
    for r in cracked_pairs():
        pred = r["s_uncracked"] * r["rule"]; cls = r["cls"]
        ax.scatter([pred], [r["s_cracked"]], s=22, facecolors=COL[cls] if r["big"] else "white", edgecolors=COL[cls], linewidths=1.0 if not r["big"] else 0.4, zorder=3)
        errs.append(abs(r["s_cracked"] - pred) / r["s_cracked"]); D["cracked"].append(dict(name=r["name"], predicted_Nm=pred, observed_Nm=r["s_cracked"], cls=cls))
    ax.plot([6, 34], [6, 34], "-", color=DARK, lw=0.7); ax.set_xlim(6, 34); ax.set_ylim(6, 34); ax.set_xlabel("intact strength × section left (N/m)", labelpad=1); ax.set_ylabel("simulated cracked strength (N/m)", labelpad=1)
    NUM["RuleMedErrCracked"] = 100 * float(np.median(errs)); NUM["NRuleCracked"] = len(errs)
    pr = [e for e, r in zip(errs, cracked_pairs()) if r["cls"] == "pristine sheet"]; NUM["RuleErrPristineMin"] = 100 * float(min(pr)); NUM["RuleErrPristineMax"] = 100 * float(max(pr))
    npr = [e for e, r in zip(errs, cracked_pairs()) if r["cls"] != "pristine sheet"]; NUM["RuleMedErrCrackedPorous"] = 100 * float(np.median(npr))
    hs = [Line2D([], [], marker="o", markerfacecolor=COL[c], markeredgecolor=COL[c], lw=0, ms=4.2) for c in COL] + [Line2D([], [], marker="o", markerfacecolor=DARK, markeredgecolor=DARK, lw=0, ms=4.2), Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=DARK, lw=0, ms=4.2)]
    fig.legend(handles=hs, labels=list(COL) + ["filled: 300 Å cell", "open: 120 Å cell"], frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.4 / FW, 1 - 2.45 / FH), ncol=6, handletextpad=0.3, columnspacing=0.9)
    DATA["figS_hier_rule"] = D
    save(fig, "figS_hier_rule", si=True)


# --------------------------------------------------------------------------------------------------- SI: pre-registered negatives (T4, T2)
def figS_hier_negatives():
    FW, FH = 7.2, 5.0
    fig = plt.figure(figsize=(FW, FH)); D = {"T4_curves": {}, "T2_curves": {}}
    t4 = [x for x in rows_T4(own, ref, preds) if "(phase 1)" not in x["name"]]; t2 = rows_T2(own, ref, preds)
    # (a) T4 curves
    ax = axin(fig, 0.5, 0.25, 2.55, 1.85, FW, FH); style(ax); letter(ax, "a")
    for base, col, lab in (("H1_ellipse", GREY, "one level (control)"), ("H4_veinsX_W16", BLUE, "16 Å veins"), ("H4_veinsX_W20", RED, "20 Å veins")):
        for reg, ls in ((0, "-"), (1, ":")):
            e, s = curve(f"{base}_r{reg}"); ax.plot(e, s, ls, color=col, lw=1.0, label=lab if reg == 0 else None); D["T4_curves"][f"{base}_r{reg}"] = dict(strain=[float(v) for v in e], stress_Nm=[float(v) for v in s])
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 21); ax.set_xlabel("engineering strain", labelpad=1); ax.set_ylabel("2D stress (N/m)", labelpad=1)
    ax.legend(frameon=False, fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.5, columnspacing=1.0, handletextpad=0.5)
    # (b) T4 residual after the first avalanche; (c) post-peak energy
    ax = axin(fig, 3.55, 0.25, 1.55, 1.85, FW, FH); style(ax); letter(ax, "b"); D["T4_metrics"] = []
    for i, Wv in enumerate((16, 20)):
        rr = [x for x in t4 if x["W_vein"] == Wv]
        for j, x in enumerate(rr):
            ax.bar(i + (j - 0.5) * 0.3, x["residual_after_first_avalanche"], 0.28, color=BLUE if Wv == 16 else RED, alpha=1.0 if j == 0 else 0.55, zorder=3)
            ax.text(i + (j - 0.5) * 0.3, max(x["residual_after_first_avalanche"], x["residual_rule"]) + 0.02, f"{x['residual_after_first_avalanche']:.2f}", ha="center", va="bottom", fontsize=5.6)
            D["T4_metrics"].append({k: x.get(k) for k in ("name", "W_vein", "registry", "sigma", "sigma_rule", "residual_after_first_avalanche", "residual_rule", "post_peak_energy_J_m2", "post_avalanche_strain_range", "control_post_peak_energy")})
        ax.plot([i - 0.42, i + 0.42], [rr[0]["residual_rule"]] * 2, "-", color=DARK, lw=1.0, zorder=4)
    ax.axhline(0.6, color=DARK, lw=0.6, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["16 Å veins", "20 Å veins"], fontsize=6.5); ax.set_ylim(0, 1.0); ax.set_ylabel("load after the first avalanche / peak", labelpad=1)
    ax = axin(fig, 5.55, 0.25, 1.5, 1.85, FW, FH); style(ax); letter(ax, "c")
    ctrl = t4[0]["control_post_peak_energy"]
    for i, Wv in enumerate((16, 20)):
        rr = [x for x in t4 if x["W_vein"] == Wv]
        for j, x in enumerate(rr):
            ax.bar(i + (j - 0.5) * 0.3, x["post_peak_energy_J_m2"], 0.28, color=BLUE if Wv == 16 else RED, alpha=1.0 if j == 0 else 0.55, zorder=3)
            ax.text(i + (j - 0.5) * 0.3, x["post_peak_energy_J_m2"] + 0.005, f"{x['post_peak_energy_J_m2']:.2f}", ha="center", va="bottom", fontsize=5.6)
    ax.bar(2, ctrl, 0.28, color=GREY, zorder=3); ax.text(2, ctrl + 0.005, f"{ctrl:.2f}", ha="center", va="bottom", fontsize=5.6)
    ax.axhline(1.5 * ctrl, color=DARK, lw=0.6, ls=":")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["16 Å", "20 Å", "control"], fontsize=6.5); ax.set_ylim(0, 0.45); ax.set_ylabel("energy absorbed after the peak (J/m²)", labelpad=1)
    # (d) T2 curves (registry 0)
    ax = axin(fig, 0.5, 2.75, 3.0, 1.85, FW, FH); style(ax); letter(ax, "d")
    for base, col, ls, lab in (("H1_pristine_L20", DARK, "-", "cracked sheet, no seams"), ("H2_seamX_s20", RED, "-", "rows every 20 Å"), ("H2_seamX_s30", ORANGE, "-", "rows every 30 Å"), ("H2_seamX_s40", GREEN, "-", "rows every 40 Å"),
                               ("H2_weakseamX_s20", PURPLE, "-", "weaker rows every 20 Å"), ("H2_seamY_s20", BLUE, "--", "rows across the load"), ("H2_random_s20", GREY, "--", "random pores")):
        e, s = curve(base + "_r0"); ax.plot(e, s, ls, color=col, lw=1.0, label=lab); D["T2_curves"][base + "_r0"] = dict(strain=[float(v) for v in e], stress_Nm=[float(v) for v in s])
    ax.set_xlim(0, 0.25); ax.set_ylim(0, 34); ax.set_xlabel("engineering strain", labelpad=1); ax.set_ylabel("2D stress (N/m)", labelpad=1)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=2, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    # (e) T2 work-to-failure ratios
    ax = axin(fig, 4.0, 2.75, 3.05, 1.85, FW, FH); style(ax); letter(ax, "e"); D["T2_ratios"] = []
    labels = [("seamX", 20, "rows\nevery 20 Å"), ("seamX", 30, "rows\nevery 30 Å"), ("seamX", 40, "rows\nevery 40 Å"), ("weakseamX", 20, "weaker rows\nevery 20 Å"), ("seamY", 20, "rows across\n(control)"), ("random", 20, "random\n(control)")]
    for i, (key, sp, lab) in enumerate(labels):
        rr = [x for x in t2 if x["key"] == key and x["spacing"] == sp]
        for j, x in enumerate(rr):
            ax.bar(i + (j - 0.5) * 0.3, x["W_ratio"], 0.28, color=RED if key in ("seamX", "weakseamX") else BLUE, alpha=1.0 if j == 0 else 0.55, zorder=3)
            ax.text(i + (j - 0.5) * 0.3, max(x["W_ratio"], x.get("W_ratio_rule") or 0) + 0.02 + 0.07 * j, f"{x['W_ratio']:.2f}", ha="center", va="bottom", fontsize=5.4)
            D["T2_ratios"].append({k: x.get(k) for k in ("name", "key", "spacing", "registry", "sigma", "sigma_ref", "sigma_ratio", "W", "W_ref", "W_ratio", "W_ratio_rule", "W_ratio_hypothesis", "deflected")})
        if rr[0].get("W_ratio_rule"): ax.plot([i - 0.42, i + 0.42], [rr[0]["W_ratio_rule"]] * 2, "-", color=DARK, lw=1.0, zorder=4)
    ax.axhline(1.3, color=DARK, lw=0.6, ls=":")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([l for _, _, l in labels], fontsize=5.8); ax.set_ylim(0, 1.5); ax.set_ylabel("work to failure / cracked sheet", labelpad=1)
    DATA["figS_hier_negatives"] = D
    save(fig, "figS_hier_negatives", si=True)


# --------------------------------------------------------------------------------------------------- SI: building blocks
def figS_hier_blocks():
    FW, FH = 7.2, 5.5
    fig = plt.figure(figsize=(FW, FH)); D = {"curves": {}, "post_peak_energy": {}}
    items = [("H3_1L_fine", BLUE, "one level, fine"), ("H3_1L_coarse", GREEN, "one level, coarse"), ("H3_2L", RED, "two levels, solid veins"), ("H5_VF_E", PURPLE, "two levels, fibrous veins"), ("H5_V3_E", BROWN, "three levels"), ("H5_GR_0", CYAN, "mesh of meshes")]
    ax = axin(fig, 0.5, 0.25, 2.9, 2.15, FW, FH); style(ax); letter(ax, "a")
    for n, col, lab in items:
        for reg, ls in ((0, "-"), (1, ":")):
            if f"{n}_r{reg}" not in own: continue
            e, s = curve(f"{n}_r{reg}"); ip = int(np.argmax(s))
            ax.plot(e[ip:] - e[ip], s[ip:] / s[ip], ls, color=col, lw=1.0, label=lab if reg == 0 else None)
            D["curves"][f"{n}_r{reg}"] = dict(strain_beyond_peak=[float(v) for v in (e[ip:] - e[ip])], stress_over_peak=[float(v) for v in (s[ip:] / s[ip])], limited=limited(f"{n}_r{reg}"))
    ax.set_xlim(0, 0.2); ax.set_ylim(0, 1.05); ax.set_xlabel("strain beyond the peak", labelpad=1); ax.set_ylabel("stress / peak stress", labelpad=1)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=1.6)
    ax = axin(fig, 3.95, 0.25, 3.1, 2.15, FW, FH); style(ax); letter(ax, "b")
    for i, (n, col, lab) in enumerate(items):
        rr = reg_names(n); vals = [cm(x)["post_peak_energy_J_m2"] for x in rr]
        ax.bar(i, np.mean(vals), 0.62, color=col, zorder=3)
        for x, v in zip(rr, vals): ax.scatter([i], [v], s=11, facecolors="white" if limited(x) else DARK, edgecolors=DARK, linewidths=0.5, zorder=4)
        ax.text(i, max(vals) + 0.03, f"{np.mean(vals):.2f}", ha="center", va="bottom", fontsize=6)
        D["post_peak_energy"][n] = dict(values=vals, runs=rr, limited=[limited(x) for x in rr])
    ax.set_xticks(range(len(items))); ax.set_xticklabels([lab.replace(", ", ",\n").replace("three levels", "three\nlevels").replace("mesh of meshes", "mesh of\nmeshes") for _, _, lab in items], fontsize=6.2)
    ax.set_ylabel("energy absorbed after the peak (J/m²)", labelpad=1); ax.set_ylim(0, 1.45)
    y0 = 3.05; H = 1.55; W = 2.1
    panels = [("H3_2L_r0", frame_for(traj("H3_2L_r0"), "major_propagation"), (60, 210, 80, 190), "solid vein"), ("H5_VF_E_r0", first_drop_frame(traj("H5_VF_E_r0")), (60, 210, 80, 190), "fiber-bundle vein"),
              ("H5_V3_E_r0", frame_for(traj("H5_V3_E_r0"), "post_peak"), "auto", "three levels, sub-veins")]
    D["snapshots"] = []
    for k, (n, f, crop, lab) in enumerate(panels):
        if crop == "auto":
            T = traj(n); c = T["cells"][f]; dmg = PF.damaged_atoms_until(own[n], T["eps_x"][f]); Pd = np.mod(T["positions"][f][dmg, :2].astype(float), [c[0, 0], c[1, 1]])
            best = None
            for x0 in np.arange(0, c[0, 0] - 150, 10):
                for y0_ in np.arange(0, c[1, 1] - 110, 10):
                    m = int(((Pd[:, 0] > x0) & (Pd[:, 0] < x0 + 150) & (Pd[:, 1] > y0_) & (Pd[:, 1] < y0_ + 110)).sum())
                    if best is None or m > best[0]: best = (m, x0, y0_)
            crop = (best[1], best[1] + 150, best[2], best[2] + 110)
        ax = axin(fig, 0.3 + k * 2.3, y0, W, H, FW, FH)
        e, s, nb = snap(ax, n, f, crop=crop, s=0.5, sdmg=2.2, lw=0.18)
        under(ax, f"{lab}\nε = {e:.3f}, σ = {s:.1f} N/m", dy=-0.04, fs=6.0)
        D["snapshots"].append(dict(run=n, frame=int(f), window_A=[float(v) for v in crop], label=lab, strain=e, stress_Nm=s, bonds_broken=nb))
        if k == 0: letter(ax, "c", dx=-0.05)
    DATA["figS_hier_blocks"] = D
    save(fig, "figS_hier_blocks", si=True)


# --------------------------------------------------------------------------------------------------- SI: fields
def field(ax, name, k, mode, s=None, dmg=True):
    """Per-atom field maps (rasterized atoms; frames and colorbars stay vector)."""
    T = traj(name); rec = OWN[name]; P = T["positions"][k].astype(float); c = T["cells"][k]
    for a in range(2): P[:, a] = np.mod(P[:, a], c[a, a])
    if mode == "conc":
        val = PF.peratom_stress(T, k) / (T["sigma_xx"][k] * EV); cmap, vmin, vmx = "inferno", 0.0, 3.0
    elif mode == "bondstrain":
        val = PF.bond_strain(T, k); cmap, vmin, vmx = "plasma", 0.0, 0.25
    else:
        val = T["peratom_energy"][k].astype(float) - T["peratom_energy"][0].astype(float); cmap, vmin, vmx = "magma", 0.0, 1.5
    big = c[0, 0] > 200; s = s or (0.22 if big else 1.3)
    sc = ax.scatter(P[:, 0], P[:, 1], c=val, s=s, cmap=cmap, vmin=vmin, vmax=vmx, linewidths=0, zorder=2, rasterized=True)
    if dmg and mode == "energy":
        d = PF.damaged_atoms_until(rec, T["eps_x"][k])
        if len(d): ax.scatter(P[d, 0], P[d, 1], s=s * 3.0, facecolors="none", edgecolors=RED, linewidths=0.3, zorder=3, rasterized=True)
    ax.set_xlim(0, c[0, 0]); ax.set_ylim(0, c[1, 1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.55"); sp.set_linewidth(0.5)
    return sc, float(T["eps_x"][k]), float(T["sigma_xx"][k] * EV), int(T["n_broken_cum"][k])


def figS_hier_fields():
    FW, FH = 7.2, 8.1
    fig = plt.figure(figsize=(FW, FH)); D = []
    rows = [("H3_2L_L60_r0", "300 Å, two levels\n+ 60 Å crack"), ("H3_1L_fine_L60_r0", "300 Å, one level (fine)\n+ 60 Å crack"), ("H1_veinsX_L20_r0", "120 Å, two levels\n+ 20 Å crack"), ("H1_ellipse_L20_r0", "120 Å, one level (fine)\n+ 20 Å crack")]
    cols = [("conc", "stress concentration at ε = 0.08"), ("bondstrain", "bond strain at the peak"), ("energy", "damage after the first drop"), ("energy", "damage at the end")]
    cb_label = {"conc": "$\\sigma_1$ / $\\sigma_\\mathrm{applied}$", "bondstrain": "bond strain ($r$ − $r_0$) / $r_0$", "energy": "energy change per atom (eV)"}
    H = 1.5; x0 = 0.95; y0 = 0.3; scs = {}
    for i, (n, lab) in enumerate(rows):
        T = traj(n); ip = int(np.argmax(T["sigma_xx"])); k8 = int(np.searchsorted(T["eps_x"], 0.08)); kd = first_drop_frame(T); kf = len(T["eps_x"]) - 1
        for j, ((mode, title), k) in enumerate(zip(cols, (k8, ip, kd, kf))):
            ax = axin(fig, x0 + j * (H + 0.08), y0 + i * (H + 0.36), H, H, FW, FH)
            sc, e, sig, nb = field(ax, n, k, mode); scs[mode] = sc
            under(ax, f"ε = {e:.3f}, σ = {sig:.1f} N/m" + ("" if mode != "energy" else f", {nb} bonds broken"), dy=-0.03, fs=5.8)
            D.append(dict(run=n, frame=int(k), field=mode, strain=e, stress_Nm=sig, bonds_broken=nb))
            if i == 0: ax.set_title(title, fontsize=6.6, pad=3)
            if j == 0:
                ax.text(-0.06, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=6.6, linespacing=1.2); letter(ax, "abcd"[i], dx=-0.62, dy=0.9)
    ycb = y0 + 4 * (H + 0.36) - 0.05
    for j, (mode, _) in enumerate(cols[:3]):
        cax = axin(fig, x0 + j * (H + 0.08) + 0.1, ycb, H - 0.2, 0.07, FW, FH); cb = fig.colorbar(scs[mode], cax=cax, orientation="horizontal")
        cb.set_label(cb_label[mode], fontsize=6.0, labelpad=1); cb.ax.tick_params(labelsize=5.6, length=2)
    DATA["figS_hier_fields"] = D
    save(fig, "figS_hier_fields", si=True)


# --------------------------------------------------------------------------------------------------- numbers, tables
def numbers_and_tables():
    t1 = rows_T1(own, ref, preds); t4 = [x for x in rows_T4(own, ref, preds) if "(phase 1)" not in x["name"]]; t2 = rows_T2(own, ref, preds); t3 = rows_T3(own, ref, preds, R); t5 = rows_T5(own, ref, preds)
    v1, v2, v3, v4, v5 = verdict_T1(t1), verdict_T2(t2), verdict_T3(t3, R), verdict_T4(t4), verdict_T5(t5)
    runs = [r for n, r in own.items() if not n.startswith("test")]
    N = NUM
    N.update(NRuns=len(runs), NRunsA=len([r for r in runs if not r["name"].startswith("H5")]), NRunsB=len([r for r in runs if r["name"].startswith("H5")]),
             NCellSmall=len([r for r in runs if r["descriptors"]["Lx"] < 200]), NCellLarge=len([r for r in runs if r["descriptors"]["Lx"] > 200]),
             AtomsSmallMin=min(r["descriptors"]["n_atoms"] for r in runs if r["descriptors"]["Lx"] < 200), AtomsSmallMax=max(r["descriptors"]["n_atoms"] for r in runs if r["descriptors"]["Lx"] < 200),
             AtomsLargeMin=min(r["descriptors"]["n_atoms"] for r in runs if r["descriptors"]["Lx"] > 200), AtomsLargeMax=max(r["descriptors"]["n_atoms"] for r in runs if r["descriptors"]["Lx"] > 200),
             NOperational=len([r for r in runs if term_code(r["name"]) == "F"]), NPostPeakLimit=len([r for r in runs if term_code(r["name"]) == "L"]), NMaxStrain=len([r for r in runs if term_code(r["name"]) == "M"]),
             NUncracked=len([r for r in runs if "_L" not in r["name"] and not r["name"].startswith("H2")]), NCrackedAll=len([r for r in runs if "_L" in r["name"] or r["name"].startswith("H2")]),
             WallHours=FACTS["wall_h"], MAtoms=FACTS["atoms"] / 1e6, FirstRun=FACTS["first"], LastRun=FACTS["last"], Evaluated=FACTS["evaluated"],
             PredWrittenA=FACTS["prediction_written"], PredShaA=FACTS["prediction_sha256"][:16], PredWrittenB=FACTS["prediction_written_T5"], PredShaB=FACTS["prediction_sha256_T5"][:16],
             RunHoursLargeMin=min(r["wall_time_s"] for r in runs if r["descriptors"]["Lx"] > 200) / 3600, RunHoursLargeMax=max(r["wall_time_s"] for r in runs if r["descriptors"]["Lx"] > 200) / 3600,
             LevelRatioSmall=own["H1_veinsX_r0"]["design"]["scale_ratio"], LevelRatioLarge=own["H3_2L_r0"]["design"]["scale_ratio"],
             OneLevelScatter=R["alignment"]["one_level_resid_std"], OneLevelWScatter=R["alignment"]["one_level_W_std"], PremiumMean=R["alignment"]["premium_mean"], PremiumStd=R["alignment"]["premium_std"],
             ClassStraight=R["classes"]["straight"]["median"], ClassCoarse=R["classes"]["coarse_round"]["median"], ClassFine=R["classes"]["fine_round"]["median"], ClassRandom=R["classes"]["random"]["median"],
             ClassStraightStored=R["classes"]["straight"]["stored_median"], ClassCoarseStored=R["classes"]["coarse_round"]["stored_median"], ClassFineStored=R["classes"]["fine_round"]["stored_median"], ClassRandomStored=R["classes"]["random"]["stored_median"],
             TrendSlope=R["alignment"]["fit_one_level"][0], TrendIntercept=R["alignment"]["fit_one_level"][1])
    v1s = json.load(open(os.path.join(ROOT, "experiments", "predictions", "hierarchy_followup_predictions_v1_superseded.json"))) if os.path.exists(os.path.join(ROOT, "experiments", "predictions", "hierarchy_followup_predictions_v1_superseded.json")) else {}
    N["PredWrittenVone"] = v1s.get("written", "2026-09-23 13:52:58"); N["PredShaVone"] = str(v1s.get("sha256_of_content", "f0a06e605a9ce952"))[:16]
    # strengths and work of the key designs (registry 0 / 1)
    def put(prefix, base, keys=(("Sig", "strength_Nm"), ("W", "work_to_failure_J_m2"), ("Y", "modulus_2d_Nm"), ("EpsF", "failure_strain"))):
        for tag, key in keys:
            for reg, suf in ((0, "A"), (1, "B")):
                n = f"{base}_r{reg}"
                if n in own: N[f"{prefix}{tag}{suf}"] = own[n]["metrics"][key]
        for reg, suf in ((0, "A"), (1, "B")):
            n = f"{base}_r{reg}"
            if n in own:
                c = cm(n); N[f"{prefix}Residual{suf}"] = c["residual_after_first_avalanche"]; N[f"{prefix}PPE{suf}"] = c["post_peak_energy_J_m2"]; N[f"{prefix}PPRange{suf}"] = c["post_peak_strain_range"]
                N[f"{prefix}PostAvRange{suf}"] = c["post_avalanche_strain_range"]; N[f"{prefix}Term{suf}"] = term_code(n); N[f"{prefix}PeakStrain{suf}"] = c["peak_strain"]
    for prefix, base in (("TwoL", "H3_2L"), ("Fine", "H3_1L_fine"), ("Coarse", "H3_1L_coarse"), ("Veins", "H1_veinsX"), ("Ellipse", "H1_ellipse"), ("Slit", "H1_slit"), ("Wsixteen", "H4_veinsX_W16"), ("Wtwenty", "H4_veinsX_W20"),
                         ("VFE", "H5_VF_E"), ("VFR", "H5_VF_R"), ("VER", "H5_VE_R"), ("VRE", "H5_VR_E"), ("VSR", "H5_VS_R"), ("GR", "H5_GR_0"), ("GFx", "H5_GFx_0"), ("Vthree", "H5_V3_E"), ("GS", "H5_GS"),
                         ("TwoLcr", "H3_2L_L60"), ("Finecr", "H3_1L_fine_L60"), ("Coarsecr", "H3_1L_coarse_L60"), ("VeinsLtw", "H1_veinsX_L20"), ("VeinsLfo", "H1_veinsX_L40"), ("EllipseLtw", "H1_ellipse_L20"), ("EllipseLfo", "H1_ellipse_L40"),
                         ("SlitLtw", "H1_slit_L20"), ("SlitLfo", "H1_slit_L40"), ("PristineLtw", "H1_pristine_L20"), ("PristineLfo", "H1_pristine_L40"),
                         ("VFEcr", "H5_VF_E_L60"), ("VFRcr", "H5_VF_R_L60"), ("VERcr", "H5_VE_R_L60"), ("VREcr", "H5_VR_E_L60"), ("VSRcr", "H5_VS_R_L60"), ("GRcr", "H5_GR_0_L40"), ("GFxcr", "H5_GFx_0_L40"), ("Vthreecr", "H5_V3_E_L60"), ("GScr", "H5_GS_L40")):
        put(prefix, base)
    for base, prefix in (("H3_2L", "TwoL"), ("H3_1L_fine", "Fine"), ("H3_1L_coarse", "Coarse"), ("H5_VF_E", "VFE"), ("H5_VF_R", "VFR"), ("H5_VE_R", "VER"), ("H5_VR_E", "VRE"), ("H5_VS_R", "VSR"), ("H5_GR_0", "GR"), ("H5_GFx_0", "GFx"), ("H5_V3_E", "Vthree"), ("H5_GS", "GS"),
                         ("H4_veinsX_W16", "Wsixteen"), ("H4_veinsX_W20", "Wtwenty"), ("H1_veinsX", "Veins"), ("H1_ellipse", "Ellipse"), ("H1_slit", "Slit")):
        p = preds[base + "_r0"]["predicted"]; N[prefix + "Rule"] = p.get("strength_Nm_class_rule") if base.startswith("H1") else p.get("strength_Nm_rule")
        if p.get("strength_Nm_rule_b") is not None: N[prefix + "RuleB"] = p["strength_Nm_rule_b"]
        N[prefix + "Sig"] = float(np.mean([own[n]["metrics"]["strength_Nm"] for n in reg_names(base)])); N[prefix + "W"] = float(np.mean([own[n]["metrics"]["work_to_failure_J_m2"] for n in reg_names(base)]))
        N[prefix + "PPE"] = float(np.mean([cm(n)["post_peak_energy_J_m2"] for n in reg_names(base)]))
    N["WsixteenResidualRule"] = preds["H4_veinsX_W16_r0"]["predicted"]["residual_fraction_rule"]; N["WtwentyResidualRule"] = preds["H4_veinsX_W20_r0"]["predicted"]["residual_fraction_rule"]
    N["TfourControlEnergy"] = t4[0]["control_post_peak_energy"]; N["TfourControlSig"] = t4[0]["control_sigma"]
    # T3 premiums and cracked retention
    for reg, suf in ((0, "A"), (1, "B")):
        two = next(x for x in t3 if x["name"] == f"H3_2L_r{reg}"); N[f"TwoLSigPremFine{suf}"] = two["sigma_premium_vs_fine"]; N[f"TwoLSigPremCoarse{suf}"] = two["sigma_premium_vs_coarse"]
        N[f"TwoLWPremFine{suf}"] = two["W_premium_vs_fine"]; N[f"TwoLWPremCoarse{suf}"] = two["W_premium_vs_coarse"]; N[f"TwoLPremTrend{suf}"] = two["premium_sigma"]
        N[f"TwoLAmeas{suf}"] = two["A_measured"]
        fine = next(x for x in t3 if x["name"] == f"H3_1L_fine_r{reg}"); coarse = next(x for x in t3 if x["name"] == f"H3_1L_coarse_r{reg}")
        N[f"FinePremTrend{suf}"] = fine["premium_sigma"]; N[f"CoarsePremTrend{suf}"] = coarse["premium_sigma"]; N[f"FineAmeas{suf}"] = fine["A_measured"]; N[f"CoarseAmeas{suf}"] = coarse["A_measured"]
    # cracked retention table values (all cracked designs) and T1 details
    for r in cracked_pairs():
        pre = r["name"].replace("H1_", "").replace("H3_", "").replace("H5_", "").replace("_", "")
        key = {"pristineL20r0": "PristineLtwA", "pristineL20r1": "PristineLtwB", "pristineL40r0": "PristineLfoA", "pristineL40r1": "PristineLfoB", "ellipseL20r0": "EllipseLtwA", "ellipseL20r1": "EllipseLtwB", "ellipseL40r0": "EllipseLfoA", "ellipseL40r1": "EllipseLfoB",
               "veinsXL20r0": "VeinsLtwA", "veinsXL20r1": "VeinsLtwB", "veinsXL40r0": "VeinsLfoA", "veinsXL40r1": "VeinsLfoB", "slitL20r0": "SlitLtwA", "slitL20r1": "SlitLtwB", "slitL40r0": "SlitLfoA", "slitL40r1": "SlitLfoB",
               "2LL60r0": "TwoLcrA", "2LL60r1": "TwoLcrB", "1LfineL60r0": "FinecrA", "1LfineL60r1": "FinecrB", "1LcoarseL60r0": "CoarsecrA", "VFEL60r0": "VFEcrA", "VFEL60r1": "VFEcrB", "VFRL60r0": "VFRcrA", "VSRL60r0": "VSRcrA",
               "VERL60r0": "VERcrA", "VREL60r0": "VREcrA", "V3EL60r0": "VthreecrA", "GR0L40r0": "GRcrA", "GR0L40r1": "GRcrB", "GFx0L40r0": "GFxcrA", "GSL40r0": "GScrA"}[pre]
        N[key + "Ret"] = r["ret"]; N[key + "Rule"] = r["rule"]; N[key + "OverRule"] = r["ret"] / r["rule"]; N[key + "Wr"] = r["Wr"]
    for x in t1:
        if x["key"] == "veinsX":
            suf = {(20, 0): "VeinsLtwA", (20, 1): "VeinsLtwB", (40, 0): "VeinsLfoA", (40, 1): "VeinsLfoB"}[(x["crack"], x["registry"])]
            N[suf + "FirstVeinDamage"] = x.get("first_vein_damage_strain"); N[suf + "ArrestedAtPeak"] = "yes" if x.get("arrested_at_peak") else "no"
    for reg, suf in ((0, "A"), (1, "B")):
        cp = crack_path(own[f"H3_2L_r{reg}"]); N[f"TwoLFirstVeinDamage{suf}"] = cp.get("first_vein_damage_strain"); N[f"TwoLFirstDamage{suf}"] = own[f"H3_2L_r{reg}"]["metrics"]["first_damage_strain"]
        cp = crack_path(own[f"H3_2L_L60_r{reg}"]); N[f"TwoLcrArrestedAtPeak{suf}"] = "yes" if cp.get("arrested_at_peak") else "no"; N[f"TwoLcrFirstVeinDamage{suf}"] = cp.get("first_vein_damage_strain")
    A = preds["H1_veinsX_r0"]["predicted"]; N["VeinsApred"] = preds["H1_veinsX_r0"].get("alignment_A"); N["VeinsPremTrend"] = N["VeinsSigA"] - (R["alignment"]["fit_one_level"][0] * N["VeinsApred"] + R["alignment"]["fit_one_level"][1])
    N["VeinsTrend"] = R["alignment"]["fit_one_level"][0] * N["VeinsApred"] + R["alignment"]["fit_one_level"][1]
    # T2 summary
    seams = [x for x in t2 if x["key"] in ("seamX", "weakseamX")]; N["SeamWratioMin"] = min(x["W_ratio"] for x in seams); N["SeamWratioMax"] = max(x["W_ratio"] for x in seams)
    N["SeamWratioRuleMin"] = min(x["W_ratio_rule"] for x in seams); N["SeamWratioRuleMax"] = max(x["W_ratio_rule"] for x in seams); N["SeamSigRatioMin"] = min(x["sigma_ratio"] for x in seams); N["SeamSigRatioMax"] = max(x["sigma_ratio"] for x in seams)
    for x in t2:
        tag = f"Ttwo{x['key']}{ {20: 'Twenty', 30: 'Thirty', 40: 'Forty'}[x['spacing']] }{ 'AB'[x['registry']] }"
        N[tag + "Wr"] = x["W_ratio"]; N[tag + "SigR"] = x["sigma_ratio"]
    # T5 verdict details
    N["FibrousResidualMin"] = min(cm(n)["residual_after_first_avalanche"] for n in ("H5_VF_E_r0", "H5_VF_E_r1", "H5_VF_R_r0", "H5_VF_R_r1")); N["FibrousResidualMax"] = max(cm(n)["residual_after_first_avalanche"] for n in ("H5_VF_E_r0", "H5_VF_E_r1", "H5_VF_R_r0", "H5_VF_R_r1"))
    # prediction-evaluation summaries
    for f, tag in (("hierarchy_followup_evaluation.json", "Eval"), ("hierarchy_T5_evaluation.json", "EvalTfive")):
        s = json.load(open(os.path.join(ROOT, "experiments", "holdouts", f)))["summary"]
        for k, name in (("modulus_2d_Nm", "Modulus"), ("work_to_failure_J_m2", "W"), ("failure_strain", "Strain")):
            if k in s: N[f"{tag}{name}Err"] = 100 * s[k]["median_abs_rel_error"]; N[f"{tag}{name}N"] = s[k]["n"]
        N[f"{tag}ModeAcc"] = 100 * s["fracture_mode_class_accuracy"]
    # verdict strings
    N["VerdictTone"], N["VerdictTtwo"], N["VerdictTthree"], N["VerdictTfour"], N["VerdictTfive"] = v1["overall"], v2["overall"], v3["overall"], v4["overall"], v5["overall"]
    # ---- macros
    def fmt(k, v):
        if isinstance(v, str): return v
        if v is None: return "--"
        if isinstance(v, (int, np.integer)): return f"{v:,}" if abs(v) >= 10000 else str(int(v))
        if k.startswith("N") and float(v).is_integer(): return str(int(v))
        if k.endswith(("Err", "Acc", "Share")): return f"{v:.0f}"
        if k.endswith("Rule") and abs(v) > 3: return f"{v:.1f}"
        if k == "OneLevelWScatter": return f"{v:.2f}"
        if "EpsF" in k: return f"{v:.3f}"
        if "Apred" in k: return f"{v:.2f}"
        if any(k.endswith(s) for s in ("Ret", "Rule", "OverRule", "Wr", "Residual", "ResidualA", "ResidualB", "ResidualRule", "ResidualMin", "ResidualMax")) or k.startswith(("WrClass", "SeamWratio", "SeamSig", "T2_")): return f"{v:.2f}"
        if k.endswith(("PPE", "PPEA", "PPEB", "W", "WA", "WB", "ControlEnergy", "Prem", "Range", "RangeA", "RangeB")) or k.startswith(("PPE_", "TwoLWPrem")): return f"{v:.2f}"
        if "Strain" in k or "Damage" in k or k.startswith("LoadMapStrain") or "Ameas" in k or k.startswith("A_") or k in ("LevelRatioSmall", "LevelRatioLarge"): return f"{v:.3f}" if ("Strain" in k or "Damage" in k) else f"{v:.2f}"
        if k.endswith(("Err", "Acc", "Share")) or k.startswith("RuleErr"): return f"{v:.0f}"
        if k in ("MAtoms",): return f"{v:.2f}"
        if k in ("WallHours",): return f"{v:.0f}"
        return f"{v:.1f}"
    lines = ["%SEPT 26 HIERARCHY EDIT BEGIN: generated numbers of the hierarchy follow-up (analysis/manuscript_figures_phase3.py in hierarchy_followup/); do not edit by hand",
             f"% generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} from {N['NRuns']} completed runs"]
    seen = set()
    for k in sorted(N):
        name = "hier" + "".join(ch for ch in k.replace("_", "") if ch.isalpha())
        assert name not in seen, name; seen.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt(k, N[k])}}}")
    lines.append("%SEPT 26 HIERARCHY EDIT END")
    open(os.path.join(OUT, "hierarchy_numbers.tex"), "w").write("\n".join(lines) + "\n")
    shutil.copy(os.path.join(OUT, "hierarchy_numbers.tex"), os.path.join(PAPER, "figures", "hierarchy_numbers.tex"))
    # ---- tables
    T = []
    # uncracked designs
    rows = []
    for base in ("H1_ellipse", "H1_veinsX", "H1_slit", "H4_veinsX_W16", "H4_veinsX_W20", "H3_1L_fine", "H3_1L_coarse", "H3_2L", "H5_VS_R", "H5_VF_E", "H5_VF_R", "H5_VE_R", "H5_VR_E", "H5_V3_E", "H5_GR_0", "H5_GFx_0", "H5_GS"):
        rr = reg_names(base); m = [own[n]["metrics"] for n in rr]; c = [cm(n) for n in rr]; p = preds[base + "_r0"]["predicted"]
        rule = p.get("strength_Nm_class_rule") if base.startswith("H1") else p.get("strength_Nm_rule")
        rows.append(" & ".join([tex_id(base), TEST[base[:2]], f"{own[rr[0]]['descriptors']['Lx']:.0f}", f"{own[rr[0]]['descriptors']['porosity']:.2f}", "/".join(f"{x['strength_Nm']:.1f}" for x in m), f1(rule), "/".join(f"{x['modulus_2d_Nm']:.0f}" for x in m),
                                "/".join(f"{x['work_to_failure_J_m2']:.2f}" for x in m), "/".join(f"{x['failure_strain']:.3f}" for x in m), "/".join(("--" if x["residual_after_first_avalanche"] is None else f"{x['residual_after_first_avalanche']:.2f}") for x in c),
                                "/".join(f"{x['post_peak_energy_J_m2']:.2f}" for x in c), "/".join(term_code(n) for n in rr)]) + " \\\\")
    head = "\\begin{tabular}{llrrrrrrrrrc}\n\\toprule\ndesign & test & cell (\\AA) & $\\phi$ & $\\sigma$ (\\Nm) & rule (\\Nm) & $Y$ (\\Nm) & $W$ (\\Jm) & $\\varepsilon_\\mathrm{end}$ & residual & $E_\\mathrm{pp}$ (\\Jm) & end \\\\\n\\midrule\n"
    open(os.path.join(TAB_DIR, "si_hier_uncracked.tex"), "w").write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # cracked designs
    rows = []
    for r in cracked_pairs():
        if r["name"].endswith("_r1"): continue
        base = r["name"][:-3]; rr = [x for x in cracked_pairs() if x["name"][:-3] == base]
        crack = base.split("_L")[-1]
        rows.append(" & ".join([tex_id(base), TEST[base[:2]], f"{own[r['name']]['descriptors']['Lx']:.0f}", crack, "/".join(f"{x['s_cracked']:.1f}" for x in rr), "/".join(f"{x['s_uncracked']:.1f}" for x in rr), "/".join(f"{x['ret']:.2f}" for x in rr),
                                "/".join(f"{x['rule']:.2f}" for x in rr), "/".join(f"{x['ret'] / x['rule']:.2f}" for x in rr), "/".join(f"{x['Wr']:.2f}" for x in rr), "/".join(term_code(x["name"]) for x in rr)]) + " \\\\")
    head = "\\begin{tabular}{llrrrrrrrrc}\n\\toprule\ndesign & test & cell (\\AA) & crack (\\AA) & $\\sigma_\\mathrm{crack}$ (\\Nm) & $\\sigma_\\mathrm{intact}$ (\\Nm) & retention & rule & retention/rule & $W$ ratio & end \\\\\n\\midrule\n"
    open(os.path.join(TAB_DIR, "si_hier_cracked.tex"), "w").write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # seams
    rows = []
    for key, sp in (("seamX", 20), ("seamX", 30), ("seamX", 40), ("weakseamX", 20), ("seamY", 20), ("random", 20)):
        rr = [x for x in t2 if x["key"] == key and x["spacing"] == sp]
        rows.append(" & ".join([tex_id(f"H2_{key}_s{sp}"), "/".join(f"{x['sigma']:.1f}" for x in rr), "/".join(f"{x['sigma_ratio']:.2f}" for x in rr), "/".join(f"{x['W']:.2f}" for x in rr), "/".join(f"{x['W_ratio']:.2f}" for x in rr),
                                f2(rr[0].get("W_ratio_rule")), "/".join(("yes" if x.get("deflected") else "no") for x in rr), "/".join(term_code(x["name"]) for x in rr)]) + " \\\\")
    head = "\\begin{tabular}{lrrrrrcc}\n\\toprule\ndesign & $\\sigma$ (\\Nm) & $\\sigma/\\sigma_\\mathrm{ref}$ & $W$ (\\Jm) & $W$ ratio & rule & deflected & end \\\\\n\\midrule\n"
    open(os.path.join(TAB_DIR, "si_hier_seams.tex"), "w").write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    for f in os.listdir(TAB_DIR):
        shutil.copy(os.path.join(TAB_DIR, f), os.path.join(P_MAIN, "tables", f))
    json.dump(dict(T1=v1, T2=v2, T3=v3, T4=v4, T5=v5), open(os.path.join(DATA_DIR, "verdicts.json"), "w"), indent=1, default=str)


def write_data():
    for name, d in DATA.items():
        json.dump(d, open(os.path.join(DATA_DIR, name + ".json"), "w"), indent=0, default=float)
        shutil.copy(os.path.join(DATA_DIR, name + ".json"), os.path.join(P_MAIN, "data", name + ".json"))
    json.dump(NUM, open(os.path.join(DATA_DIR, "numbers.json"), "w"), indent=1, default=float); shutil.copy(os.path.join(DATA_DIR, "numbers.json"), os.path.join(P_MAIN, "data", "numbers.json"))
    json.dump(AUDIT, open(os.path.join(OUT, "audit.json"), "w"), indent=1)


if __name__ == "__main__":
    only = sys.argv[1:]
    todo = {"scale": fig_hier_scale, "flaw": fig_hier_flaw, "designs": figS_hier_designs, "cracked": figS_hier_cracked, "rule": figS_hier_rule, "negatives": figS_hier_negatives, "blocks": figS_hier_blocks, "fields": figS_hier_fields}
    for k, f in todo.items():
        if not only or k in only: f()
    numbers_and_tables(); write_data()
    print("wrote", OUT, "and copied to", PAPER); print("audit:", AUDIT)
