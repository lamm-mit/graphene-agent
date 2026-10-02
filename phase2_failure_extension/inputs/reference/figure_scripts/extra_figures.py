"""Additional manuscript figures and the supplementary material:
  fig0_experiment      workflow of the experiment (one prompt -> instrument -> campaign; then human-AI phase)
  fig2_design_space    Ashby-style chart regenerated in Arial (from analysis/ashby_chart.py)
  fig7_progression     fracture-progression comparison (slits 0 vs 20 deg; square veins vs load-parallel veins)
  figures/si/*         validation figures, atlas, costs, ten progression panels, tables (S1 validation, S3 holdouts)
All numbers come from the database, the validation results and the source tree.  Fonts: Arial everywhere."""
from __future__ import annotations
import os, sys, json, glob, shutil, subprocess, datetime, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FONT = {"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300}
plt.rcParams.update(FONT)
from experiments import db
from analysis.fracture_viz import load_traj, select_event_frames, draw_frame, damaged_atoms_until, atom_colors, progression_panel
from analysis.campaign_analysis import M
EV = 16.0217663
OUT = os.path.join(HERE, "figures"); SI = os.path.join(OUT, "si"); os.makedirs(SI, exist_ok=True)
PY = sys.executable
DARK, RED, BLUE, ORANGE, GREEN, PURPLE = "#222222", "#c0392b", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"


def save(fig, name, out=OUT):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rc_file():
    """A matplotlibrc with Arial for subprocesses that regenerate campaign figures."""
    p = os.path.join(tempfile.gettempdir(), "paper_arial_matplotlibrc")
    open(p, "w").write("font.family: Arial\nmathtext.fontset: custom\nmathtext.rm: Arial\nmathtext.it: Arial:italic\nmathtext.bf: Arial:bold\npdf.fonttype: 42\nsvg.fonttype: none\n")
    return p


def count_lines(paths):
    n = 0
    for p in paths:
        for f in glob.glob(os.path.join(ROOT, p), recursive=True):
            n += sum(1 for _ in open(f, errors="ignore"))
    return n


# --------------------------------------------------------------------------- facts about the experiment
def facts():
    R = [r for r in db.all_records() if r.get("status") == "completed"]
    ts = lambda r: datetime.datetime.strptime(r["run_id"][4:19], "%Y%m%d_%H%M%S")
    g = {"discovery": [r for r in R if r.get("campaign") == "discovery"], "paper": [r for r in R if r.get("campaign") == "paper"],
         "other": [r for r in R if r.get("campaign") not in ("discovery", "paper")]}
    F = {k: dict(n=len(v), first=min(ts(r) for r in v).strftime("%Y-%m-%d %H:%M"), last=max(ts(r) for r in v).strftime("%Y-%m-%d %H:%M"),
                 wall_h=sum((r.get("wall_time_s") or 0) for r in v) / 3600, atoms=sum(r["n_atoms"] for r in v)) for k, v in g.items() if v}
    comps = {"force-field kernel": ["potentials/rebo2scr/pytorch/*.py"], "parameters and tables": ["potentials/rebo2scr/reference/*.py"],
             "neighbor list, FIRE, AQS, MD": ["simulation/**/*.py"], "generators, descriptors, I/O": ["atomistics/**/*.py"], "analysis and figures": ["analysis/*.py"],
             "campaign, database, queue": ["experiments/*.py", "scripts/*.py"], "validation suite": ["validation/*.py"],
             "browser app": ["app/backend/*.py", "app/frontend/*.html", "app/frontend/static/*.js", "app/frontend/static/*.css"]}
    F["loc"] = {k: count_lines(v) for k, v in comps.items()}; F["loc_total"] = sum(F["loc"].values())
    V = json.load(open(os.path.join(ROOT, "validation", "results", "validation_table.json")))
    rows = V["tests"] if isinstance(V, dict) and "tests" in V else V
    F["n_tests"] = len(rows); F["n_pass"] = sum(r["status"] == "PASS" for r in rows)
    F["total_wall_h"] = sum((r.get("wall_time_s") or 0) for r in R) / 3600; F["total_runs"] = len(R)
    json.dump(F, open(os.path.join(OUT, "facts.json"), "w"), indent=1, default=str)
    return F


# --------------------------------------------------------------------------- figure 1: the experiment
def fig_experiment(F):
    fig = plt.figure(figsize=(7.0, 3.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4", ec="0.45", size=6.6, bold=False, color=DARK, lw=0.8, align="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2 if align == "center" else x + 1.2, y + h / 2, text, ha=align, va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal", linespacing=1.3, zorder=3)

    def arrow(x0, y0, x1, y1, color="0.35", lw=1.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=8), zorder=4)

    # phase 1 band
    ax.add_patch(FancyBboxPatch((1, 26), 98, 29, boxstyle="round,pad=0,rounding_size=1.5", fc="#eef3fa", ec="none", zorder=1))
    ax.text(2.5, 53.2, "Phase 1: one prompt, autonomous execution (Claude Fable 5.1 in Claude Code, 2026-09-04 to 09-06)", fontsize=7.2, fontweight="bold", color=BLUE, va="center")
    box(2.5, 39, 15, 12, "task prompt\n(one message):\nbuild, validate\nand use a discovery\nplatform for 2D\ncarbon architectures", fc="white", ec=BLUE, size=6.0)
    arrow(17.5, 45, 20.5, 45)
    box(20.5, 39, 25, 12, f"the instrument\nscreened-REBO2 engine in PyTorch (autograd, GPU),\nneighbour list, FIRE, quasi-static loading driver,\nstructure generators, descriptors, database, app\n{F['loc_total']:,} lines of new code", fc="white", ec=BLUE, size=5.6)
    arrow(45.5, 45, 48.5, 45)
    box(48.5, 39, 16, 12, f"validation gate\n{F['n_tests']} tests against the\nFortran reference:\n{F['n_pass']}/{F['n_tests']} passed\n(10$^{{-13}}$ eV/atom)", fc="#fdecea", ec=RED, size=5.8)
    arrow(64.5, 45, 67.5, 45)
    box(67.5, 39, 30, 12, f"the science\nhypothesis-driven campaign, 7 stages:\nbaselines, reconnaissance, hypotheses,\ndiscriminating tests, mechanisms, seeds,\n12 hashed holdout predictions; {F['discovery']['n']} simulations", fc="white", ec=GREEN, size=5.6)
    box(20.5, 28, 77, 8.5, "deliverables of phase 1: validated engine, database with trajectories, 70 figures, 7 movies, browser app, 54-page report, reproducibility records, one archive", fc="#f4f4f4", ec="0.6", size=6.0)
    arrow(58.5, 39, 58.5, 36.5)
    # phase 2 band
    ax.add_patch(FancyBboxPatch((1, 1), 98, 23, boxstyle="round,pad=0,rounding_size=1.5", fc="#f7f1ea", ec="none", zorder=1))
    ax.text(2.5, 22.2, "Phase 2: human-AI collaboration on the results (2026-09-06 to 09-13)", fontsize=7.2, fontweight="bold", color=ORANGE, va="center")
    steps = [("questions from the\nhuman scientist:\nAshby chart? novelty?\nmeasure alignment?\nhierarchy premium?", "white", ORANGE),
             ("new analyses:\ndesign-space chart,\nalignment index $A$,\nslit-tip overlap $O$,\nliterature check", "white", ORANGE),
             (f"pre-registered sweeps:\ntwo competing rules,\nhashed, then {F['paper']['n']} runs\n(angles, overlap\ncontrols, alignment)", "#fdecea", RED),
             ("paper:\nfigures regenerated\nfrom the database,\nnumbers as macros,\noverlap audit", "white", GREEN)]
    x = 2.5
    for k, (t, fc, ec) in enumerate(steps):
        box(x, 4, 21.5, 15.5, t, fc=fc, ec=ec, size=5.8); x += 24.5
        if k < 3: arrow(x - 3, 11.75, x, 11.75)
    ax.text(50, 2.2, f"{F['total_runs']} simulations in total, {F['total_wall_h']:.0f} process-hours on one Apple M4 Max; every number in this paper is regenerated from the archive", ha="center", fontsize=6.0, color="0.35")
    save(fig, "fig0_experiment")


# --------------------------------------------------------------------------- figure 7: fracture progression comparisons
def fig_progression():
    by = {r["name"]: r for r in db.all_records() if r.get("status") == "completed"}
    rows = [("S2_slit_0deg", "slits parallel to the load (0°)"), ("S7_slit_20deg", "slits at 20°: en-echelon linking"),
            ("S5_H2_D40_W8_ellipseX", "two levels, square veins, pores along the load"), ("S7_H2_veinsX_W12_ellipseX", "two levels, veins and pores along the load")]
    keys = ["relaxed", "first_damage", "peak_load", "post_peak", "final"]
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(len(rows), len(keys) + 2, width_ratios=[1] * len(keys) + [0.32, 1.25], wspace=0.06, hspace=0.32, left=0.03, right=0.99, top=0.95, bottom=0.05)
    for i, (name, title) in enumerate(rows):
        rec = by[name]; T = load_traj(os.path.join(ROOT, rec["trajectory"]))
        allsel = select_event_frames(T)
        def frame_for(key):
            for lab, f in allsel:
                if key in lab.split(" = "): return f
            return 0 if key == "relaxed" else len(T["eps_x"]) - 1
        pairs = sorted(zip(keys, [frame_for(k) for k in keys]), key=lambda kf: kf[1]); keys_r = [k for k, _ in pairs]; frames = [f for _, f in pairs]
        vals = [atom_colors(T, f, "energy_rel")[0] for f in frames]
        vmin = min(min(np.percentile(v, 1) for v in vals), 0); vmax = max(np.percentile(v, 99) for v in vals)
        for j, (k, f) in enumerate(zip(keys_r, frames)):
            ax = fig.add_subplot(gs[i, j])
            sc, clab = draw_frame(ax, T, f, "energy_rel", vmin, vmax, damage_atoms=damaged_atoms_until(rec, T["eps_x"][f]), s=1.2, lw=0.25, show_bonds=True)
            ax.set_title(f"{k.replace('_', ' ')}\nε = {T['eps_x'][f]:.3f}, σ = {T['sigma_xx'][f] * EV:.1f} N/m", fontsize=5.6, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.4)
            if j == 0: ax.text(-0.02, 0.5, f"{'abcd'[i]}", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="center")
        axs = fig.add_subplot(gs[i, len(keys) + 1])
        axs.plot(T["eps_x"], T["sigma_xx"] * EV, "-", color=DARK, lw=1.0)
        for f, col in zip(frames, plt.cm.tab10(range(len(frames)))):
            axs.plot(T["eps_x"][f], T["sigma_xx"][f] * EV, "o", ms=3.5, color=col)
        axs.set_xlabel("strain", fontsize=6.5, labelpad=1); axs.set_ylabel("2D stress (N/m)", fontsize=6.5, labelpad=1); axs.tick_params(labelsize=5.8, length=2)
        for sp in ("top", "right"): axs.spines[sp].set_visible(False)
        axs.set_title(f"{title}\nσ$_\\mathrm{{max}}$ = {M(rec, 'strength_Nm'):.1f} N/m, W = {M(rec, 'work_to_failure_J_m2'):.2f} J/m$^2$", fontsize=6.0, pad=2, linespacing=1.15)
    save(fig, "fig7_progression")


# --------------------------------------------------------------------------- supplementary material
def si_material(F):
    rc = rc_file(); env = dict(os.environ, MATPLOTLIBRC=rc, MPLBACKEND="Agg")
    # 1. campaign figures regenerated in Arial: Ashby chart (main Fig. 2) + atlas (SI), validation figures (SI)
    subprocess.run([PY, os.path.join(ROOT, "analysis", "ashby_chart.py")], cwd=ROOT, env=env, check=True, capture_output=True)
    subprocess.run([PY, os.path.join(ROOT, "validation", "validation_figures.py")], cwd=ROOT, env=env, check=True, capture_output=True)
    for src, dst, where in [("campaign/ashby_overview", "fig2_design_space", OUT), ("campaign/architecture_atlas", "figS_atlas", SI),
                            ("validation/agreement_with_reference", "figS_agreement", SI), ("validation/reference_backend_aqs", "figS_reference_curve", SI),
                            ("validation/performance", "figS_performance", SI), ("validation/precision_fast_vs_reference", "figS_precision", SI)]:
        for ext in ("png", "svg", "pdf"):
            shutil.copy(os.path.join(ROOT, "figures", f"{src}.{ext}"), os.path.join(where, f"{dst}.{ext}"))
    # 2. progression panels of the ten selected structures
    T = json.load(open(os.path.join(ROOT, "final_designs", "top_structures.json")))
    names = []
    for s in T["selected"]:
        progression_panel(s["run_id"], SI, "energy_rel"); names.append((s["name"], s["title"]))
    json.dump(names, open(os.path.join(SI, "progression_list.json"), "w"), indent=1)
    # 3. tables: validation (from the report generator), holdout predictions
    shutil.copy(os.path.join(ROOT, "report", "generated", "validation_table.tex"), os.path.join(OUT, "si_validation_table.tex"))
    p = os.path.join(OUT, "si_validation_table.tex"); t = open(p).read(); i = t.index("\\caption{"); j = t.index("\\label{tab:validation}") + len("\\label{tab:validation}")
    t = t[:i] + "\\caption{The \\nTests{} validation tests with expected and observed results, numerical error, tolerance and status.}\\label{tab:si-validation}" + t[j:]
    for a, b in (("minimiser", "minimizer"), ("neighbour", "neighbor"), ("colour", "color"), ("grey", "gray")): t = t.replace(a, b)   # American spelling in the paper copy
    open(p, "w").write(t)
    E = json.load(open(os.path.join(ROOT, "experiments", "holdouts", "holdout_evaluation.json")))
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%")
    rows = ["\\begin{tabular}{lrrrrrrl}", "\\toprule", "design & $\\sigma_\\mathrm{pred}$ & $\\sigma_\\mathrm{sim}$ & $\\varepsilon^f_\\mathrm{pred}$ & $\\varepsilon^f_\\mathrm{sim}$ & $W_\\mathrm{pred}$ & $W_\\mathrm{sim}$ & mode pred / sim\\\\ \\midrule"]
    for r in E["rows"]:
        p, o = r["predicted"], r["observed"]
        rows.append(f"{esc(r['name'])} & {p['strength_Nm']:.1f} & {o['strength_Nm']:.1f} & {p['failure_strain']:.3f} & {o['failure_strain']:.3f} & {p['work_to_failure_J_m2']:.2f} & {o['work_to_failure_J_m2']:.2f} & {esc(p['fracture_mode_class'])} / {esc(str(o['fracture_mode']).split(' ')[0])}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "si_holdout_table.tex"), "w").write("\n".join(rows))
    # 4. facts as LaTeX macros
    m = []
    def macro(n, v): m.append(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    macro("locTotal", f"{F['loc_total']:,}"); macro("nTests", F["n_tests"]); macro("nPass", F["n_pass"])
    macro("discFirst", F["discovery"]["first"]); macro("discLast", F["discovery"]["last"]); macro("discWall", f"{F['discovery']['wall_h']:.0f}"); macro("discN", F["discovery"]["n"])
    macro("paperFirst", F["paper"]["first"]); macro("paperLast", F["paper"]["last"]); macro("paperWall", f"{F['paper']['wall_h']:.0f}")
    macro("otherN", F.get("other", {}).get("n", 0)); macro("totalRuns", F["total_runs"]); macro("totalWall", f"{F['total_wall_h']:.0f}")
    macro("discFirstDay", F["discovery"]["first"][:10]); macro("discLastDay", F["discovery"]["last"][:10]); macro("paperFirstDay", F["paper"]["first"][:10]); macro("paperLastDay", F["paper"]["last"][:10])
    for k, v in F["loc"].items(): macro("loc" + "".join(w.capitalize() for w in k.replace(",", "").replace("/", " ").replace("-", " ").split()), f"{v:,}")
    open(os.path.join(OUT, "facts.tex"), "w").write("".join(m))


if __name__ == "__main__":
    F = facts()
    fig_experiment(F)
    fig_progression()
    si_material(F)
    print("extra figures and SI material written; facts:", {k: (v if not isinstance(v, dict) else "...") for k, v in F.items()})
