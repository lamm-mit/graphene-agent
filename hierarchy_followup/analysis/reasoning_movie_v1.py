"""Version 1 (short, 95 s; kept for reference and imported by reasoning_movie.py for its helpers and shared scenes).

A wordless film (about 90 s, MP4 H.264, 2560x1440, 30 fps) that follows the trajectory by which the agent explored the
world of hierarchical graphene metamaterials and how its decisions were made: the design language, the exploration of
the design space, the first-principles rules it extracted (weakest column x ligament strength; load-path alignment),
the decision to test hierarchy at a larger scale with pre-registered predictions, the mechanisms seen at 300 A, the
outcome of every registered test, and the principles that survived.

No words: the only text is numbers and single symbols on axes (sigma, epsilon, A, rho).  Rendering conventions are those of
the other movies (paper/make_movies.py, analysis/make_movies_phase3.py): white background, dark atoms and grey bonds for
structures, inferno energy colouring and red rings for fracture, tab10 panel colours matching the curves, interpolation
between stored quasi-static states, model results only.  Every number is read from the archives (phase-1 reference
copies and the follow-up database); nothing is drawn by hand.

    python analysis/reasoning_movie.py --preview          # 960x540, 8 fps, fast encoding, for checking the storyboard
    python analysis/reasoning_movie.py                    # 2560x1440, 30 fps master + 1080p compact version + poster frames

Outputs: movies_hq/R01_reasoning_trajectory.mp4 (+ _1080p.mp4), R01_reasoning_trajectory.json (scene list with times and
the data behind every scene), poster frames in movies_hq/R01_frames/.
"""
from __future__ import annotations
import os, sys, json, time, argparse, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.animation import FFMpegWriter
from scipy.spatial import cKDTree, ConvexHull
from experiments import db
from analysis.fracture_viz import load_traj, select_event_frames, damaged_atoms_until, EV
from analysis.assess_phase1 import ordered_mesh_level
from experiments.campaign_paper_sweeps import alignment_index
from atomistics.structures import design_space as D
from atomistics.descriptors.columns import vein_bands, in_vein

R_BOND = 2.0
DARK, GREY, RED, BLUE, ORANGE, GREEN, PURPLE, BROWN, PINK, CYAN = "#222222", "0.55", "#c0392b", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"
FAM = {"pristine": "0.45", "vacancies": "0.6", "precrack": BROWN, "nanomesh_single": BLUE, "nanomesh_hier": RED, "slit_array": ORANGE, "strut_lattice": GREEN, "voronoi_network": PURPLE, "ring_around_hole": PINK, "graded_pores": CYAN, "composite": PURPLE}
CMAP = LinearSegmentedColormap.from_list("inferno_trunc", plt.cm.inferno(np.linspace(0.0, 0.9, 256)))
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
OWN = {r["name"]: r for r in db.own_records() if r.get("status") == "completed"}
REF = {r["name"]: r for r in db.reference_records() if r.get("status") == "completed"}
ALL = dict(REF); ALL.update(OWN)
RREF = json.load(open(os.path.join(ROOT, "reference", "phase1_reference_numbers.json")))
PRED = {}
for f in ("hierarchy_followup_predictions.json", "hierarchy_T5_predictions.json"):
    PRED.update({p["name"]: p for p in json.load(open(os.path.join(ROOT, "experiments", "predictions", f)))["predictions"]})
LOG = {}


# ----------------------------------------------------------------------------- structures
def struct(name):
    """Relaxed-geometry-free structure of a design as generated (positions in A), from its record params."""
    r = ALL[name]; params = dict(r["params"])
    if r.get("seed") is not None and r["family"] != "pristine" and "seed" not in params: params["seed"] = r["seed"]
    a = D.generate(r["family"], **params)
    return a.positions[:, :2].astype(float), np.array([a.cell[0, 0], a.cell[1, 1]], float), a.info.get("design", {})


def bonds_of(P, L):
    tree = cKDTree(np.mod(P, L), boxsize=L)
    pairs = np.array(sorted(tree.query_pairs(R_BOND)), int)
    if len(pairs) == 0: return np.zeros((0, 2, 2))
    p0, p1 = P[pairs[:, 0]], P[pairs[:, 1]]; d = p1 - p0; d -= L * np.round(d / L)
    return np.stack([p0, p0 + d], 1)


def draw_struct(ax, P, L, px_per_A, color=DARK, alpha=1.0, bond_color="0.55"):
    """Atoms and bonds in the paper's snapshot style; returns the two artists (for crossfades)."""
    atom_pt = 0.95 * px_per_A * 72 / 100.0
    lc = LineCollection(bonds_of(P, L), colors=bond_color, linewidths=max(0.5, 0.18 * atom_pt), zorder=1, alpha=alpha); ax.add_collection(lc)
    sc = ax.scatter(P[:, 0], P[:, 1], s=atom_pt ** 2, c=color, linewidths=0, zorder=2, alpha=alpha)
    return lc, sc


def square_axes(ax, L, side=None, pad=1.5):
    side = side or float(max(L)); ax.set_xlim(L[0] / 2 - side / 2 - pad, L[0] / 2 + side / 2 + pad); ax.set_ylim(L[1] / 2 - side / 2 - pad, L[1] / 2 + side / 2 + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def style_chart(ax, xlabel, ylabel, fs):
    ax.set_xlabel(xlabel, fontsize=fs * 1.6, labelpad=6); ax.set_ylabel(ylabel, fontsize=fs * 1.6, labelpad=6); ax.tick_params(labelsize=fs * 1.15); ax.grid(alpha=0.2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def load_arrows(fig, box, color="0.6", s=1.0):
    """Two arrows pointing outward left and right of a panel box (figure fractions): uniaxial tension along x."""
    x, y, w, h = box; ym = y + h / 2; dx = 0.028 * s; g = 0.006
    for x0, x1 in ((x - g, x - g - dx), (x + w + g, x + w + g + dx)):
        fig.patches.append(FancyArrowPatch((x0, ym), (x1, ym), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22 * s, lw=2.5 * s, color=color))


# ----------------------------------------------------------------------------- trajectories (as in make_movies_phase3.Panel)
class Panel:
    def __init__(self, name, tile_below=60.0):
        self.rec = ALL[name]; self.name = name
        T = self.T = load_traj(db.trajectory_path(self.rec))
        self.eps = T["eps_x"].astype(float); self.sig = T["sigma_xx"].astype(float) * EV; self.nb = T["n_broken_cum"].astype(int)
        self.L = np.array([[c[0, 0], c[1, 1]] for c in T["cells"]], float)
        P = T["positions"][:, :, :2].astype(float); self.S = (P / self.L[:, None, :]) % 1.0
        self.E = T["peratom_energy"].astype(float) - T["peratom_energy"][0].astype(float)
        V = T["peratom_virial"].astype(float); self.Vxx = V[:, :, 0]
        self.n = len(self.eps); self.eps_max = float(self.eps[-1]); self.N = P.shape[1]
        self.tile = 2 if self.L[0].max() < tile_below else 1; self.side = float(self.L.max() * self.tile)
        ev = {}
        for lab, f in select_event_frames(T):
            for key in lab.split(" = "): ev[key] = f
        self.peak_eps = float(self.eps[int(np.argmax(self.sig))]); self.ip = int(np.argmax(self.sig))
        self.first_damage = float(self.eps[ev["first_damage"]]) if (self.nb[-1] > self.nb[0] and "first_damage" in ev) else None
        self.event_strains = sorted({float(self.eps[ev[k]]) for k in ("first_damage", "peak_load", "major_propagation") if k in ev} | {self.peak_eps, self.eps_max})
        self.smax = self.rec["metrics"]["strength_Nm"]

    def state(self, e):
        e = min(max(e, 0.0), self.eps_max); i = int(np.searchsorted(self.eps, e, side="right") - 1); i = min(max(i, 0), self.n - 1)
        if i >= self.n - 1 or self.eps[i + 1] <= self.eps[i]:
            s, L, en, sg = self.S[i], self.L[i], self.E[i], self.sig[i]
        else:
            t = (e - self.eps[i]) / (self.eps[i + 1] - self.eps[i]); ds = self.S[i + 1] - self.S[i]; ds -= np.round(ds)
            s = (self.S[i] + t * ds) % 1.0; L = self.L[i] + t * (self.L[i + 1] - self.L[i]); en = self.E[i] + t * (self.E[i + 1] - self.E[i]); sg = self.sig[i] + t * (self.sig[i + 1] - self.sig[i])
        P = s * L
        if self.tile > 1:
            shifts = np.array([[a, b] for a in range(self.tile) for b in range(self.tile)], float) * L
            P = np.concatenate([P + sh for sh in shifts]); en = np.tile(en, self.tile ** 2); L = L * self.tile
        return P, L, en, sg

    def damaged(self, e):
        d = damaged_atoms_until(self.rec, min(e, self.eps_max))
        if len(d) and self.tile > 1: d = np.concatenate([d + k * self.N for k in range(self.tile ** 2)])
        return d


def schedule(panels, n_frames, slow=3.0, halfwin=0.0025):
    """n_frames strains from 0 to the common e_max, uniform except 3x slower around the event strains."""
    e_max = max(p.eps_max for p in panels); ev = sorted({e for p in panels for e in p.event_strains if e < e_max - 1e-9})
    w = lambda e: slow if any(abs(e - x) <= halfwin for x in ev) else 1.0
    grid = np.linspace(0, e_max, 4000); cost = np.array([w(e) for e in grid]); cum = np.concatenate([[0], np.cumsum(cost[:-1] * np.diff(grid))])
    return np.interp(np.linspace(0, cum[-1], n_frames), cum, grid)


# ----------------------------------------------------------------------------- the movie
class Movie:
    def __init__(self, width, fps, out, crf=17, preset="slow"):
        self.W, self.H, self.fps, self.dpi = width, int(width * 9 / 16), fps, 100; self.s = self.H / 1440.0
        self.fig = plt.figure(figsize=(self.W / self.dpi, self.H / self.dpi), dpi=self.dpi); self.fig.patch.set_facecolor("white")
        self.writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1, extra_args=["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        self.out = out; self.nframes = 0; self.overlay = None; self.scene_log = []; self.posters = {}

    def n(self, seconds): return max(1, int(round(seconds * self.fps)))
    def fs(self, x): return x * self.s

    def begin(self, name):
        self.fig.clf(); self.fig.patches.clear()
        ov = self.fig.add_axes([0, 0, 1, 1], zorder=100); ov.axis("off"); ov.patch.set_alpha(0)
        self.overlay = ov.add_patch(Rectangle((0, 0), 1, 1, transform=ov.transAxes, facecolor="white", edgecolor="none", alpha=1.0, zorder=100))
        self.scene_log.append(dict(scene=name, start_s=round(self.nframes / self.fps, 2))); self.scene = name; print(f"[{self.nframes / self.fps:6.1f} s] {name}", flush=True)

    def frame(self, poster=None):
        self.writer.grab_frame()
        if poster and poster not in self.posters:
            self.posters[poster] = self.nframes
            self.fig.savefig(os.path.join(os.path.dirname(self.out), "R01_frames", f"{poster}.png"), dpi=self.dpi, facecolor="white")
        self.nframes += 1

    def fade_in(self, seconds=0.5, poster=None):
        for k in range(self.n(seconds)):
            self.overlay.set_alpha(1 - (k + 1) / self.n(seconds)); self.frame(poster if k == self.n(seconds) - 1 else None)
        self.overlay.set_alpha(0)

    def fade_out(self, seconds=0.5):
        for k in range(self.n(seconds)):
            self.overlay.set_alpha((k + 1) / self.n(seconds)); self.frame()
        self.scene_log[-1]["end_s"] = round(self.nframes / self.fps, 2)

    def hold(self, seconds, poster=None):
        for k in range(self.n(seconds)): self.frame(poster if k == 0 else None)


def ease(t): return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


# ----------------------------------------------------------------------------- scene 1: the design language
def scene_language(mv, seconds=9.0):
    mv.begin("1 design language: from the pristine sheet to nested levels")
    names = ["S1_pristine_zz", "S2_H1_p16", "S2_ellipse_90deg", "S2_slit_0deg", "H1_veinsX_r0", "S2_H3_D30_120"]
    # a 120 A pristine sheet instead of the 40 A baseline cell, so that all frames share one physical scale
    structs = []
    for n in names:
        if n == "S1_pristine_zz":
            a = D.generate("pristine", Lx=120.0, Ly=120.0, orientation="zigzag"); structs.append((a.positions[:, :2].astype(float), np.array([a.cell[0, 0], a.cell[1, 1]])))
        else:
            P, L, _ = struct(n); structs.append((P, L))
    side = max(max(L) for _, L in structs)
    box = [0.5 - 0.42 * mv.H / mv.W, 0.06, 0.84 * mv.H / mv.W, 0.84]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    px_per_A = box[2] * mv.W / (side + 3)
    for sp in ax.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
    load_arrows(mv.fig, box, s=mv.s)
    arts = [draw_struct(ax, P, L, px_per_A, alpha=0.0) for P, L in structs]
    square_axes(ax, structs[0][1], side)
    hold, xf = 0.9, 0.3
    for k in range(len(structs)):
        lc, sc = arts[k]; lc.set_alpha(1); sc.set_alpha(1)
        if k == 0: mv.fade_in(0.5, poster="s1_pristine")
        mv.hold(hold, poster=f"s1_{k}")
        if k + 1 < len(structs):
            for j in range(mv.n(xf)):
                t = ease((j + 1) / mv.n(xf)); arts[k][0].set_alpha(1 - t); arts[k][1].set_alpha(1 - t); arts[k + 1][0].set_alpha(t); arts[k + 1][1].set_alpha(t); mv.frame()
            arts[k][0].set_alpha(0); arts[k][1].set_alpha(0)
    LOG["scene1_designs"] = names
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 2: exploring the design space
def scene_exploration(mv, seconds=8.0):
    mv.begin("2 exploration: 96 discovery runs in the order they were run")
    disc = sorted([r for n, r in REF.items() if n.startswith("S")], key=lambda r: r["run_id"])
    rho = np.array([1 - r["descriptors"]["porosity"] for r in disc]); sig = np.array([r["metrics"]["strength_Nm"] for r in disc]); fam = [r["family"] for r in disc]
    fs = mv.fs(11)
    axc = mv.fig.add_axes([0.07, 0.12, 0.50, 0.80]); style_chart(axc, r"$\bar{\rho}$", r"$\sigma$", fs)
    axc.set_xlim(0.55, 1.03); axc.set_ylim(0, 42)
    pts = axc.scatter([], [], s=[], edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    box = [0.625, 0.27, 0.46 * mv.H / mv.W, 0.46]
    axs = mv.fig.add_axes(box); axs.set_facecolor("white")
    for sp in axs.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
    n_show = 24; show_idx = np.linspace(0, len(disc) - 1, n_show).round().astype(int)
    frames = mv.n(seconds); per_point = frames / len(disc)
    cur = [None]
    order_log = []
    def show_struct(i):
        for a in (cur[0] or []): a.remove()
        P, L, _ = struct(disc[i]["name"]); px = box[2] * mv.W / (max(L) + 3)
        cur[0] = draw_struct(axs, P, L, px, color=FAM.get(fam[i], DARK)); square_axes(axs, L); axs.set_xlim(axs.get_xlim()); 
        for sp in axs.spines.values(): sp.set_edgecolor(FAM.get(fam[i], DARK))
    show_struct(0); mv.fade_in(0.5)
    hulls = {}
    for k in range(frames):
        m = min(len(disc), int(k / per_point) + 1)
        sizes = np.full(m, 90.0 * mv.s ** 2); sizes[-1] = 90.0 * mv.s ** 2 * (1 + 2.5 * max(0.0, 1 - (k - (m - 1) * per_point) / per_point))
        pts.set_offsets(np.c_[rho[:m], sig[:m]]); pts.set_sizes(sizes); pts.set_facecolor([FAM.get(f, DARK) for f in fam[:m]])
        j = int(np.searchsorted(show_idx, m - 1, side="right") - 1)
        if j >= 0 and show_idx[j] != getattr(show_struct, "last", -1):
            show_struct(show_idx[j]); show_struct.last = show_idx[j]; order_log.append(disc[show_idx[j]]["name"])
        mv.frame(poster="s2_exploration" if k == frames // 2 else None)
    # family envelopes fade in
    for f in sorted(set(fam)):
        idx = [i for i, g in enumerate(fam) if g == f]
        if len(idx) >= 3:
            X = np.c_[rho[idx], sig[idx]]
            try:
                h = ConvexHull(X); poly = axc.fill(X[h.vertices, 0], X[h.vertices, 1], color=FAM.get(f, DARK), alpha=0.0, zorder=1, lw=0); hulls[f] = poly[0]
            except Exception: pass
    for k in range(mv.n(1.2)):
        for p in hulls.values(): p.set_alpha(0.13 * (k + 1) / mv.n(1.2))
        mv.frame(poster="s2_envelopes" if k == mv.n(1.2) - 1 else None)
    mv.hold(0.8)
    LOG["scene2"] = dict(n_designs=len(disc), families=sorted(set(fam)), thumbnails_in_order=order_log)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 3: the weakest column
def column_profile(P, L, dx=1.0, r=1.4):
    nx, ny = int(L[0] / dx), int(L[1] / dx); occ = np.zeros((nx, ny), bool)
    ix = ((P[:, 0] % L[0]) / dx).astype(int) % nx; iy = ((P[:, 1] % L[1]) / dx).astype(int) % ny
    for ddx in (-1, 0, 1):
        for ddy in (-1, 0, 1):
            occ[(ix + ddx) % nx, (iy + ddy) % ny] = True
    return occ.mean(axis=1), (np.arange(nx) + 0.5) * dx


def scene_rule(mv, seconds=12.0):
    mv.begin("3 first principle: strength = weakest column x ligament strength")
    fs = mv.fs(11); name = "S2_graded_x"
    P, L, _ = struct(name); prof, xs = column_profile(P, L); imin = int(np.argmin(prof))
    box = [0.06, 0.30, 0.34 * mv.H / mv.W, 0.34 * (mv.W / mv.H) * (mv.H / mv.W)]; box = [0.06, 0.32, 0.30 * mv.H / mv.W, 0.30 * (mv.W / mv.H)]
    boxw = 0.56; box = [0.05, 0.36, boxw * mv.H / mv.W, boxw]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
    draw_struct(ax, P, L, box[2] * mv.W / (max(L) + 3)); square_axes(ax, L); load_arrows(mv.fig, box, s=mv.s)
    band = ax.axvspan(0, 3, color=BLUE, alpha=0.25, zorder=4)
    axp = mv.fig.add_axes([box[0], 0.12, box[2], 0.18]); style_chart(axp, "", r"$A_\mathrm{col}/A$", fs); axp.set_xlim(0, L[0]); axp.set_ylim(0, 1.05); axp.set_xticks([])
    line, = axp.plot([], [], "-", color=BLUE, lw=2.5 * mv.s); dot, = axp.plot([], [], "o", color=BLUE, ms=9 * mv.s)
    axc = mv.fig.add_axes([0.56, 0.14, 0.40, 0.78]); style_chart(axc, r"$A_\mathrm{min}/A$", r"$\sigma$", fs); axc.set_xlim(0.1, 1.05); axc.set_ylim(0, 42)
    por = [r for n, r in REF.items() if n.startswith("S") and r["family"] not in ("pristine", "vacancies")]
    X = np.array([r["descriptors"]["min_solid_fraction_across_x"] for r in por]); Y = np.array([r["metrics"]["strength_Nm"] for r in por]); C = [FAM.get(r["family"], DARK) for r in por]
    pts = axc.scatter([], [], s=70 * mv.s ** 2, edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    l1, = axc.plot([], [], "-", color=DARK, lw=2.2 * mv.s, zorder=2); l2, = axc.plot([], [], "-", color=DARK, lw=2.2 * mv.s, zorder=2)
    mv.fade_in(0.5)
    # sweep the column
    for k in range(mv.n(2.6)):
        t = (k + 1) / mv.n(2.6); i = int(t * (len(xs) - 1)); band.set_x(xs[i] - 1.5); band.set_width(3.0)
        line.set_data(xs[:i + 1], prof[:i + 1]); dot.set_data([xs[i]], [prof[i]]); mv.frame()
    band.set_color(RED); band.set_alpha(0.35); band.set_x(xs[imin] - 1.5); dot.set_data([xs[imin]], [prof[imin]]); dot.set_color(RED)
    axp.axhline(prof[imin], color=RED, lw=1.5 * mv.s, ls="--"); mv.hold(1.2, poster="s3_weakest_column")
    # the points
    order = np.argsort(X)
    for k in range(mv.n(2.5)):
        m = int((k + 1) / mv.n(2.5) * len(por)); idx = order[:m]
        pts.set_offsets(np.c_[X[idx], Y[idx]].reshape(-1, 2)); pts.set_facecolor([C[i] for i in idx] or "none"); mv.frame()
    # the lines: sigma = (Amin/A) x sigma_lig for the two ligament classes of the predictor
    s_hi, s_lo = RREF["classes"]["straight"]["stored_median"], RREF["classes"]["fine_round"]["stored_median"]
    for k in range(mv.n(1.5)):
        t = ease((k + 1) / mv.n(1.5)); xx = np.array([0, 1.05 * t])
        l1.set_data(xx, s_hi * xx); l2.set_data(xx, s_lo * xx); mv.frame()
    mv.hold(1.7, poster="s3_net_section_rule")
    LOG["scene3"] = dict(structure=name, min_column_fraction=float(prof[imin]), n_points=len(por), lines_Nm=[s_hi, s_lo])
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 4: alignment, not hierarchy (120 A)
def scene_alignment(mv, seconds=12.0):
    mv.begin("4 alignment, not hierarchy: 40 meshes at 100-120 A")
    fs = mv.fs(11)
    meshes = [r for r in REF.values() if ordered_mesh_level(r) and 0.17 <= (r.get("porosity") or 0) <= 0.23]
    A = np.array([alignment_index(r["descriptors"]) for r in meshes]); S = np.array([r["metrics"]["strength_Nm"] for r in meshes]); lev = np.array([r.get("hierarchy_levels", 1) for r in meshes])
    one = lev == 1; fit = np.polyfit(A[one], S[one], 1)
    sweep = ["S2_ellipse_0deg", "P3_H1_ellipse_30deg", "P3_H1_ellipse_60deg", "S2_ellipse_90deg"]
    boxw = 0.50; box = [0.05, 0.40, boxw * mv.H / mv.W, boxw]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_edgecolor(BLUE); sp.set_linewidth(2 * mv.s)
    load_arrows(mv.fig, box, s=mv.s)
    structs = [struct(n) for n in sweep]; side = max(max(L) for _, L, _ in structs)
    arts = [draw_struct(ax, P, L, box[2] * mv.W / (side + 3), alpha=0.0, color=BLUE) for P, L, _ in structs]; square_axes(ax, structs[0][1], side)
    axc = mv.fig.add_axes([0.50, 0.14, 0.46, 0.78]); style_chart(axc, r"$A$", r"$\sigma$", fs); axc.set_xlim(-0.35, 0.72); axc.set_ylim(5, 22)
    p1 = axc.scatter([], [], s=90 * mv.s ** 2, c=BLUE, edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    p2 = axc.scatter([], [], s=100 * mv.s ** 2, c=RED, marker="s", edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    p3 = axc.scatter([], [], s=110 * mv.s ** 2, c=PURPLE, marker="^", edgecolors="white", linewidths=0.6 * mv.s, zorder=3)
    trend, = axc.plot([], [], "-", color=BLUE, lw=2.2 * mv.s, zorder=2)
    names = [r["name"] for r in meshes]; sweep_idx = [names.index(n) for n in sweep if n in names]
    mv.fade_in(0.5)
    # the sweep: structure crossfades while its point lights up
    shown = []
    for k, (n, i) in enumerate(zip(sweep, sweep_idx)):
        arts[k][0].set_alpha(1); arts[k][1].set_alpha(1); shown.append(i)
        p1.set_offsets(np.c_[A[shown], S[shown]]); mv.hold(0.8, poster=f"s4_sweep_{k}")
        if k + 1 < len(sweep):
            for j in range(mv.n(0.3)):
                t = ease((j + 1) / mv.n(0.3)); arts[k][0].set_alpha(1 - t); arts[k][1].set_alpha(1 - t); arts[k + 1][0].set_alpha(t); arts[k + 1][1].set_alpha(t); mv.frame()
            arts[k][0].set_alpha(0); arts[k][1].set_alpha(0)
    # all one-level meshes, then the trend
    idx1 = [i for i in np.where(one)[0]]
    for k in range(mv.n(1.5)):
        m = int((k + 1) / mv.n(1.5) * len(idx1)); ii = sorted(set(shown) | set(idx1[:m])); p1.set_offsets(np.c_[A[ii], S[ii]]); mv.frame()
    for k in range(mv.n(1.0)):
        t = ease((k + 1) / mv.n(1.0)); xx = np.array([A[one].min(), A[one].min() + t * (A[one].max() - A[one].min())]); trend.set_data(xx, np.polyval(fit, xx)); mv.frame()
    # nested meshes drop below the line; a stem from the trend to each point marks the premium
    idx2 = np.where(lev == 2)[0]; idx3 = np.where(lev >= 3)[0]; stems = []
    for k in range(mv.n(1.5)):
        t = (k + 1) / mv.n(1.5); m2 = int(t * len(idx2)); m3 = int(t * len(idx3))
        p2.set_offsets(np.c_[A[idx2[:m2]], S[idx2[:m2]]] if m2 else np.zeros((0, 2))); p3.set_offsets(np.c_[A[idx3[:m3]], S[idx3[:m3]]] if m3 else np.zeros((0, 2)))
        for i in list(idx2[:m2]) + list(idx3[:m3]):
            if i not in [s_[0] for s_ in stems]:
                ln, = axc.plot([A[i], A[i]], [np.polyval(fit, A[i]), S[i]], "-", color=RED, lw=1.6 * mv.s, alpha=0.7, zorder=2); stems.append((i, ln))
        mv.frame()
    mv.hold(2.0, poster="s4_hierarchy_premium")
    prem = S[lev >= 2] - np.polyval(fit, A[lev >= 2])
    LOG["scene4"] = dict(n_meshes=len(meshes), n_one=int(one.sum()), n_nested=int((lev >= 2).sum()), trend=[float(fit[0]), float(fit[1])], premium_mean=float(prem.mean()), premium_std=float(prem.std()), sweep=sweep)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 5: the decision, change the scale and register predictions
TESTS = [("H1_veinsX_L20_r0", "strength_Nm_rule"), ("H2_seamX_s30_r0", "strength_Nm_rule"), ("H3_2L_r0", "strength_Nm_rule"), ("H4_veinsX_W20_r0", "strength_Nm_rule"), ("H5_VF_E_r0", "strength_Nm_rule")]


def test_layout(mv, fig, tests, y_thumb=0.56, y_axis=0.14, h_axis=0.36, x0=0.08, x1=0.92, alpha=0.0):
    """Five thumbnails in a row above one shared sigma axis (0-25 N/m) whose x positions coincide with the thumbnails."""
    n = len(tests); axp = fig.add_axes([x0, y_axis, x1 - x0, h_axis]); style_chart(axp, "", r"$\sigma$", mv.fs(11)); axp.set_xlim(-0.6, n - 0.4); axp.set_ylim(0, 25); axp.set_xticks([])
    tw = (x1 - x0) / n * 0.72; thumbs = []
    for i, (n_, key) in enumerate(tests):
        xc = x0 + (x1 - x0) * (i + 0.6) / n; bx = [xc - tw / 2, y_thumb, tw, tw * mv.W / mv.H]
        axt = fig.add_axes(bx); axt.set_facecolor("white"); P, L, _ = struct(n_); draw_struct(axt, P, L, tw * mv.W / (max(L) + 3), alpha=alpha); square_axes(axt, L)
        for sp in axt.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(2 * mv.s)
        thumbs.append(axt)
    return thumbs, axp


def scene_decision(mv, seconds=9.0):
    mv.begin("5 decision: the same design at level ratio 3 and 8; five tests registered before running")
    fs = mv.fs(11)
    P1, L1, _ = struct("H1_veinsX_r0"); P3, L3, _ = struct("H3_2L_r0")
    boxw = 0.80; box = [0.5 - boxw * mv.H / mv.W / 2, 0.10, boxw * mv.H / mv.W, boxw]
    ax = mv.fig.add_axes(box); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(2 * mv.s)
    px3 = box[2] * mv.W / (max(L3) + 3); px1_full = box[2] * mv.W / (max(L1) + 3)
    # the 120 A cell first fills the panel, then shrinks to its true scale inside the 300 A cell, which fades in
    a1 = draw_struct(ax, P1, L1, px1_full, color=RED); a3 = draw_struct(ax, P3, L3, px3, alpha=0.0)
    square_axes(ax, L1); load_arrows(mv.fig, box, s=mv.s)
    mv.fade_in(0.5, poster="s5_120A"); mv.hold(0.8)
    n = mv.n(2.2)
    for k in range(n):
        t = ease((k + 1) / n); side = max(L1) + t * (max(L3) - max(L1)); c = L1 / 2 + t * (L3 / 2 - L1 / 2)
        ax.set_xlim(c[0] - side / 2 - 1.5, c[0] + side / 2 + 1.5); ax.set_ylim(c[1] - side / 2 - 1.5, c[1] + side / 2 + 1.5)
        pt = 0.95 * (box[2] * mv.W / (side + 3)) * 72 / 100.0; a1[1].set_sizes([pt ** 2]); a1[0].set_linewidths([max(0.5, 0.18 * pt)]); a3[1].set_sizes([pt ** 2]); a3[0].set_linewidths([max(0.5, 0.18 * pt)])
        a3[0].set_alpha(t); a3[1].set_alpha(t); mv.frame()
    mv.hold(1.0, poster="s5_300A")
    # five tests: thumbnails above a common sigma axis; a hollow circle marks the strength predicted (and hashed) before each run
    tests = TESTS
    for k in range(mv.n(0.4)):
        t = (k + 1) / mv.n(0.4); a1[0].set_alpha(1 - t); a1[1].set_alpha(1 - t); a3[0].set_alpha(1 - t); a3[1].set_alpha(1 - t); mv.frame()
    ax.remove(); mv.fig.patches.clear()
    thumbs, axp = test_layout(mv, mv.fig, tests)
    for k in range(mv.n(0.8)):
        t = (k + 1) / mv.n(0.8)
        for axt in thumbs:
            for c in axt.collections: c.set_alpha(t)
        mv.frame()
    for i, (n_, key) in enumerate(tests):
        axp.plot([i], [PRED[n_]["predicted"][key]], "o", ms=18 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.4 * mv.s, zorder=4); mv.hold(0.3)
    mv.hold(1.4, poster="s5_tests_registered")
    LOG["scene5"] = dict(tests=[t[0] for t in tests], predicted_strength_Nm={t[0]: PRED[t[0]]["predicted"][t[1]] for t in tests}, prediction_files=["hierarchy_followup_predictions.json", "hierarchy_T5_predictions.json"])
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 6: the mechanisms at 300 A
def fracture_scene(mv, name_tag, panels_names, seconds, poster_tag, chart=True, vein_load_intro=False):
    mv.begin(name_tag)
    panels = [Panel(n) for n in panels_names]; n = len(panels); colors = plt.cm.tab10(np.arange(n) % 10)
    s = mv.s; m = 0.03; g = 0.02; W, H = mv.W, mv.H
    ph = 0.62; pw = ph * H / W; total = n * pw + (n - 1) * g; x0 = (1 - total) / 2 - 0.02
    boxes = [[x0 + k * (pw + g), 0.34, pw, ph] for k in range(n)]
    def prepeak(p): return max(1e-3, max(np.percentile(p.E[k], 97) for k in range(p.ip + 1)))
    vmax = float(np.exp(np.mean([np.log(prepeak(p)) for p in panels]))); vmin = min(0.0, min(np.percentile(p.E[k], 0.5) for p in panels for k in range(p.n)))
    e_max = max(p.eps_max for p in panels)
    axc = mv.fig.add_axes([0.09, 0.07, 0.84, 0.22]); style_chart(axc, r"$\varepsilon$", r"$\sigma$", mv.fs(11))
    for k, p in enumerate(panels): axc.plot(p.eps, p.sig, "-", color=colors[k], lw=2.6 * s)
    markers = [axc.plot([], [], "o", ms=14 * s, color=colors[k], markeredgecolor="k", markeredgewidth=1.2 * s, zorder=5)[0] for k in range(n)]
    vline = axc.axvline(0, color="0.5", lw=1.0 * s, ls="--"); axc.set_xlim(0, e_max * 1.02); axc.set_ylim(0, max(p.sig.max() for p in panels) * 1.08)
    arts = []
    for k, (p, box) in enumerate(zip(panels, boxes)):
        ax = mv.fig.add_axes(box); ax.set_facecolor("white")
        P, L, en, sg = p.state(0.0); px_per_A = box[2] * W / (p.side + 3.0); atom_pt = 0.95 * px_per_A * 72 / 100.0
        lc = LineCollection(bonds_of(P, L), colors="0.55", linewidths=max(0.6, 0.18 * atom_pt), zorder=1); ax.add_collection(lc)
        sc = ax.scatter(P[:, 0], P[:, 1], c=en, s=atom_pt ** 2, cmap=CMAP, vmin=vmin, vmax=vmax, linewidths=0, zorder=2)
        dm = ax.scatter([], [], s=atom_pt ** 2 * 5, facecolors="none", edgecolors="red", linewidths=max(0.8, 0.25 * atom_pt), zorder=3)
        square_axes(ax, L, p.side)
        for sp in ax.spines.values(): sp.set_edgecolor(colors[k]); sp.set_linewidth(3 * s)
        arts.append(dict(ax=ax, lc=lc, sc=sc, dm=dm)); load_arrows(mv.fig, box, s=s * 0.8)
    cax = mv.fig.add_axes([boxes[-1][0] + boxes[-1][2] + 0.012, 0.34, 0.008, ph]); cb = mv.fig.colorbar(arts[-1]["sc"], cax=cax); cb.ax.tick_params(labelsize=mv.fs(11))
    def update(e):
        for k, (p, a) in enumerate(zip(panels, arts)):
            P, L, en, sg = p.state(e); a["lc"].set_segments(bonds_of(P, L)); a["sc"].set_offsets(P); a["sc"].set_array(en)
            dmg = p.damaged(e); a["dm"].set_offsets(P[dmg] if len(dmg) else np.zeros((0, 2)))
            hs = p.side / 2 + 1.5; a["ax"].set_xlim(L[0] / 2 - hs, L[0] / 2 + hs); a["ax"].set_ylim(L[1] / 2 - hs, L[1] / 2 + hs)
            markers[k].set_data([min(e, p.eps_max)], [sg])
        vline.set_xdata([e, e])
    update(0.0)
    if vein_load_intro:
        # before any damage: the two-level panel coloured by the load carried per atom (magma), then back to the energy colouring
        p = panels[-1]; a = arts[-1]; ip = p.ip; k = int(np.searchsorted(p.eps, 0.8 * p.eps[ip])); v = p.Vxx[k]; share = v / v.mean()
        P, L, en, sg = p.state(float(p.eps[k])); a["sc"].set_offsets(P); a["sc"].set_array(share); a["sc"].set_cmap("magma"); a["sc"].set_clim(0, 2)
        a["lc"].set_segments(bonds_of(P, L)); markers[-1].set_data([p.eps[k]], [sg]); vline.set_xdata([p.eps[k]] * 2)
        mv.fade_in(0.5); mv.hold(1.8, poster=f"{poster_tag}_load_share")
        for j in range(mv.n(0.4)):
            mv.frame()
        a["sc"].set_cmap(CMAP); a["sc"].set_clim(vmin, vmax); update(0.0); mv.hold(0.3)
    else:
        mv.fade_in(0.5)
    strains = schedule(panels, mv.n(seconds))
    for j, e in enumerate(strains):
        update(float(e)); mv.frame(poster=f"{poster_tag}_peak" if abs(e - panels[-1].peak_eps) < strains[1] - strains[0] and j > 0 else None)
    mv.hold(1.0, poster=f"{poster_tag}_end")
    LOG[name_tag] = dict(panels=panels_names, colour_scale_eV=[round(vmin, 3), round(vmax, 3)], strain_seconds=seconds)
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 7: outcomes of the registered tests, and the rule under a crack
def scene_outcomes(mv, seconds=10.0):
    mv.begin("7 outcomes: observed against predicted strength; every registered hypothesis against its criterion; no design above the net-section rule under a crack")
    fs = mv.fs(11); tests = TESTS
    res = json.load(open(os.path.join(ROOT, "results", "hierarchy_followup_results.json")))
    # hypothesis outcomes as the ratio of the achieved quantity to its registered criterion (1 = met exactly)
    t1 = res["T1"]["verdict"]["L20_r0"]["over_rule"] / 1.15
    t2 = max(x["W_ratio"] for x in res["T2"]["rows"] if x["key"] in ("seamX", "weakseamX")) / 1.3
    t3r = res["T3"]["verdict"]["r0"]; t3 = min(t3r["W_premium_vs_fine"], t3r["W_premium_vs_coarse"]) / RREF["alignment"]["one_level_W_std"]
    t4rows = [x for x in res["T4"]["rows"] if "(phase 1)" not in x["name"]]; t4 = max(min(x["residual_after_first_avalanche"] / 0.6, x["post_peak_energy_J_m2"] / (1.5 * x["control_post_peak_energy"])) for x in t4rows)
    t5rows = [x for x in res["T5"]["rows"] if x["name"] in ("H5_VF_E_r0", "H5_VF_E_r1")]; t5 = min(x["residual_after_first_avalanche"] for x in t5rows) / 0.6
    ratios = [t1, t2, t3, t4, t5]
    # left: the five tests (thumbnails, sigma axis with the hollow predictions already there, hypothesis axis above)
    thumbs, axp = test_layout(mv, mv.fig, tests, y_thumb=0.06, y_axis=0.30, h_axis=0.26, x0=0.06, x1=0.50, alpha=1.0)
    for i, (n_, key) in enumerate(tests):
        axp.plot([i], [PRED[n_]["predicted"][key]], "o", ms=18 * mv.s, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=2.4 * mv.s, zorder=4)
    axh = mv.fig.add_axes([0.06, 0.62, 0.44, 0.30]); style_chart(axh, "", "", fs); axh.set_xlim(-0.6, 4.6); axh.set_ylim(0, 1.8); axh.set_xticks([]); axh.axhline(1.0, color=DARK, lw=1.8 * mv.s, ls="--")
    axh.scatter(range(5), [1.0] * 5, s=260 * mv.s ** 2, facecolors="white", edgecolors=DARK, linewidths=2.2 * mv.s, zorder=4)
    filled = [axh.scatter([], [], s=260 * mv.s ** 2, zorder=5) for _ in tests]; stems = [axh.plot([], [], "-", lw=2.5 * mv.s, zorder=3)[0] for _ in tests]
    # right: retention against the net-section rule for every cracked run of the follow-up
    axr = mv.fig.add_axes([0.58, 0.30, 0.38, 0.62]); style_chart(axr, "", "", fs); axr.set_xlim(0.6, 0.97); axr.set_ylim(0.35, 1.0)
    axr.plot([0.5, 1.0], [0.5, 1.0], "-", color=DARK, lw=2 * mv.s); axr.fill_between([0.5, 1.0], [0.5, 1.0], [0.3, 0.3], color="0.94", zorder=0)
    rows = []
    for n_, r in OWN.items():
        if "_L" in n_ and not n_.startswith("H2"):
            unc = n_.split("_L")[0] + "_r" + n_[-1]; s_u = RREF["references"]["S1_pristine_zz"]["strength_Nm"] if n_.startswith("H1_pristine") else OWN[unc]["metrics"]["strength_Nm"]
            rows.append((PRED[n_]["predicted"]["retention_rule"], r["metrics"]["strength_Nm"] / s_u, r["descriptors"]["Lx"] > 200, FAM.get(r["family"], DARK) if r["family"] != "composite" else PURPLE))
    rows.sort(key=lambda x: x[0]); ret = axr.scatter([], [], s=110 * mv.s ** 2, zorder=3)
    mv.fade_in(0.5); mv.hold(0.4)
    # beat 1: the observed strengths fill in (green within 15 % of the prediction, red beyond)
    obs = []
    for i, (n_, key) in enumerate(tests):
        pr, ob = PRED[n_]["predicted"][key], OWN[n_]["metrics"]["strength_Nm"]; col = GREEN if abs(ob - pr) / ob <= 0.15 else RED; obs.append(ob)
        ln, = axp.plot([], [], "-", color=col, lw=2.5 * mv.s, zorder=3); dot, = axp.plot([], [], "o", ms=18 * mv.s, markerfacecolor=col, markeredgecolor=DARK, markeredgewidth=1.5 * mv.s, zorder=5)
        for k in range(mv.n(0.4)):
            t = ease((k + 1) / mv.n(0.4)); y = pr + t * (ob - pr); ln.set_data([i, i], [pr, y]); dot.set_data([i], [y]); mv.frame()
        mv.hold(0.15)
    mv.hold(0.7, poster="s7_predicted_vs_observed")
    # beat 2: the hypotheses
    for i, v in enumerate(ratios):
        col = GREEN if v >= 1.0 else RED
        for k in range(mv.n(0.4)):
            t = ease((k + 1) / mv.n(0.4)); y = 1.0 + t * (v - 1.0); filled[i].set_offsets([[i, y]]); filled[i].set_facecolor(col); filled[i].set_edgecolor(DARK); stems[i].set_data([i, i], [1.0, y]); stems[i].set_color(col); mv.frame()
        mv.hold(0.15)
    mv.hold(0.8, poster="s7_test_outcomes")
    # beat 3: retention against the rule
    for k in range(mv.n(2.0)):
        m = int((k + 1) / mv.n(2.0) * len(rows)); ret.set_offsets(np.array([[x, y] for x, y, _, _ in rows[:m]], float).reshape(-1, 2)); ret.set_facecolor([c if big else "white" for _, _, big, c in rows[:m]]); ret.set_edgecolor([c for _, _, _, c in rows[:m]]); ret.set_linewidths(1.6 * mv.s); mv.frame()
    mv.hold(1.4, poster="s7_retention_rule")
    LOG["scene7"] = dict(tests=[t[0] for t in tests], predicted_strength_Nm=[PRED[t[0]]["predicted"][t[1]] for t in tests], observed_strength_Nm=obs, hypothesis_ratio_to_criterion=[round(v, 3) for v in ratios], n_cracked=len(rows))
    mv.fade_out(0.5)


# ----------------------------------------------------------------------------- scene 8: the principles that survived
def scene_principles(mv, seconds=6.0):
    mv.begin("8 principles: the weakest column sets the strength; hierarchy compartmentalises; the building blocks set what happens after the peak")
    fs = mv.fs(11); s = mv.s
    boxes = [[0.05 + k * 0.315, 0.22, 0.27, 0.27 * mv.W / mv.H] for k in range(3)]
    # (i) the rule
    ax = mv.fig.add_axes(boxes[0]); style_chart(ax, r"$A_\mathrm{min}/A$", r"$\sigma$", fs); ax.set_xlim(0.1, 1.05); ax.set_ylim(0, 42)
    por = [r for n, r in REF.items() if n.startswith("S") and r["family"] not in ("pristine", "vacancies")]
    ax.scatter([r["descriptors"]["min_solid_fraction_across_x"] for r in por], [r["metrics"]["strength_Nm"] for r in por], s=40 * s ** 2, c=[FAM.get(r["family"], DARK) for r in por], edgecolors="white", linewidths=0.5 * s, zorder=3)
    xx = np.array([0, 1.05]); ax.plot(xx, RREF["classes"]["straight"]["stored_median"] * xx, "-", color=DARK, lw=2 * s); ax.plot(xx, RREF["classes"]["fine_round"]["stored_median"] * xx, "-", color=DARK, lw=2 * s)
    # (ii) compartments: the two-level mesh during major propagation
    p = Panel("H3_2L_r0"); ev = {}
    for lab, f in select_event_frames(p.T):
        for key in lab.split(" = "): ev[key] = f
    k = ev.get("major_propagation", p.ip + 2); P, L, en, sg = p.state(float(p.eps[k]))
    ax2 = mv.fig.add_axes(boxes[1]); px = boxes[1][2] * mv.W / (p.side + 3); pt = 0.95 * px * 72 / 100
    ax2.add_collection(LineCollection(bonds_of(P, L), colors="0.55", linewidths=max(0.5, 0.18 * pt), zorder=1)); ax2.scatter(P[:, 0], P[:, 1], s=pt ** 2, c="#2b2b2b", linewidths=0, zorder=2)
    d = p.damaged(float(p.eps[k])); ax2.scatter(P[d, 0], P[d, 1], s=pt ** 2 * 4, c=RED, linewidths=0, zorder=3); square_axes(ax2, L, p.side)
    for sp in ax2.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(2 * s)
    # (iii) building blocks: post-peak curves of solid veins, fiber-bundle veins, three levels (registry 0)
    ax3 = mv.fig.add_axes(boxes[2]); style_chart(ax3, r"$\varepsilon - \varepsilon_\mathrm{peak}$", r"$\sigma/\sigma_\mathrm{peak}$", fs); ax3.set_xlim(0, 0.2); ax3.set_ylim(0, 1.05)
    for n_, col in (("H3_1L_fine_r0", BLUE), ("H3_2L_r0", RED), ("H5_VF_E_r0", PURPLE), ("H5_V3_E_r0", BROWN)):
        c = db.stress_strain(OWN[n_]); e, sg = np.asarray(c["eps_x"]), np.asarray(c["sigma_xx_Nm"]); ip = int(np.argmax(sg)); ax3.plot(e[ip:] - e[ip], sg[ip:] / sg[ip], "-", color=col, lw=2.4 * s)
    mv.fade_in(0.8, poster="s8_principles"); mv.hold(seconds - 0.8)
    LOG["scene8"] = dict(panels=["net-section rule (81 porous phase-1 designs, lines at 30 and 17 N/m per unit section)", "H3_2L_r0 during major propagation", "post-peak curves of H3_1L_fine_r0, H3_2L_r0, H5_VF_E_r0, H5_V3_E_r0"])
    mv.fade_out(1.5)


# ----------------------------------------------------------------------------- main
def compact(base, width=1920, crf=22, suffix="_1080p"):
    src, dst = base + ".mp4", base + suffix + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); ap.add_argument("--outdir", default=os.path.join(ROOT, "movies_hq")); ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    width, fps, crf, preset = (960, 8, 28, "ultrafast") if a.preview else (2560, 30, 17, "slow")
    os.makedirs(os.path.join(a.outdir, "R01_frames"), exist_ok=True)
    base = os.path.join(a.outdir, "R01_reasoning_trajectory" + ("_preview" if a.preview else ""))
    mv = Movie(width, fps, base + ".mp4", crf=crf, preset=preset); t0 = time.time()
    scenes = [("1", lambda: scene_language(mv)), ("2", lambda: scene_exploration(mv)), ("3", lambda: scene_rule(mv)), ("4", lambda: scene_alignment(mv)), ("5", lambda: scene_decision(mv)),
              ("6a", lambda: fracture_scene(mv, "6a mechanisms at 300 A: one level against two levels", ["H3_1L_fine_r0", "H3_2L_r0"], 7.5, "s6a", vein_load_intro=True)),
              ("6b", lambda: fracture_scene(mv, "6b a contained crack is held at the veins and deflected", ["H3_2L_L60_r0"], 5.0, "s6b")),
              ("6c", lambda: fracture_scene(mv, "6c fiber-bundle veins fail strip by strip", ["H5_VF_E_r0"], 5.0, "s6c")),
              ("7", lambda: scene_outcomes(mv)), ("8", lambda: scene_principles(mv))]
    with mv.writer.saving(mv.fig, base + ".mp4", mv.dpi):
        for key, fn in scenes:
            if not a.only or key in a.only: fn()
    plt.close(mv.fig)
    meta = dict(file=os.path.basename(base) + ".mp4", resolution=f"{mv.W}x{mv.H}", fps=fps, seconds=round(mv.nframes / fps, 1), scenes=mv.scene_log, data=LOG, text_in_movie="numbers and single-symbol axis labels only",
                conventions="white background; structures: dark atoms, grey bonds; fracture: energy change per atom (inferno, truncated), red rings = atoms that lost a bond, panel frame colour = curve colour, frames between stored states interpolated; model results only (screened REBO2, athermal quasi-static tension along x)",
                render_minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(base + ".json", "w"), indent=1)
    if not a.preview: compact(base)
    print("done", base, f"{mv.nframes / fps:.1f} s", f"{(time.time() - t0) / 60:.1f} min")
