"""High-quality fracture movies (MP4, H.264, 2560x1440, 30 fps) of the key trajectories: synchronised multi-panel
movies that cover the design space and the mechanisms, plus single-panel close-ups.

All panels of a movie advance at the same engineering strain; a panel whose simulation has ended holds its final
state and is marked as failed.  Between the stored quasi-static states the atomic positions, cell and per-atom
energies are interpolated linearly (in fractional coordinates, minimum image): the intermediate configurations are
interpolations, not simulated states.  Bonds are recomputed from the interpolated positions with the 2.0 A
criterion of the analysis, so that a bond disappears when the interpolated distance crosses the criterion.
Atoms are coloured by the energy change per atom relative to the relaxed state (inferno colour map truncated so that
the top of the range is a saturated yellow); atoms that have lost a bond are ringed in red; the chart shows the stress-strain curves with a
moving marker per panel.  Cells smaller than 60 A are shown as 2x2 periodic images.  The movies contain NO text: all
captions, panel identities (left to right), scales and numbers are in the JSON/README/TeX files written next to them;
panel frame colours match the curve colours.

    python make_movies.py --list
    python make_movies.py --only M08 --seconds-per-strain 20 --width 1280 --fps 12   # quick test
    python make_movies.py --all --jobs 4                                      # the full set

Outputs: <outdir>/<id>_<slug>.mp4, a poster PNG, a JSON caption per movie, captions.json and README.md."""
from __future__ import annotations
import os, sys, json, argparse, time, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FFMpegWriter
from scipy.spatial import cKDTree
from experiments import db
from analysis.fracture_viz import load_traj, select_event_frames, damaged_atoms_until, EV

R_BOND = 2.0
OUT_DEFAULT = os.path.join(HERE, "movies_hq")
# inferno truncated at 0.9 so that saturated (top-of-range) atoms are a strong yellow, visible on the white background
CMAP = LinearSegmentedColormap.from_list("inferno_trunc", plt.cm.inferno(np.linspace(0.0, 0.9, 256)))
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})

# ----------------------------------------------------------------------------- the movies
MOVIES = [
 dict(id="M01", slug="baselines_pristine_vs_precrack", title="Baselines: intrinsic strength and a single flaw",
      insight="The pristine zigzag sheet reaches 38.8 N/m and fails in one avalanche; a 20 Å crack halves the strength (18.5 N/m) and fails by crack propagation at 9% strain.",
      panels=[("S1_pristine_zz", "pristine zigzag graphene"), ("S1_precrack_L20", "20 Å precrack")]),
 dict(id="M02", slug="slit_angle_series", title="Tilting a load-aligned slit array: three mechanisms, one minimum",
      insight="Same porosity, five slit angles. 0°: straight load paths (25.4 N/m). 20°: tips of adjacent rows overlap and link en echelon (8.9 N/m, the minimum). 45°: ligaments rotate before they break (18.5 N/m, modulus collapses). 60° and 90°: the load crosses bridges between slits in bending (7.8 and 4.0 N/m).",
      panels=[("S2_slit_0deg", "0°"), ("S7_slit_20deg", "20°"), ("S2_slit_45deg", "45°"), ("P1_slit_60deg", "60°"), ("S2_slit_90deg", "90°")]),
 dict(id="M03", slug="tip_overlap_control", title="Pre-registered control: tip overlap is necessary for the collapse",
      insight="Three arrays at 20°. With slits too short to overlap (porosity 0.10) the array keeps 20.0 N/m, 78% of the untilted value; with overlapping tips (porosity 0.20 and 0.30) the ligaments between the tips fail first and link across the sheet.",
      panels=[("P2_slit_20deg_phi0.10", "φ = 0.10, no overlap"), ("S7_slit_20deg", "φ = 0.20, overlap"), ("P2_slit_20deg_phi0.30", "φ = 0.30, overlap")]),
 dict(id="M04", slug="hierarchy_vs_alignment", title="Hierarchy is not a free lunch: alignment, level by level",
      insight="All at porosity 0.20. The single-level square mesh (17.0 N/m) beats the nested mesh with veins both ways and round pores (12.8). Aligning the veins with the load (16.0) or elongating the fine pores along it (17.5) recovers the loss; doing both gives the strongest nested design (19.4), because each level then adds straight load paths.",
      panels=[("S2_H1_square_D40", "single level"), ("S2_H2_D40_W8", "nested, unaligned"), ("S4_H2_D40_W8_veins_x", "veins along load"), ("S5_H2_D40_W8_ellipseX", "pores along load"), ("S7_H2_veinsX_W12_ellipseX", "both aligned")]),
 dict(id="M05", slug="alignment_sweep_single_level", title="Load-path alignment in a single-level mesh",
      insight="Elongated pores rotated from across the load (alignment index A = −0.24, 8.6 N/m) through 30° and 60° to along the load (A = +0.35, 19.0 N/m): strength rises monotonically with the alignment of the material with the load, at constant mass.",
      panels=[("S2_ellipse_0deg", "pores across the load"), ("P3_H1_ellipse_30deg", "30°"), ("P3_H1_ellipse_60deg", "60°"), ("S2_ellipse_90deg", "pores along the load")]),
 dict(id="M06", slug="disorder_and_gradients_are_costs", title="Disorder and gradients are costs",
      insight="Same porosity as the ordered fine mesh (11.7 N/m for this lattice registry, 11.7 to 13.3 across registries): positional jitter of 2 Å (11.8 for this seed, 10.4 to 11.8 across seeds), pore-size dispersion (11.3), a Voronoi network (14.4 for this seed, 9.2 to 14.4 across seeds) and a pore-size gradient along the load (10.1, set by its weakest band). None is a toughening mechanism in this model.",
      panels=[("S2_H1_p16", "ordered mesh"), ("S2_mesh_jitter2.0", "jitter 2 Å"), ("S2_mesh_sizedis0.3", "size dispersion 0.3"), ("S2_voronoi_n25_reg0.6", "Voronoi, regularity 0.6"), ("S2_graded_x", "graded along load")]),
 dict(id="M07", slug="flaw_halo", title="A halo of pores around a hole enlarges the flaw",
      insight="A bare 20 Å hole (19.1 N/m), the same hole surrounded by two rings of pores (16.2 N/m) and by far-field pores (13.2 N/m): the rings and the pores turn the hole into a wider effective flaw bounded by thin ligaments; they do not shield it.",
      panels=[("S1_hole_d20", "hole, 20 Å"), ("S2_hole20_rings2", "hole + 2 rings of pores"), ("S2_hole20_background", "hole + far-field pores")]),
 dict(id="M08", slug="fracture_modes_progressive_vs_avalanche", title="Fracture mode is a load-transfer property",
      insight="Eight parallel strips (armchair slit array) fail row by row and keep carrying 8 to 10 N/m to 25% strain; three wide rows with 15% width dispersion cascade in one avalanche at 14% strain. Many parallel paths of moderate dispersion share the released load; few wide paths cannot.",
      panels=[("S7_slit_0deg_armchair", "8 strips, progressive"), ("S4_square_D40_dispersed", "3 wide rows, one avalanche")]),
 dict(id="M09", slug="closeup_en_echelon_linking", title="Close-up: en-echelon linking of overlapping slit tips (20°)",
      insight="The short inclined ligaments between the tips of slits in adjacent rows carry the load in combined tension and shear and fail first; each failed ligament hands its load to the next one along the same diagonal, so the cracks link across the sheet in a staircase while the remaining strips keep carrying about 8 N/m.",
      panels=[("S7_slit_20deg", "slit array at 20°, porosity 0.20")]),
 dict(id="M10", slug="closeup_aligned_hierarchy", title="Close-up: the strongest nested design (veins and pores along the load)",
      insight="Fine pores elongated along the load inside veins that run along the load: 19.4 N/m at porosity 0.21. First damage appears at the bridges between the fine pores, the peak follows, and the fine-pore rows fail in a few steps while the veins carry the remaining load. A holdout that the data-driven predictor underestimated by a third.",
      panels=[("S7_H2_veinsX_W12_ellipseX", "two levels, both aligned, porosity 0.21")]),
]
FOOT = ("Screened REBO2, athermal quasi-static uniaxial tension along x; colour: energy change per atom relative to the relaxed state (eV); "
        "red rings: atoms that have lost a bond; frames between stored quasi-static states are interpolated. All statements are model results.")


# ----------------------------------------------------------------------------- data access
class Panel:
    def __init__(self, name, label, tile_below=60.0):
        recs = {r["name"]: r for r in db.all_records()}
        self.rec = recs[name]; self.name = name; self.label = label
        T = self.T = load_traj(os.path.join(ROOT, self.rec["trajectory"]))
        self.eps = T["eps_x"].astype(float); self.sig = T["sigma_xx"].astype(float) * EV; self.nb = T["n_broken_cum"].astype(int)
        assert np.all(np.diff(self.eps) >= -1e-12), f"{name}: stored strains not monotonic"
        self.L = np.array([[c[0, 0], c[1, 1]] for c in T["cells"]], float)
        P = T["positions"][:, :, :2].astype(float)
        self.S = (P / self.L[:, None, :]) % 1.0                                   # fractional coordinates per stored frame
        self.E = T["peratom_energy"].astype(float) - T["peratom_energy"][0].astype(float)
        V = T["peratom_virial"].astype(float)                                  # per-atom (xx, yy, xy) virial, eV
        area_per_atom = (self.L[:, 0] * self.L[:, 1] / P.shape[1])[:, None]     # A^2
        self.S1 = ((V[:, :, 0] + V[:, :, 1]) / 2 + np.sqrt(((V[:, :, 0] - V[:, :, 1]) / 2) ** 2 + V[:, :, 2] ** 2)) / area_per_atom * EV   # max principal 2D stress per atom, N/m
        self.n = len(self.eps); self.eps_max = float(self.eps[-1]); self.N = P.shape[1]
        ev_atoms = {}
        for e in self.rec.get("broken_bond_events", []):
            for a in (int(e[1]), int(e[2])): ev_atoms.setdefault(a, float(e[0]))
        self.break_strain = np.full(self.N, np.inf); 
        for a, e in ev_atoms.items(): self.break_strain[a] = e
        self.tile = 2 if self.L[0].max() < tile_below else 1                        # 2x2 periodic images for small cells
        self.side = float(self.L.max() * self.tile)                                  # fixed square view (cells stretch)
        ev = {}
        for lab, f in select_event_frames(T):
            for key in lab.split(" = "): ev[key] = f
        m = self.rec["metrics"]; self.smax = m["strength_Nm"]; self.ef = m["failure_strain"]
        self.first_damage = float(self.eps[ev["first_damage"]]) if (self.nb[-1] > self.nb[0] and "first_damage" in ev) else None
        self.peak_eps = float(self.eps[int(np.argmax(self.sig))])
        self.event_strains = sorted({float(self.eps[ev[k]]) for k in ("first_damage", "peak_load", "major_propagation") if k in ev} | ({self.first_damage} if self.first_damage is not None else set()) | {self.peak_eps, self.eps_max})

    def state(self, e, field="energy"):
        """Interpolated positions (A, wrapped, tiled), cell lengths, per-atom field (tiled), stress, broken bonds at strain e."""
        F = self.E if field == "energy" else self.S1
        e = min(max(e, 0.0), self.eps_max)
        i = int(np.searchsorted(self.eps, e, side="right") - 1); i = min(max(i, 0), self.n - 1)
        if i >= self.n - 1 or self.eps[i + 1] <= self.eps[i]:
            s, L, en, sg = self.S[i], self.L[i], F[i], self.sig[i]
        else:
            t = (e - self.eps[i]) / (self.eps[i + 1] - self.eps[i])
            ds = self.S[i + 1] - self.S[i]; ds -= np.round(ds)
            s = (self.S[i] + t * ds) % 1.0; L = self.L[i] + t * (self.L[i + 1] - self.L[i])
            en = F[i] + t * (F[i + 1] - F[i]); sg = self.sig[i] + t * (self.sig[i + 1] - self.sig[i])
        P = s * L
        if self.tile > 1:
            shifts = np.array([[a, b] for a in range(self.tile) for b in range(self.tile)], float) * L
            P = np.concatenate([P + sh for sh in shifts]); en = np.tile(en, self.tile ** 2); L = L * self.tile
        return P, L, en, sg, int(self.nb[i])

    def spark_alpha(self, e, decay=0.008):
        """Per-atom highlight for recently broken bonds: 1 at the moment of bond loss, fading with strain."""
        e = min(e, self.eps_max); a = np.exp(-(e - self.break_strain) / decay); a[self.break_strain > e + 1e-12] = 0.0
        return np.tile(np.clip(a, 0, 1), self.tile ** 2) if self.tile > 1 else np.clip(a, 0, 1)

    def bonds(self, P, L):
        tree = cKDTree(np.mod(P, L), boxsize=L)
        pairs = np.array(sorted(tree.query_pairs(R_BOND)), int)
        if len(pairs) == 0: return np.zeros((0, 2, 2))
        p0, p1 = P[pairs[:, 0]], P[pairs[:, 1]]; d = p1 - p0; d -= L * np.round(d / L)
        segs = np.stack([p0, p0 + d], 1)
        cross = np.any(np.abs((p1 - p0) - d) > 1e-6, axis=1)
        if cross.any():
            segs = np.concatenate([segs, np.stack([p1[cross], p1[cross] - d[cross]], 1)], 0)
        return segs

    def damaged(self, e):
        d = damaged_atoms_until(self.rec, min(e, self.eps_max))
        if len(d) and self.tile > 1:
            d = np.concatenate([d + k * self.N for k in range(self.tile ** 2)])
        return d


# ----------------------------------------------------------------------------- rendering
def layout(n, W, H):
    """Panel boxes, chart box and colorbar box (pixels from the top-left) for n panels; everything scales with H."""
    s = H / 1440.0
    m = 40 * s; g = 24 * s; title_h = 30 * s; foot_h = 30 * s; cbw = 60 * s
    if n == 1:
        w = min(H - title_h - foot_h, 1300 * s); boxes = [(m, title_h, w, w)]
        cbar = (m + w + 18 * s, title_h, 16 * s, w)
        chart = (m + w + 190 * s, title_h + 40 * s, W - (m + w + 190 * s) - 50 * s, w - 120 * s)
    else:
        w = min((W - 2 * m - cbw - (n - 1) * g) / n, 760 * s)
        x0 = (W - cbw - (n * w + (n - 1) * g)) / 2
        boxes = [(x0 + k * (w + g), title_h, w, w) for k in range(n)]
        cbar = (x0 + n * w + (n - 1) * g + 18 * s, title_h, 16 * s, w)
        top = title_h + w + 40 * s; ch = max(H - top - foot_h - 50 * s, 150 * s)
        chart = (m + 90 * s, top, W - 2 * m - 130 * s, ch)
    tofrac = lambda b: [b[0] / W, 1 - (b[1] + b[3]) / H, b[2] / W, b[3] / H]
    return [tofrac(b) for b in boxes], tofrac(chart), tofrac(cbar), w


def schedule(panels, fps, seconds_per_strain=110.0, min_seconds=15.0, slow=3.0, halfwin=0.0025):
    """Strain per frame: a uniform base rate (seconds_per_strain), slowed by `slow` within +-halfwin of any event strain."""
    e_max = max(p.eps_max for p in panels)
    de0 = e_max / (max(min_seconds, seconds_per_strain * e_max) * fps)
    ev = sorted({e for p in panels for e in p.event_strains if e < e_max - 1e-9})
    es, w = [0.0], [1.0]
    while es[-1] < e_max - 1e-12:
        e = es[-1]; wk = slow if any(abs(e - x) <= halfwin for x in ev) else 1.0
        es.append(min(e_max, e + de0 / wk)); w.append(wk)
    return np.array(es), np.array(w), de0, halfwin


def render(movie, outdir, width=2560, fps=30, seconds_per_strain=110.0, hold_start=1.5, hold_end=3.0, crf=17, preset="slow", poster_only=False):
    H = int(width * 9 / 16); dpi = 100; s = H / 1440.0
    panels = [Panel(n, lab) for n, lab in movie["panels"]]
    n = len(panels); boxes, cbox, cbarbox, wpx = layout(n, width, H)
    e_max = max(p.eps_max for p in panels)
    es, ws, de, halfwin = schedule(panels, fps, seconds_per_strain)
    strains = np.concatenate([np.zeros(int(hold_start * fps)), es, np.full(int(hold_end * fps), e_max)])
    weights = np.concatenate([np.ones(int(hold_start * fps)), ws, np.ones(int(hold_end * fps))])
    # colour scale shared by all panels: set by the pre-peak strain energy (97th percentile of the energy change per atom over the
    # frames up to peak load), combined across panels by the geometric mean so that no single panel dominates; crack faces and
    # strongly strained states beyond the range saturate (marked by the arrow on the colour bar)
    def prepeak(p):
        ip = int(np.argmax(p.sig)); return max(1e-3, max(np.percentile(p.E[k], 97) for k in range(ip + 1)))
    vmax = float(np.exp(np.mean([np.log(prepeak(p)) for p in panels]))); vmin = min(0.0, min(np.percentile(p.E[k], 0.5) for p in panels for k in range(p.n)))
    if vmax <= vmin: vmax = vmin + 1e-6
    saturates = max(np.percentile(p.E[k], 99.5) for p in panels for k in range(p.n)) > vmax * 1.05
    fig = plt.figure(figsize=(width / dpi, H / dpi), dpi=dpi); fig.patch.set_facecolor("white")
    colors = plt.cm.tab10(np.arange(n) % 10)
    fs = lambda x: x * s
    axc = fig.add_axes(cbox)
    for k, p in enumerate(panels):
        axc.plot(p.eps, p.sig, "-", color=colors[k], lw=2.6 * s)
    markers = [axc.plot([], [], "o", ms=14 * s, color=colors[k], markeredgecolor="k", markeredgewidth=1.2 * s, zorder=5)[0] for k in range(n)]
    vline = axc.axvline(0, color="0.5", lw=1.0 * s, ls="--")
    axc.set_xlim(0, e_max * 1.02); axc.set_ylim(0, max(p.sig.max() for p in panels) * 1.08)
    axc.set_xlabel("engineering strain", fontsize=fs(16)); axc.set_ylabel("2D stress (N/m)", fontsize=fs(16)); axc.tick_params(labelsize=fs(14)); axc.grid(alpha=0.2)
    for sp in ("top", "right"): axc.spines[sp].set_visible(False)
    arts = []
    for k, (p, box) in enumerate(zip(panels, boxes)):
        ax = fig.add_axes(box); ax.set_facecolor("white")
        P, L, en, sg, nbk = p.state(0.0)
        px_per_A = wpx / (p.side + 3.0); atom_pt = 0.95 * px_per_A * 72 / dpi; s_atom = atom_pt ** 2
        lc = LineCollection(p.bonds(P, L), colors="0.55", linewidths=max(0.6, 0.18 * px_per_A * 72 / dpi), zorder=1); ax.add_collection(lc)
        sc = ax.scatter(P[:, 0], P[:, 1], c=en, s=s_atom, cmap=CMAP, vmin=vmin, vmax=vmax, linewidths=0, zorder=2)
        dm = ax.scatter([], [], s=s_atom * 5, facecolors="none", edgecolors="red", linewidths=max(0.8, 0.25 * px_per_A * 72 / dpi), zorder=3)
        hs = p.side / 2 + 1.5; ax.set_xlim(L[0] / 2 - hs, L[0] / 2 + hs); ax.set_ylim(L[1] / 2 - hs, L[1] / 2 + hs); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(colors[k]); sp.set_linewidth(3 * s)
        arts.append(dict(ax=ax, lc=lc, sc=sc, dm=dm))
    cax = fig.add_axes(cbarbox); cb = fig.colorbar(arts[-1]["sc"], cax=cax, extend="max" if saturates else "neither")
    cb.set_label("energy change per atom (eV)", fontsize=fs(13)); cb.ax.tick_params(labelsize=fs(12))

    def update(e, slow=False):
        for k, (p, a) in enumerate(zip(panels, arts)):
            P, L, en, sg, nbk = p.state(e)
            a["lc"].set_segments(p.bonds(P, L)); a["sc"].set_offsets(P); a["sc"].set_array(en)
            dmg = p.damaged(e); a["dm"].set_offsets(P[dmg] if len(dmg) else np.zeros((0, 2)))
            hs = p.side / 2 + 1.5; a["ax"].set_xlim(L[0] / 2 - hs, L[0] / 2 + hs); a["ax"].set_ylim(L[1] / 2 - hs, L[1] / 2 + hs)
            ee = min(e, p.eps_max)
            markers[k].set_data([ee], [sg])
        vline.set_xdata([e, e])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{movie['id']}_{movie['slug']}")
    update(panels[0].peak_eps); fig.savefig(base + "_poster.png", dpi=dpi, facecolor="white")
    if poster_only:
        plt.close(fig); return base
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1, extra_args=["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    t0 = time.time()
    with writer.saving(fig, base + ".mp4", dpi=dpi):
        for e, w in zip(strains, weights):
            update(float(e), slow=w > 1.0); writer.grab_frame()
    plt.close(fig)
    meta = dict(id=movie["id"], file=os.path.basename(base) + ".mp4", title=movie["title"], caption=movie["insight"] + " " + FOOT,
                panels=[dict(name=p.name, label=p.label, run_id=p.rec["run_id"], strength_Nm=p.smax, failure_strain=p.ef, fracture_mode=p.rec["metrics"]["fracture_mode"], periodic_images=p.tile ** 2) for p in panels],
                resolution=f"{width}x{H}", fps=fps, seconds=round(len(strains) / fps, 1), strain_per_second=round(de * fps, 4), slow_motion="3x within ±0.0025 strain of first damage, peak load and major propagation",
                text_in_movie="chart axis labels and colour-bar numbers only", colour="energy change per atom relative to the relaxed state (eV), inferno colour map truncated at 0.9", colour_scale_eV=[round(vmin, 3), round(vmax, 3)],
                chart="2D stress (N/m) against engineering strain, one curve per panel in the panel's frame colour, moving marker at the current state", render_minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(base + ".json", "w"), indent=1)
    return base


def write_index(outdir):
    metas = []
    for m in MOVIES:
        p = os.path.join(outdir, f"{m['id']}_{m['slug']}.json")
        if os.path.exists(p): metas.append(json.load(open(p)))
    json.dump(metas, open(os.path.join(outdir, "captions.json"), "w"), indent=1)
    lines = ["# Fracture movies (supplementary material)\n",
             "MP4 (H.264, 2560x1440 masters at 30 fps; `*_1080p.mp4` are compact 1920x1080 versions for supplementary information). The movies contain no text other than the axis labels of the chart and the numbers on the colour bar; this file and `captions.json` carry the captions. Layout: the structures (left to right in the order of the table below; the frame colour of each structure is the colour of its curve) above a chart of 2D stress against engineering strain with a moving marker per structure and a dashed line at the current strain; the vertical bar at the right is the colour scale (an arrow means that values above the range saturate). All panels advance at the same engineering strain (uniform rate, slowed 3× around first damage, peak load and major propagation); a panel whose simulation has ended holds its final state. Frames between the stored quasi-static states are interpolated (positions, cell and per-atom energies; bonds recomputed with the 2.0 Å criterion), so intermediate configurations are interpolations, not simulated states. Colour: energy change per atom relative to the relaxed state (inferno colour map) on a scale shared by the panels of a movie and set by their pre-peak strain energy; red rings: atoms that have lost a bond; cells smaller than 60 Å are shown as 2×2 periodic images. Model results (screened REBO2, athermal quasi-static tension). Generated by `paper_analysis/make_movies.py` from the experiment database.\n"]
    for m in metas:
        lines.append(f"## {m['id']}: {m['title']}\n")
        lines.append(f"`{m['file']}` ({m['seconds']} s). {m['caption'].replace(FOOT, '').strip()}\n")
        lines.append(f"Colour scale {m['colour_scale_eV'][0]:.2f} to {m['colour_scale_eV'][1]:.2f} eV per atom.\n")
        lines.append("| panel (left to right) | design | run id | strength (N/m) | failure strain | fracture mode |\n|---|---|---|---|---|---|")
        for p in m["panels"]:
            lines.append(f"| {p['label']} | `{p['name']}` | `{p['run_id']}` | {p['strength_Nm']:.1f} | {p['failure_strain']:.3f} | {p['fracture_mode']} |")
        lines.append("")
    open(os.path.join(outdir, "README.md"), "w").write("\n".join(lines))
    tex = ["% Supplementary movie captions (generated by make_movies.py); Movie S<k> = M<k>", "\\section*{Supplementary movies}"]
    for k, m in enumerate(metas, 1):
        cap = m["caption"].replace("%", "\\%").replace("×", "$\\times$").replace("±", "$\\pm$").replace("Å", "\\AA{}").replace("°", "$^\\circ$").replace("φ", "$\\phi$").replace("−", "$-$")
        panels = "; ".join(f"{p['label']} (\\texttt{{{p['name'].replace('_', chr(92) + '_')}}})" for p in m["panels"]).replace("°", "$^\\circ$").replace("φ", "$\\phi$").replace("Å", "\\AA{}").replace("−", "$-$")
        lay = f" Panels from left to right: {panels}. Below the panels: 2D stress against engineering strain, one curve per panel in the colour of the panel frame, with a marker at the current state; colour scale {m['colour_scale_eV'][0]:.2f} to {m['colour_scale_eV'][1]:.2f} eV per atom."
        tex.append(f"\\paragraph{{Movie S{k} ({m['file'].replace('_', chr(92) + '_')}).}} {m['title'].replace('°', '$^\\circ$')}. {cap}{lay}")
    open(os.path.join(outdir, "si_movie_captions.tex"), "w").write("\n\n".join(tex) + "\n")
    return len(metas)


def compact(base, width=1920, crf=22, suffix="_1080p"):
    """Smaller version of a master movie (re-encode, no re-render)."""
    import subprocess
    src, dst = base + ".mp4", base + suffix + ".mp4"
    if not os.path.exists(src): return None
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)
    print(os.path.basename(dst), f"{os.path.getsize(dst) / 1e6:.0f} MB")
    return dst


# ----------------------------------------------------------------------------- clean single-structure movies (social media)
SOCIAL = [
 ("S1_pristine_zz", "pristine zigzag graphene: one avalanche at 27% strain"),
 ("S1_precrack_L20", "20 Å precrack: crack propagation at 9% strain"),
 ("S2_slit_0deg", "slit array parallel to the load: straight load paths, 25.4 N/m"),
 ("S7_slit_20deg", "slit array at 20°: en-echelon linking of overlapping slit tips, 8.9 N/m"),
 ("S2_slit_45deg", "slit array at 45°: ligaments rotate before they break"),
 ("P1_slit_60deg", "slit array at 60°: bridges between slits bend and fail"),
 ("S2_slit_90deg", "slit array perpendicular to the load, 4.0 N/m"),
 ("S2_H1_square_D40", "single-level square mesh, 17.0 N/m"),
 ("S2_H2_D40_W8", "nested mesh, veins both ways and round pores, 12.8 N/m"),
 ("S7_H2_veinsX_W12_ellipseX", "nested mesh with veins and pores along the load: the strongest nested design, 19.4 N/m"),
 ("S7_slit_0deg_armchair", "eight armchair strips failing row by row"),
 ("S2_voronoi_n25_reg0.6", "Voronoi network"),
 ("S2_hole20_rings2", "hole with two rings of pores"),
 ("S2_ellipse_90deg", "elongated pores along the load, 19.0 N/m"),
]
SOCIAL_BG = "#0b0b12"
CMAP_ENERGY_DARK = LinearSegmentedColormap.from_list("inferno_dark", plt.cm.inferno(np.linspace(0.12, 0.92, 256)))
CMAP_STRESS_DARK = LinearSegmentedColormap.from_list("plasma_dark", plt.cm.plasma(np.linspace(0.08, 0.95, 256)))


def render_social(name, outdir, field="energy", size=1440, fps=30, seconds_per_strain=80.0, hold_start=1.0, hold_end=2.0, crf=17, preset="slow", poster_only=False):
    """One structure, full frame, square, dark background, no chart, no text: atoms coloured by `field`
    ("energy": energy change per atom, eV; "stress": per-atom maximum principal 2D stress, N/m), bonds in grey,
    recently broken bonds flash as fading white rings.  Interpolation and slow motion as in the multi-panel movies."""
    p = Panel(name, name); dpi = 100; s = size / 1440.0
    es, ws, de, halfwin = schedule([p], fps, seconds_per_strain)
    strains = np.concatenate([np.zeros(int(hold_start * fps)), es, np.full(int(hold_end * fps), p.eps_max)])
    F = p.E if field == "energy" else p.S1; ip = int(np.argmax(p.sig))
    vmax = float(max(np.percentile(F[k], 97) for k in range(ip + 1))); vmin = 0.0
    if vmax <= vmin: vmax = vmin + 1e-6
    cmap = CMAP_ENERGY_DARK if field == "energy" else CMAP_STRESS_DARK
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi); fig.patch.set_facecolor(SOCIAL_BG)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94]); ax.set_facecolor(SOCIAL_BG); ax.axis("off")
    P, L, en, sg, nbk = p.state(0.0, field)
    px_per_A = 0.94 * size / (p.side + 3.0); atom_pt = 1.05 * px_per_A * 72 / dpi; s_atom = atom_pt ** 2
    lc = LineCollection(p.bonds(P, L), colors="#8a8a96", linewidths=max(0.8, 0.22 * px_per_A * 72 / dpi), zorder=1); ax.add_collection(lc)
    sc = ax.scatter(P[:, 0], P[:, 1], c=en, s=s_atom, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2)
    spark = ax.scatter(P[:, 0], P[:, 1], s=s_atom * 6, facecolors="none", edgecolors=np.zeros((len(P), 4)), linewidths=max(1.2, 0.3 * px_per_A * 72 / dpi), zorder=3)
    ax.set_aspect("equal")

    def update(e):
        P, L, en, sg, nbk = p.state(e, field)
        lc.set_segments(p.bonds(P, L)); sc.set_offsets(P); sc.set_array(en)
        a = p.spark_alpha(e); rgba = np.zeros((len(P), 4)); rgba[:, :3] = 1.0; rgba[:, 3] = a; spark.set_offsets(P); spark.set_edgecolor(rgba)
        hs = p.side / 2 + 1.5; ax.set_xlim(L[0] / 2 - hs, L[0] / 2 + hs); ax.set_ylim(L[1] / 2 - hs, L[1] / 2 + hs)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{name}_{field}")
    update(p.peak_eps); fig.savefig(base + "_poster.png", dpi=dpi, facecolor=SOCIAL_BG)
    if poster_only:
        plt.close(fig); return base
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1, extra_args=["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    t0 = time.time()
    with writer.saving(fig, base + ".mp4", dpi=dpi):
        for e in strains:
            update(float(e)); writer.grab_frame()
    plt.close(fig)
    meta = dict(file=os.path.basename(base) + ".mp4", design=name, run_id=p.rec["run_id"], description=dict(SOCIAL).get(name, ""), field=("energy change per atom relative to the relaxed state (eV)" if field == "energy" else "per-atom maximum principal 2D stress (N/m)"),
                colour_scale=[round(vmin, 3), round(vmax, 3)], highlight="white rings: atoms that just lost a bond, fading with strain", periodic_images=p.tile ** 2,
                strength_Nm=p.smax, failure_strain=p.ef, fracture_mode=p.rec["metrics"]["fracture_mode"], resolution=f"{size}x{size}", fps=fps, seconds=round(len(strains) / fps, 1),
                slow_motion="3x within ±0.0025 strain of first damage, peak load and major propagation", interpolated="frames between stored quasi-static states are interpolated", text_in_movie=False, render_minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(base + ".json", "w"), indent=1)
    return base


def write_social_index(outdir):
    metas = []
    for name, _ in SOCIAL:
        for field in ("energy", "stress"):
            pth = os.path.join(outdir, f"{name}_{field}.json")
            if os.path.exists(pth): metas.append(json.load(open(pth)))
    json.dump(metas, open(os.path.join(outdir, "captions.json"), "w"), indent=1)
    lines = ["# Clean single-structure fracture movies\n",
             "Square MP4 (H.264, 1440x1440 masters at 30 fps; `*_1080.mp4` are 1080x1080 versions), no text, no chart, dark background: one structure under uniaxial tension along the horizontal axis, atoms coloured by a per-atom field (`_energy`: energy change per atom relative to the relaxed state, inferno colour map; `_stress`: maximum principal 2D stress per atom, plasma colour map), bonds in grey, atoms that just lost a bond flash as fading white rings. Uniform strain rate with 3× slow motion around first damage, peak load and major propagation; frames between the stored quasi-static states are interpolated (positions, cell, field; bonds recomputed with the 2.0 Å criterion) and are not simulated states; cells smaller than 60 Å are shown as 2×2 periodic images. Model results (screened REBO2, athermal quasi-static tension). Generated by `paper_analysis/make_movies.py --social`.\n",
             "| file | design | run id | field | colour scale | strength (N/m) | failure strain | fracture mode |\n|---|---|---|---|---|---|---|---|"]
    for m in metas:
        lines.append(f"| `{m['file']}` | {m['description']} | `{m['run_id']}` | {m['field']} | {m['colour_scale'][0]:.2f} to {m['colour_scale'][1]:.2f} | {m['strength_Nm']:.1f} | {m['failure_strain']:.3f} | {m['fracture_mode']} |")
    open(os.path.join(outdir, "README.md"), "w").write("\n".join(lines) + "\n")
    return len(metas)


def _job_social(args):
    name, field, outdir, kw = args
    try:
        base = render_social(name, outdir, field, **kw); return f"{name}_{field}", os.path.basename(base), None
    except Exception:
        import traceback; return f"{name}_{field}", None, traceback.format_exc()


def _job(args):
    movie, outdir, kw = args
    try:
        base = render(movie, outdir, **kw); return movie["id"], os.path.basename(base), None
    except Exception:
        import traceback; return movie["id"], None, traceback.format_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--all", action="store_true"); ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--outdir", default=OUT_DEFAULT); ap.add_argument("--width", type=int, default=2560); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds-per-strain", type=float, default=110.0, help="playback: seconds per unit engineering strain (before slow motion)")
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--crf", type=int, default=17); ap.add_argument("--preset", default="slow")
    ap.add_argument("--poster-only", action="store_true"); ap.add_argument("--compact", action="store_true", help="also write 1080p versions (crf 22) of the rendered masters")
    ap.add_argument("--social", action="store_true", help="render the clean single-structure series instead (see SOCIAL)"); ap.add_argument("--social-fields", nargs="*", default=["energy", "stress"])
    ap.add_argument("--social-only", nargs="*", default=[], help="subset of design names for --social")
    a = ap.parse_args()
    if a.social:
        names = [n for n, _ in SOCIAL if not a.social_only or n in a.social_only]
        jobs = [(n, f, a.outdir, dict(size=a.width if a.width != 2560 else 1440, fps=a.fps, seconds_per_strain=a.seconds_per_strain if a.seconds_per_strain != 110.0 else 80.0, crf=a.crf, preset=a.preset, poster_only=a.poster_only)) for n in names for f in a.social_fields]
        if a.jobs > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.jobs) as pool:
                for mid, f, err in pool.imap_unordered(_job_social, jobs):
                    print(mid, f or "FAILED", flush=True)
                    if err: print(err, flush=True)
        else:
            for j in jobs:
                mid, f, err = _job_social(j); print(mid, f or "FAILED", flush=True)
                if err: print(err)
        if a.compact and not a.poster_only:
            for n in names:
                for f in a.social_fields: compact(os.path.join(a.outdir, f"{n}_{f}"), width=1080, suffix="_1080")
        print("index:", write_social_index(a.outdir), "movies")
        sys.exit()
    if a.list:
        for m in MOVIES: print(m["id"], m["slug"], "|", ", ".join(nm for nm, _ in m["panels"]))
        sys.exit()
    sel = [m for m in MOVIES if a.all or m["id"] in a.only]
    kw = dict(width=a.width, fps=a.fps, seconds_per_strain=a.seconds_per_strain, crf=a.crf, preset=a.preset, poster_only=a.poster_only)
    if a.jobs > 1 and len(sel) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for mid, f, err in pool.imap_unordered(_job, [(m, a.outdir, kw) for m in sel]):
                print(mid, f or "FAILED", flush=True)
                if err: print(err, flush=True)
    else:
        for m in sel:
            t0 = time.time(); base = render(m, a.outdir, **kw); print(m["id"], os.path.basename(base), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if a.compact and not a.poster_only:
        for m in sel: compact(os.path.join(a.outdir, f"{m['id']}_{m['slug']}"))
    print("index:", write_index(a.outdir), "movies")
