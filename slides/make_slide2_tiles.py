"""Tiles for talk slide 2 (mechanisms): each tile = hand-drawn schematic of the idea (left) + real simulation
snapshot(s) from the experiment database (right).  Writes slides/tiles/tile{1..4}.png (300 dpi) and .svg."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch, Polygon
from matplotlib.collections import LineCollection
from experiments import db
from analysis.fracture_viz import load_traj, damaged_atoms_until

OUT = os.path.join(HERE, "tiles"); os.makedirs(OUT, exist_ok=True)
GREY, DARK, RED, ACC = "#d8d8d8", "#222222", "#c0392b", "#1f77b4"
EV = 16.0217663
REC = {r["name"]: r for r in db.all_records()}


# --------------------------------------------------------------------------- simulation snapshots
def snapshot(ax, name, frame, title, s_atom=None, lw=None):
    rec = REC[name]; T = load_traj(os.path.join(ROOT, rec["trajectory"]))
    k = frame if isinstance(frame, int) else frame(T)
    P = T["positions"][k].astype(float); c = T["cells"][k]; b = T["bonds"][k]
    L = max(c[0, 0], c[1, 1])
    if s_atom is None: s_atom = (60.0 / L) ** 2 * 2.2
    if lw is None: lw = 22.0 / L
    if len(b):
        p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]; d = p1 - p0
        for a in range(2):
            d[:, a] -= c[a, a] * np.round(d[:, a] / c[a, a])
        ax.add_collection(LineCollection(np.stack([p0, p0 + d], 1), colors="0.45", linewidths=lw, zorder=1))
    ax.scatter(P[:, 0], P[:, 1], s=s_atom, c="k", linewidths=0, zorder=2)
    dmg = damaged_atoms_until(rec, T["eps_x"][k])
    if len(dmg):
        ax.scatter(P[dmg, 0], P[dmg, 1], s=s_atom * 9, facecolors="none", edgecolors=RED, linewidths=0.7, zorder=3)
    ax.set_xlim(-1.5, c[0, 0] + 1.5); ax.set_ylim(-1.5, c[1, 1] + 1.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.6"); sp.set_linewidth(0.6)
    eps, sig = T["eps_x"][k], T["sigma_xx"][k] * EV
    ax.set_title(title.format(eps=eps, sig=sig, smax=rec["metrics"]["strength_Nm"]), fontsize=6.3, pad=2.2, color=DARK)
    return T, k


def avalanche_frame(T):
    nb = T["n_broken_cum"]; return int(np.argmax(nb >= 0.5 * nb.max()))


# --------------------------------------------------------------------------- schematic helpers
def sheet(ax, x, y, w, h, fc=GREY):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none", zorder=1))


def load_arrows(ax, x0, x1, y, size=8, gap=0.12, length=0.75, label=None):
    ax.annotate("", xy=(x0 - gap - length, y), xytext=(x0 - gap, y), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2, mutation_scale=size))
    ax.annotate("", xy=(x1 + gap + length, y), xytext=(x1 + gap, y), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2, mutation_scale=size))
    if label:
        ax.text(x1 + gap + length / 2, y + 0.22, label, fontsize=7, ha="center", va="bottom", color=DARK, style="italic")


def slit(ax, cx, cy, L, w, ang, clip, fc="white"):
    r = Rectangle((cx - L / 2, cy - w / 2), L, w, angle=ang, rotation_point=(cx, cy), fc=fc, ec="none", zorder=2)
    r.set_clip_path(clip); ax.add_patch(r); return r


def rot(p, c, ang):
    a = np.radians(ang); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); return c + R @ (np.asarray(p) - c)


def slit_array(ax, x, y, w, h, ang, rows=3, per=1.35, L=0.95, sw=0.16, echelon=False):
    clip = Rectangle((x, y), w, h, fc=GREY, ec="none", zorder=1); ax.add_patch(clip)
    c = np.array([x + w / 2, y + h / 2]); tips = {}
    dy = h / (rows + 1)
    for j in range(rows):
        cy = y + dy * (j + 1); off = 0.5 * per * (j % 2)
        for i in range(-3, 5):
            cx = x - 0.4 + off + i * per
            p = rot((cx, cy), c, ang)
            if not (x - 0.6 < p[0] < x + w + 0.6 and y - 0.6 < p[1] < y + h + 0.6): continue
            slit(ax, p[0], p[1], L, sw, ang, clip)
            tips.setdefault(j, []).append((rot((cx - L / 2, cy), c, ang), rot((cx + L / 2, cy), c, ang)))
    if echelon:
        n = 0
        for j in range(rows - 1):
            for (a0, a1) in tips.get(j, []):
                best = None
                for (b0, b1) in tips.get(j + 1, []):
                    d = np.hypot(*(b0 - a1))
                    if best is None or d < best[0]: best = (d, b0)
                if best and best[0] < 0.9 and x < a1[0] < x + w and y < best[1][1] < y + h:
                    ax.plot([a1[0], best[1][0]], [a1[1], best[1][1]], color=RED, lw=1.5, ls=(0, (2, 1.2)), zorder=4); n += 1
    return clip


def mesh(ax, x, y, w, h, veins_x=False, veins_y=False, elong=False, n=5, vein=0.28):
    sheet(ax, x, y, w, h)
    cell = w / n
    for k in (1, 3):
        if veins_y: ax.add_patch(Rectangle((x + cell * k, y), cell, h, fc="#bdbdbd", ec="none", zorder=1.5))
        if veins_x: ax.add_patch(Rectangle((x, y + cell * k), w, cell, fc="#bdbdbd", ec="none", zorder=1.5))
    for i in range(n):
        for j in range(n):
            cx, cy = x + cell * (i + 0.5), y + cell * (j + 0.5)
            if veins_y and i in (1, 3): continue
            if veins_x and j in (1, 3): continue
            if elong:
                ax.add_patch(Ellipse((cx, cy), cell * 0.78, cell * 0.36, fc="white", ec="none", zorder=2))
            else:
                ax.add_patch(Circle((cx, cy), cell * 0.27, fc="white", ec="none", zorder=2))


def ligaments(ax, x, y, w, h, n, broken, gapw=0.35, arrows=1):
    """Horizontal ligaments (load along x); one ligament broken in the middle; arrows show transferred load."""
    sheet(ax, x, y, w, h, fc="white")
    t = h / (2 * n - 1)  # ligament thickness = slit thickness
    ys = []
    for j in range(n):
        yy = y + 2 * j * t
        ax.add_patch(Rectangle((x, yy), w, t, fc=GREY, ec="none", zorder=2)); ys.append(yy + t / 2)
    jb = broken; yb = y + 2 * jb * t
    ax.add_patch(Rectangle((x + w / 2 - gapw / 2, yb - 0.02), gapw, t + 0.04, fc="white", ec="none", zorder=3))
    ax.plot([x + w / 2 - gapw / 2, x + w / 2 - gapw / 2], [yb, yb + t], color=RED, lw=1.2, zorder=4)
    ax.plot([x + w / 2 + gapw / 2, x + w / 2 + gapw / 2], [yb, yb + t], color=RED, lw=1.2, zorder=4)
    for jj in (jb - 1, jb + 1):
        if 0 <= jj < n:
            y1 = ys[jj]; ms = 7 if arrows == 1 else 10
            ax.annotate("", xy=(x + w / 2, y1), xytext=(x + w / 2, ys[jb]), arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9 * arrows, mutation_scale=ms), zorder=5)
    return ys


def blank(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")


def label(ax, x, y, s, size=7.2, **kw):
    ax.text(x, y, s, fontsize=size, ha=kw.pop("ha", "center"), va=kw.pop("va", "top"), color=kw.pop("color", DARK), **kw)


def new_tile():
    fig = plt.figure(figsize=(4.0, 2.2), dpi=300)
    return fig


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- tile 1: net section x ligament strength
def tile1():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.55, 1.0]); blank(ax)
    x, y, w, h = 1.0, 1.05, 4.1, 3.7
    sheet(ax, x, y, w, h)
    ax.add_patch(Rectangle((2.2, 2.15), 1.3, 1.45, fc="white", ec="none", zorder=2))
    ax.add_patch(Circle((4.35, 3.9), 0.4, fc="white", ec="none", zorder=2))
    ax.add_patch(Circle((4.25, 1.95), 0.45, fc="white", ec="none", zorder=2))
    ax.add_patch(Ellipse((1.7, 4.0), 0.8, 0.48, fc="white", ec="none", zorder=2))
    load_arrows(ax, x, x + w, y + h / 2, size=7, gap=0.1, length=0.55)
    xs = 2.85
    ax.plot([xs, xs], [y - 0.22, y + h + 0.22], color=RED, lw=1.3, ls=(0, (3, 1.5)), zorder=5)
    ax.annotate("", xy=(xs + 0.25, y + h), xytext=(xs + 0.25, 3.6), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0, mutation_scale=7), zorder=6)
    ax.annotate("", xy=(xs + 0.25, y), xytext=(xs + 0.25, 2.15), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0, mutation_scale=7), zorder=6)
    label(ax, xs + 0.42, 4.2, "w", size=8.5, ha="left", va="center", color=RED, style="italic")
    label(ax, xs + 0.42, 1.6, "w", size=8.5, ha="left", va="center", color=RED, style="italic")
    label(ax, xs, y - 0.32, "weakest section", size=7, color=RED, va="top")
    label(ax, 4.9, 5.55, "$\\sigma_\\mathrm{max} = (A_\\mathrm{min}/A)\\;\\sigma_\\mathrm{lig}(w,\\ \\mathrm{chirality})$", size=8.2, va="center")
    label(ax, 7.55, 4.25, "$\\sigma_\\mathrm{lig}$", size=8, va="center", ha="center")
    label(ax, 7.55, 3.75, "$w>2$ nm: 30–32 N/m\n$w\\approx1$ nm: 17–19 N/m\narmchair: $\\times$0.8", size=6.3, va="top", ha="center", linespacing=1.35)
    label(ax, 7.55, 1.75, "pore shape:\nno effect", size=6.3, va="top", ha="center", linespacing=1.3)
    ax1 = fig.add_axes([0.575, 0.11, 0.195, 0.70]); snapshot(ax1, "S4_square_D40", 0, "coarse, $\\phi$=0.20\n{smax:.1f} N/m")
    ax2 = fig.add_axes([0.795, 0.11, 0.195, 0.70]); snapshot(ax2, "S2_H1_p16", 0, "fine, $\\phi$=0.21\n{smax:.1f} N/m")
    fig.text(0.785, 0.03, "same mass, wider ligaments: stronger", fontsize=6.6, ha="center", va="bottom", color=DARK, style="italic")
    save(fig, "tile1")


# --------------------------------------------------------------------------- tile 2: alignment cliff
def tile2():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.58, 1.0]); blank(ax)
    W, H, y0 = 2.5, 2.7, 1.75
    xs = [0.85, 3.75, 6.65]
    for k, (ang, xx, val, ech) in enumerate([(0, xs[0], "25 N/m", False), (20, xs[1], "8.9 N/m", True), (90, xs[2], "4.0 N/m", False)]):
        slit_array(ax, xx, y0, W, H, ang, echelon=ech)
        label(ax, xx + W / 2, y0 + H + 0.14, f"{ang}°", size=9, va="bottom", fontweight="bold")
        label(ax, xx + W / 2, y0 - 0.16, val, size=8.5, va="top", color=RED if ech else DARK, fontweight="bold" if ech else "normal")
    load_arrows(ax, xs[0], xs[2] + W, y0 + H / 2, size=7, gap=0.1, length=0.55)
    label(ax, 5.0, 0.62, "20°: tips of adjacent rows overlap and link (red)", size=6.8, va="top", color=RED)
    ax1 = fig.add_axes([0.605, 0.11, 0.385, 0.78]); snapshot(ax1, "S7_slit_20deg", 26, "20° slits, $\\varepsilon$={eps:.2f}")
    save(fig, "tile2")


# --------------------------------------------------------------------------- tile 3: hierarchy pays only when aligned
def tile3():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.58, 1.0]); blank(ax)
    S, y0 = 2.8, 1.75
    xl, xr = 1.0, 6.1
    mesh(ax, xl, y0, S, S, veins_x=True, veins_y=True, elong=False)
    label(ax, xl + S / 2, y0 + S + 0.14, "veins both ways\nround fine pores", size=6.8, va="bottom", linespacing=1.2)
    label(ax, xl + S / 2, y0 - 0.16, "12.8 N/m", size=8.5, va="top")
    mesh(ax, xr, y0, S, S, veins_x=True, veins_y=False, elong=True)
    label(ax, xr + S / 2, y0 + S + 0.14, "veins along load\nelongated fine pores", size=6.5, va="bottom", linespacing=1.2)
    label(ax, xr + S / 2, y0 - 0.16, "19.4 N/m", size=8.5, va="top", color=RED, fontweight="bold")
    load_arrows(ax, xl, xr + S, y0 + S / 2, size=7, gap=0.1, length=0.55)
    ax.annotate("", xy=(xr - 0.3, y0 + S / 2), xytext=(xl + S + 0.3, y0 + S / 2), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2, mutation_scale=9))
    label(ax, (xl + S + xr) / 2, y0 + S / 2 + 0.22, "align\nboth levels", size=6.6, va="bottom", linespacing=1.2)
    label(ax, 5.0, 0.72, "veins only 16.1 · elongated pores only 17.5\nboth 19.4 · single-level mesh 17.0", size=6.3, va="top", linespacing=1.3)
    ax1 = fig.add_axes([0.605, 0.11, 0.385, 0.78]); snapshot(ax1, "S7_H2_veinsX_W12_ellipseX", 18, "post-peak: veins carry {sig:.0f} N/m")
    save(fig, "tile3")


# --------------------------------------------------------------------------- tile 4: fibre-bundle load transfer
def tile4():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.54, 1.0]); blank(ax)
    W, H, y0 = 3.1, 3.1, 1.7
    xl, xr = 1.1, 5.8
    ligaments(ax, xl, y0, W, H, n=8, broken=3, arrows=1)
    label(ax, xl + W / 2, y0 + H + 0.14, "many ligaments", size=7.2, va="bottom")
    label(ax, xl + W / 2, y0 - 0.16, "small share:\nrow by row", size=6.2, va="top", linespacing=1.25)
    ligaments(ax, xr, y0, W, H, n=3, broken=1, gapw=0.45, arrows=1.5)
    label(ax, xr + W / 2, y0 + H + 0.14, "few wide strips", size=7.2, va="bottom")
    label(ax, xr + W / 2, y0 - 0.16, "large share:\none avalanche", size=6.2, va="top", linespacing=1.25, color=RED)
    load_arrows(ax, xl, xr + W, y0 + H / 2, size=7, gap=0.1, length=0.55)
    ax1 = fig.add_axes([0.555, 0.11, 0.19, 0.72]); snapshot(ax1, "S7_slit_0deg_armchair", 20, "row by row\n$\\varepsilon$={eps:.2f}, {sig:.0f}/{smax:.0f} N/m")
    ax2 = fig.add_axes([0.80, 0.11, 0.19, 0.72]); snapshot(ax2, "S4_square_D40_dispersed", avalanche_frame, "avalanche\n$\\varepsilon$={eps:.2f} (3 rows)")
    save(fig, "tile4")


if __name__ == "__main__":
    tile1(); tile2(); tile3(); tile4(); print("tiles written to", OUT)
