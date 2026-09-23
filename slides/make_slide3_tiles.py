"""Tiles for talk slide 3: each mechanism as a pure schematic (left) next to one simple chart of real
campaign data (right).  Writes slides/tiles3/tile{1..4}.png (300 dpi) and .svg.  All numbers are read
from the experiment database (no hard-coded values)."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from analysis.campaign_analysis import curve, M
import make_slide2_tiles as S  # schematic helpers + database records (REC)

OUT = os.path.join(HERE, "tiles3"); os.makedirs(OUT, exist_ok=True)
DARK, RED, GREY = S.DARK, S.RED, S.GREY
BLUE, ORANGE, HIER = "#1f77b4", "#ff7f0e", "#d62728"


def strength(name):
    return S.REC[name]["metrics"]["strength_Nm"]


def new_tile():
    return plt.figure(figsize=(6.0, 2.35), dpi=300)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), facecolor="white")
    plt.close(fig)


def style(ax, ylabel="2D strength (N/m)"):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.7); ax.spines[sp].set_color("0.3")
    ax.tick_params(labelsize=7, length=2.5, width=0.6, colors="0.25")
    ax.set_ylabel(ylabel, fontsize=7.5, color=DARK)


def bars(ax, groups, colors, ylim=None, group_gap=0.55):
    """groups: list of (label, [(bar_label, value, color, bold)])."""
    x = 0.0; ticks, ticklabels = [], []
    for glabel, items in groups:
        xs = []
        for (bl, v, col, bold) in items:
            ax.bar(x, v, width=0.8, color=col, edgecolor="none", zorder=2)
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold" if bold else "normal", color=RED if bold else DARK)
            if bl: ax.text(x, -0.9, bl, ha="center", va="top", fontsize=6.4, color="0.3")
            xs.append(x); x += 1.0
        ticks.append(np.mean(xs)); ticklabels.append(glabel); x += group_gap
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=7, color=DARK)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.set_xlim(-0.7, x - group_gap - 0.3)
    if ylim: ax.set_ylim(*ylim)
    ax.yaxis.grid(True, alpha=0.25, lw=0.5); ax.set_axisbelow(True)


# --------------------------------------------------------------------------- tile 1: weakest section x ligament strength
def tile1():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.43, 1.0]); S.blank(ax)
    x, y, w, h = 1.6, 0.9, 4.4, 3.8
    S.sheet(ax, x, y, w, h)
    ax.add_patch(Rectangle((2.9, 2.0), 1.35, 1.5, fc="white", ec="none", zorder=2))
    ax.add_patch(Circle((5.15, 3.9), 0.42, fc="white", ec="none", zorder=2))
    ax.add_patch(Circle((5.05, 1.85), 0.48, fc="white", ec="none", zorder=2))
    ax.add_patch(Ellipse((2.3, 3.95), 0.85, 0.5, fc="white", ec="none", zorder=2))
    S.load_arrows(ax, x, x + w, y + h / 2, size=8, gap=0.1, length=0.7, label="$\\sigma$")
    xs = 3.57
    ax.plot([xs, xs], [y - 0.22, y + h + 0.22], color=RED, lw=1.4, ls=(0, (3, 1.5)), zorder=5)
    ax.annotate("", xy=(xs + 0.25, y + h), xytext=(xs + 0.25, 3.5), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0, mutation_scale=7), zorder=6)
    ax.annotate("", xy=(xs + 0.25, y), xytext=(xs + 0.25, 2.0), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0, mutation_scale=7), zorder=6)
    S.label(ax, xs + 0.42, 4.1, "w", size=9, ha="left", va="center", color=RED, style="italic")
    S.label(ax, xs + 0.42, 1.45, "w", size=9, ha="left", va="center", color=RED, style="italic")
    S.label(ax, xs, y - 0.32, "weakest section $A_\\mathrm{min}$", size=7.5, color=RED, va="top")
    S.label(ax, 5.0, 5.55, "$\\sigma_\\mathrm{max} = (A_\\mathrm{min}/A)\\;\\sigma_\\mathrm{lig}(w,\\ \\mathrm{chirality})$", size=8.5, va="center")
    S.label(ax, 8.75, 3.1, "$\\sigma_\\mathrm{lig}$ falls for\nnarrow ligaments\n(edge atoms) and\nfor armchair\nligaments;\npore shape:\nno effect", size=6.2, va="center", ha="center", linespacing=1.3)
    axc = fig.add_axes([0.53, 0.24, 0.45, 0.66]); style(axc)
    groups = [("ligament width\n(same porosity)", [("20 Å", strength("S4_square_D40"), BLUE, False), ("10 Å", strength("S2_H1_p16"), BLUE, False)]),
              ("pore shape\n(same net section)", [("square", strength("S4_square_D40"), BLUE, False), ("round", strength("S4_circle_D40_sameMinSF"), BLUE, False)]),
              ("lattice orientation\n(parallel slits)", [("zigzag", strength("S2_slit_0deg"), ORANGE, False), ("armchair", strength("S7_slit_0deg_armchair"), ORANGE, False)])]
    bars(axc, groups, None, ylim=(0, 31))
    save(fig, "tile1")


# --------------------------------------------------------------------------- tile 2: alignment cliff
def tile2():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.46, 1.0]); S.blank(ax)
    W, H, y0 = 2.5, 2.6, 1.7
    xs = [0.75, 3.75, 6.75]
    for ang, xx, ech in [(0, xs[0], False), (20, xs[1], True), (90, xs[2], False)]:
        S.slit_array(ax, xx, y0, W, H, ang, echelon=ech)
        S.label(ax, xx + W / 2, y0 + H + 0.14, f"{ang}°", size=9.5, va="bottom", fontweight="bold", color=RED if ech else DARK)
    S.load_arrows(ax, xs[0], xs[2] + W, y0 + H / 2, size=8, gap=0.1, length=0.55, label="$\\sigma$")
    S.label(ax, 5.0, 1.25, "tilt the slits: tips of adjacent rows overlap,\nthe slivers between them link up (red)", size=6.8, va="top", color=RED, linespacing=1.3)
    axc = fig.add_axes([0.55, 0.2, 0.43, 0.7]); style(axc)
    angs = [0, 20, 45, 90]; names = ["S2_slit_0deg", "S7_slit_20deg", "S2_slit_45deg", "S2_slit_90deg"]
    ys = [strength(n) for n in names]
    axc.plot(angs, ys, "--", color="0.55", lw=0.9, zorder=1)
    axc.scatter(angs, ys, s=[40, 60, 40, 40], color=[ORANGE, RED, ORANGE, ORANGE], edgecolor="white", linewidths=0.6, zorder=3)
    for a, v in zip(angs, ys):
        axc.text(a + (2.5 if a < 80 else -2.5), v + 1.2, f"{v:.1f}", fontsize=7.5, ha="left" if a < 80 else "right", va="bottom", fontweight="bold" if a == 20 else "normal", color=RED if a == 20 else DARK)
    axc.annotate("cliff", xy=(20, ys[1]), xytext=(31, 5.5), fontsize=7.5, color=RED, ha="left", arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    axc.set_xticks(angs); axc.set_xticklabels([f"{a}°" for a in angs]); axc.set_xlabel("slit angle to the load", fontsize=7.5, color=DARK)
    axc.set_ylim(0, 30); axc.set_xlim(-6, 96); axc.yaxis.grid(True, alpha=0.25, lw=0.5); axc.set_axisbelow(True)
    axc.text(0.98, 0.95, "same porosity (0.19–0.21)", transform=axc.transAxes, fontsize=6.4, ha="right", va="top", color="0.4", style="italic")
    save(fig, "tile2")


# --------------------------------------------------------------------------- tile 3: hierarchy pays only when aligned
def tile3():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.46, 1.0]); S.blank(ax)
    Sz, y0 = 2.8, 1.75
    xl, xr = 1.3, 5.9
    S.mesh(ax, xl, y0, Sz, Sz, veins_x=True, veins_y=True, elong=False)
    S.label(ax, xl + Sz / 2, y0 + Sz + 0.14, "unaligned: veins both ways,\nround pores", size=6.2, va="bottom", linespacing=1.2)
    S.mesh(ax, xr, y0, Sz, Sz, veins_x=True, veins_y=False, elong=True)
    S.label(ax, xr + Sz / 2, y0 + Sz + 0.14, "aligned: veins along load,\nelongated pores", size=6.2, va="bottom", linespacing=1.2, color=RED)
    S.load_arrows(ax, xl, xr + Sz, y0 + Sz / 2, size=8, gap=0.1, length=0.55, label="$\\sigma$")
    ax.annotate("", xy=(xr - 0.3, y0 + Sz / 2), xytext=(xl + Sz + 0.3, y0 + Sz / 2), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2, mutation_scale=9))
    S.label(ax, (xl + Sz + xr) / 2, y0 + Sz / 2 + 0.22, "align each level\nwith the load", size=6.6, va="bottom", linespacing=1.2)
    S.label(ax, 5.0, 1.2, "same porosity for all five designs (0.20)", size=6.6, va="top", color="0.4", style="italic")
    axc = fig.add_axes([0.55, 0.24, 0.43, 0.66]); style(axc)
    items = [("single\nlevel", strength("S2_H1_square_D40"), BLUE, False), ("nested,\nunaligned", strength("S2_H2_D40_W8"), "#e8a0a0", False),
             ("veins\naligned", strength("S4_H2_D40_W8_veins_x"), "#e07070", False), ("pores\naligned", strength("S5_H2_D40_W8_ellipseX"), "#d84a4a", False),
             ("both\naligned", strength("S7_H2_veinsX_W12_ellipseX"), HIER, True)]
    for i, (bl, v, col, bold) in enumerate(items):
        axc.bar(i, v, width=0.72, color=col, edgecolor="none", zorder=2)
        axc.text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold" if bold else "normal", color=RED if bold else DARK)
    axc.axhline(strength("S2_H1_square_D40"), color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axc.text(0.02, 0.97, "dashed line: single-level mesh", transform=axc.transAxes, fontsize=6.2, color=BLUE, ha="left", va="top")
    axc.set_xticks(range(5)); axc.set_xticklabels([it[0] for it in items], fontsize=6.8, color=DARK); axc.tick_params(axis="x", length=0, pad=3)
    axc.set_ylim(0, 23); axc.yaxis.grid(True, alpha=0.25, lw=0.5); axc.set_axisbelow(True)
    save(fig, "tile3")


# --------------------------------------------------------------------------- tile 4: fracture mode = load transfer
def tile4():
    fig = new_tile(); ax = fig.add_axes([0.0, 0.0, 0.46, 1.0]); S.blank(ax)
    W, H, y0 = 3.1, 3.1, 1.6
    xl, xr = 1.0, 5.9
    S.ligaments(ax, xl, y0, W, H, n=8, broken=3, arrows=1)
    S.label(ax, xl + W / 2, y0 + H + 0.14, "many parallel ligaments", size=7, va="bottom")
    S.label(ax, xl + W / 2, y0 - 0.16, "small share transferred\n→ next row holds", size=6.5, va="top", linespacing=1.25)
    S.ligaments(ax, xr, y0, W, H, n=3, broken=1, gapw=0.45, arrows=1.5)
    S.label(ax, xr + W / 2, y0 + H + 0.14, "few wide strips", size=7, va="bottom")
    S.label(ax, xr + W / 2, y0 - 0.16, "large share transferred\n→ next strip fails too", size=6.5, va="top", linespacing=1.25, color=RED)
    S.load_arrows(ax, xl, xr + W, y0 + H / 2, size=8, gap=0.1, length=0.55, label="$\\sigma$")
    axc = fig.add_axes([0.55, 0.2, 0.43, 0.7]); style(axc, ylabel="2D stress (N/m)")
    for name, col, lab, dy in [("S7_slit_0deg_armchair", ORANGE, "8 strips: row by row,\nload retained", 1.0), ("S4_square_D40_dispersed", BLUE, "3 wide rows (15% dispersion):\none avalanche", -1.0)]:
        e, s, nb = curve(S.REC[name]); axc.plot(e, s, "-", color=col, lw=1.4)
        k = int(np.argmax(s)); axc.scatter([e[k]], [s[k]], s=18, color=col, zorder=3)
    axc.text(0.175, 12.0, "row by row,\nload retained", fontsize=6.8, color=ORANGE, ha="left", va="bottom")
    axc.text(0.147, 3.0, "one avalanche", fontsize=6.8, color=BLUE, ha="left", va="bottom")
    axc.text(0.02, 0.97, "8 armchair strips (orange) vs\n3 wide rows, 15% width dispersion (blue)", transform=axc.transAxes, fontsize=6.2, color="0.4", va="top", style="italic")
    axc.set_xlabel("engineering strain", fontsize=7.5, color=DARK); axc.set_xlim(0, 0.27); axc.set_ylim(0, 27)
    axc.yaxis.grid(True, alpha=0.25, lw=0.5); axc.set_axisbelow(True)
    save(fig, "tile4")


if __name__ == "__main__":
    tile1(); tile2(); tile3(); tile4(); print("tiles written to", OUT)
