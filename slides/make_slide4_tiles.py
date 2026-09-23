"""Tiles for talk slide 4 (how the force engine was written and validated).  Three tiles, 4.0 x 3.6 in:
A) pipeline schematic + lines of new code per component, B) GPU batching schematic + measured speed,
C) agreement with the Atomistica reference per configuration + full stress-strain overlay (TEST 18).
All numbers are read from validation/results and the source tree."""
from __future__ import annotations
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(p for p in (os.path.join(os.path.dirname(HERE), "carbon_discovery"), os.path.dirname(HERE)) if os.path.isdir(os.path.join(p, "experiments")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(HERE, "tiles4"); os.makedirs(OUT, exist_ok=True)
DARK, RED, BLUE, ORANGE, GREEN, GREY = "#222222", "#c0392b", "#1f77b4", "#ff7f0e", "#2ca02c", "#d8d8d8"
EV = 16.0217663


def new_tile():
    return plt.figure(figsize=(4.0, 3.6), dpi=300)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), facecolor="white")
    plt.close(fig)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color("0.3")
    ax.tick_params(labelsize=6.5, length=2, width=0.5, colors="0.25")


def box(ax, x, y, w, h, text, fc="#f0f0f0", ec="0.5", size=6.3, color=DARK, bold=False, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="0.35", lw=1.0, ms=8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms), zorder=4)


def count_lines(paths):
    n = 0
    for p in paths:
        for f in glob.glob(os.path.join(ROOT, p), recursive=True):
            n += sum(1 for _ in open(f, errors="ignore"))
    return n


# --------------------------------------------------------------------------- tile A: written from scratch
def tileA():
    fig = new_tile()
    ax = fig.add_axes([0.02, 0.50, 0.96, 0.49]); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.2, 3.3, 4.3, 1.45, "published functional form\nBrenner 2002 (REBO2)\n+ Pastewka 2008/2013 (screening)", fc="#ffffff", ec="0.45")
    box(ax, 5.5, 3.3, 4.3, 1.45, "Atomistica Fortran source\n(read for conventions and tables;\nnever run in production)", fc="#ffffff", ec="0.45")
    box(ax, 1.2, 1.15, 7.6, 1.55, "PyTorch port, new code\ndense pair/triplet tensors · screening · bond-order splines\nautograd forces and per-atom virials · float64 reference / float32 fast\nMPS · CUDA · CPU", fc="#e8eef7", ec=BLUE, size=6.0, bold=False, lw=1.0)
    arrow(ax, 2.35, 3.3, 3.6, 2.72); arrow(ax, 7.65, 3.3, 6.4, 2.72)
    box(ax, 1.2, 0.05, 3.5, 0.75, "validation gate: 20 tests", fc="#fdecea", ec=RED, size=6.3, bold=True)
    box(ax, 5.3, 0.05, 3.5, 0.75, "campaign: 113 simulations", fc="#eef7ee", ec=GREEN, size=6.3, bold=True)
    arrow(ax, 2.95, 1.15, 2.95, 0.82); arrow(ax, 4.7, 0.42, 5.3, 0.42)
    # lines of new code per component
    comps = [("force-field kernel (PyTorch)", ["potentials/rebo2scr/pytorch/*.py"]),
             ("parameters and spline tables", ["potentials/rebo2scr/reference/*.py"]),
             ("neighbour list, FIRE, AQS, MD", ["simulation/**/*.py"]),
             ("structure generators, descriptors, I/O", ["atomistics/**/*.py"]),
             ("analysis and figures", ["analysis/*.py"]),
             ("campaign design, database, queue", ["experiments/*.py", "scripts/*.py"]),
             ("validation suite", ["validation/*.py"]),
             ("browser app", ["app/backend/*.py", "app/frontend/*.html", "app/frontend/static/*.js", "app/frontend/static/*.css"])]
    vals = [count_lines(p) for _, p in comps]
    axb = fig.add_axes([0.42, 0.10, 0.53, 0.33]); style(axb)
    y = np.arange(len(comps))[::-1]
    axb.barh(y, vals, color=BLUE, height=0.65)
    for yy, v, (n, _) in zip(y, vals, comps):
        axb.text(v + 40, yy, f"{v:,}", va="center", fontsize=5.8, color=DARK)
        axb.text(-60, yy, n, va="center", ha="right", fontsize=5.8, color=DARK)
    axb.set_yticks([]); axb.spines["left"].set_visible(False)
    axb.set_xlim(0, max(vals) * 1.25); axb.set_xlabel("lines of new code", fontsize=6.3, color=DARK, labelpad=1)
    axb.set_title(f"about {sum(vals):,} lines written in this project; Atomistica and ASE used only as reference and I/O", fontsize=5.6, color="0.35", pad=2, loc="right")
    save(fig, "tileA")
    return sum(vals)


# --------------------------------------------------------------------------- tile B: batched for the GPU + honest speed
def tileB():
    P = json.load(open(os.path.join(ROOT, "validation", "results", "performance.json")))
    fig = new_tile()
    ax = fig.add_axes([0.02, 0.52, 0.96, 0.47]); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    # three structures -> one super-system tensor -> GPU -> autograd
    for i, (xx, col) in enumerate([(0.3, BLUE), (1.5, ORANGE), (2.7, GREEN)]):
        ax.add_patch(Rectangle((xx, 3.45), 1.0, 1.0, fc=GREY, ec=col, lw=1.0, zorder=2))
        for (px, py) in [(0.28, 0.3), (0.7, 0.3), (0.28, 0.72), (0.7, 0.72)]:
            ax.add_patch(plt.Circle((xx + px, 3.45 + py), 0.1, fc="white", ec="none", zorder=3))
        ax.text(xx + 0.5, 4.6, "ABC"[i], ha="center", va="bottom", fontsize=6.5, color=col, fontweight="bold")
    ax.text(2.0, 3.2, "structures in one batch", ha="center", va="top", fontsize=6.0, color=DARK)
    arrow(ax, 3.9, 3.95, 4.6, 3.95)
    for i, (x0, w, col) in enumerate([(4.7, 1.2, BLUE), (5.9, 1.2, ORANGE), (7.1, 1.2, GREEN)]):
        ax.add_patch(Rectangle((x0, 3.7), w, 0.5, fc=col, ec="white", lw=0.5, alpha=0.8, zorder=2))
    ax.text(6.5, 4.35, "one 'super-system' tensor: all atoms, all pairs, all triplets\n(dense, padded, masked)", ha="center", va="bottom", fontsize=5.8, color=DARK, linespacing=1.2)
    ax.text(6.5, 3.55, "structure ids carry per-structure energies and stresses", ha="center", va="top", fontsize=5.6, color="0.35")
    box(ax, 4.9, 1.35, 3.4, 1.15, "GPU (MPS) or CPU\nenergy  →  autograd  →  forces,\nper-atom energies and virials", fc="#e8eef7", ec=BLUE, size=6.0)
    arrow(ax, 6.6, 3.7, 6.6, 2.5)
    for i, yy in enumerate([2.45, 1.9, 1.35]):
        box(ax, 0.3, yy - 0.02, 2.9, 0.45, f"simulation process {i + 1}", fc="#ffffff", ec="0.5", size=5.8)
        arrow(ax, 3.2, yy + 0.2, 4.9, 1.93, color="0.5", lw=0.7, ms=6)
    ax.text(2.0, 1.15, "several processes share the GPU\n(the campaign queue)", ha="center", va="top", fontsize=5.8, color="0.35", linespacing=1.2)
    # speed: microseconds per atom per evaluation at ~5700 atoms
    def at(series, n=5748):
        e = [r for r in series if r["n_atoms"] == n]; return e[0]["us_per_atom"] if e else float("nan")
    items = [("Fortran\nCPU f64", at(P["atomistica"]), "0.35"), ("Torch\nGPU f32", at(P["torch_fast"]), BLUE),
             ("Torch\nCPU f32", at(P["torch_cpu32"]), "#7fa8d0"), ("Torch\nCPU f64", at(P["torch_cpu64"]), "#b3c9e3")]
    axs = fig.add_axes([0.12, 0.09, 0.42, 0.33]); style(axs)
    for i, (n, v, c) in enumerate(items):
        axs.bar(i, v, color=c, width=0.7); axs.text(i, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.3, color=DARK)
    axs.set_xticks(range(4)); axs.set_xticklabels([it[0] for it in items], fontsize=5.6, color=DARK); axs.tick_params(axis="x", length=0, pad=2)
    axs.set_ylabel("µs per atom per evaluation", fontsize=6.0, color=DARK); axs.set_ylim(0, max(v for _, v, _ in items) * 1.25)
    axs.set_title("speed (5,748 atoms): not faster per call\nthan the Fortran reference", fontsize=6.0, color=DARK, pad=2)
    axc = fig.add_axes([0.66, 0.09, 0.31, 0.33]); style(axc)
    conc = P["concurrency"]
    for i, r in enumerate(conc):
        axc.bar(i, r["aggregate_evals_per_s"], color=BLUE, width=0.6); axc.text(i, r["aggregate_evals_per_s"] + 1.5, f"{r['aggregate_evals_per_s']:.0f}", ha="center", va="bottom", fontsize=6.3, color=DARK)
    axc.set_xticks(range(len(conc))); axc.set_xticklabels([str(r["n_processes"]) for r in conc], fontsize=6.3); axc.tick_params(axis="x", length=0, pad=2)
    axc.set_xlabel("processes on one GPU", fontsize=6.0, color=DARK, labelpad=1); axc.set_ylabel("evaluations / s", fontsize=6.0, color=DARK)
    axc.set_ylim(0, max(r["aggregate_evals_per_s"] for r in conc) * 1.25); axc.set_title("throughput (≈3,000 atoms)", fontsize=6.0, color=DARK, pad=2)
    save(fig, "tileB")


# --------------------------------------------------------------------------- tile C: validated against the reference
def tileC():
    E = json.load(open(os.path.join(ROOT, "validation", "results", "energy_force_vs_reference.json")))
    R = json.load(open(os.path.join(ROOT, "validation", "results", "reference_aqs.json")))["precrack"]
    fig = new_tile()
    axa = fig.add_axes([0.42, 0.61, 0.55, 0.33]); style(axa)
    y = np.arange(len(E))[::-1]
    dE = np.array([max(e["max_abs_dE_atom"], 1e-17) for e in E]); dF = np.array([max(e["max_abs_dF"], 1e-17) for e in E])
    axa.barh(y + 0.18, dE, height=0.34, color=BLUE, label="energy per atom (eV)")
    axa.barh(y - 0.18, dF, height=0.34, color=ORANGE, label="force component (eV/Å)")
    axa.set_xscale("log"); axa.set_xlim(1e-16, 1e-7)
    axa.axvline(1e-9, color=RED, ls="--", lw=0.8); axa.text(1e-9, len(E) - 0.3, " tolerance", color=RED, fontsize=5.6, va="top", ha="left")
    axa.set_yticks(y); axa.set_yticklabels([e["config"].replace("near rupture: ", "near rupture, ").replace("post-rupture: ", "after rupture, ") for e in E], fontsize=4.9, color=DARK)
    axa.tick_params(axis="y", length=0)
    axa.set_xlabel("max |Torch − Atomistica|, float64", fontsize=6.0, color=DARK, labelpad=1)
    axa.legend(fontsize=5.2, loc="lower right", frameon=True, framealpha=0.95, edgecolor="none", handlelength=1.0, borderaxespad=0.1)
    axa.set_title("12 configurations incl. pore edges, crack tips, rupture", fontsize=6.0, color=DARK, pad=2)
    axb = fig.add_axes([0.14, 0.09, 0.83, 0.36]); style(axb)
    a = R["atomistica_ase_fire"]; t64 = R["torch_cpu64_ase_fire"]; t32 = R["torch_mps32_batched_fire"]
    sc = EV if max(abs(v) for v in a["sxx"]) < 5 else 1.0
    axb.plot(a["strain"], np.array(a["sxx"]) * sc, "-", color="k", lw=2.4, label="Atomistica + ASE FIRE (reference)")
    axb.plot(t64["strain"], np.array(t64["sxx"]) * sc, "--", color=RED, lw=1.1, label="PyTorch float64 + ASE FIRE")
    axb.plot(t32["strain"], np.array(t32["sxx"]) * sc, "-.", color=GREEN, lw=1.1, label="PyTorch GPU float32 + batched FIRE")
    axb.set_xlabel("engineering strain", fontsize=6.3, color=DARK, labelpad=1); axb.set_ylabel("2D stress (N/m)", fontsize=6.3, color=DARK)
    axb.legend(fontsize=5.4, loc="upper left", frameon=False, handlelength=1.6)
    d64 = R["torch_cpu64_ase_fire_max_abs_dsigma_Nm"]; d32 = R["torch_mps32_batched_fire_max_abs_dsigma_Nm"]
    axb.set_title(f"precracked sheet ({R['n_atoms']} atoms) loaded to fracture", fontsize=6.0, color=DARK, pad=2)
    axb.text(0.98, 0.55, f"max |Δσ| vs reference:\n{d64:.0e} N/m (same minimiser)\n{d32:.2f} N/m (GPU float32, batched minimiser)", transform=axb.transAxes, fontsize=5.4, color=DARK, ha="right", va="center", linespacing=1.3)
    save(fig, "tileC")


if __name__ == "__main__":
    total = tileA(); tileB(); tileC(); print("tiles written to", OUT, "| lines of code:", total)
