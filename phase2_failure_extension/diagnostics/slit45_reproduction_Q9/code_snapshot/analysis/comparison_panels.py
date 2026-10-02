"""Comparative fracture panels for contrasting mechanisms: two runs side by side at matched event frames
(relaxed, first damage, peak, post-peak, final) with their stress-strain curves."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.fracture_viz import load_traj, select_event_frames, draw_frame, damaged_atoms_until, atom_colors
EV = 16.0217663
PF = os.path.join(ROOT, "figures", "progression")


def comparison(run_a, run_b, name, title_a, title_b, mode="energy_rel", caption=""):
    recs = {r: db.load_record(r) for r in (run_a, run_b)}
    T = {r: load_traj(os.path.join(ROOT, recs[r]["trajectory"])) for r in (run_a, run_b)}
    labels = ["relaxed", "first_damage", "peak_load", "post_peak", "final"]
    fig = plt.figure(figsize=(3.2 * len(labels) + 0.5, 8.4))
    gs = fig.add_gridspec(3, len(labels), height_ratios=[1, 1, 0.75])
    vals = [atom_colors(T[r], len(T[r]["eps_x"]) - 1, mode)[0] for r in (run_a, run_b)]
    vmin = min(np.percentile(v, 1) for v in vals); vmax = max(np.percentile(v, 99) for v in vals); vmin = min(vmin, 0)
    for row, (r, ttl) in enumerate([(run_a, title_a), (run_b, title_b)]):
        sel = dict((lab.split(" = ")[0], f) for lab, f in select_event_frames(T[r]))
        # map merged labels back
        allsel = select_event_frames(T[r])
        def frame_for(key):
            for lab, f in allsel:
                if key in lab.split(" = "): return f
            return 0 if key == "relaxed" else len(T[r]["eps_x"]) - 1
        for col, key in enumerate(labels):
            f = frame_for(key); ax = fig.add_subplot(gs[row, col])
            sc, clab = draw_frame(ax, T[r], f, mode, vmin, vmax, damage_atoms=damaged_atoms_until(recs[r], T[r]["eps_x"][f]), s=4, lw=0.4)
            ax.set_title((ttl + "\n" if col == 0 else "") + f"{key.replace('_',' ')}: ε={T[r]['eps_x'][f]:.3f}, σ={T[r]['sigma_xx'][f]*EV:.1f} N/m", fontsize=7.5)
    axs = fig.add_subplot(gs[2, :])
    for r, ttl, col in [(run_a, title_a, "#1f77b4"), (run_b, title_b, "#d62728")]:
        axs.plot(T[r]["eps_x"], T[r]["sigma_xx"] * EV, "-", color=col, lw=1.5, label=f"{ttl}: σmax={recs[r]['metrics']['strength_Nm']:.1f} N/m, εf={recs[r]['metrics']['failure_strain']:.3f}, W={recs[r]['metrics']['work_to_failure_J_m2']:.2f} J/m², L={recs[r]['metrics']['damage_localization']:.2f}")
    axs.set_xlabel("engineering strain"); axs.set_ylabel("2D stress (N/m)"); axs.grid(alpha=0.3); axs.legend(fontsize=7)
    cb = fig.colorbar(sc, ax=axs, fraction=0.03, pad=0.01); cb.set_label(clab, fontsize=8)
    plt.tight_layout()
    base = os.path.join(PF, f"comparison_{name}")
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(base + "." + ext, dpi=170 if ext == "png" else None)
    plt.close(fig)
    open(base + ".caption.txt", "w").write(caption or f"Comparative fracture panels: {title_a} (top) versus {title_b} (bottom) at matched event frames (relaxed, first bond loss, peak load, post-peak, final); colour = energy change per atom, red circles = atoms that lost a bond; bottom: stress-strain curves.")
    return base + ".png"


if __name__ == "__main__":
    print(comparison(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
