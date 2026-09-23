"""Stage 4 discriminating experiments: predicted vs observed strength per pair, with the alternative-hypothesis prediction."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from analysis.campaign_analysis import M, save, curve
plt.rcParams.update({"font.size": 8.5})

PAIRS = [("H-A vs H-B: pore shape at equal net section", ["S2_H1_square_D40", "S4_circle_D40_sameMinSF", "S4_circle_D40_samePhi"], {"S4_circle_D40_sameMinSF": (11.3, "H-A"), "S4_circle_D40_samePhi": (9.8, "H-A")}),
         ("H-C: ligament dispersion (equivalence of load paths)", ["S2_H1_square_D40", "S4_square_D40_dispersed", "S2_H1_p32", "S4_p32_dispersed"], {"S4_square_D40_dispersed": (14.7, "H-C: progressive"), "S4_p32_dispersed": (14.4, "H-C: progressive")}),
         ("H-E: flaw size plateau", ["S1_precrack_L10", "S1_precrack_L20", "S1_precrack_L30", "S4_precrack_L30_cell150", "S4_precrack_L40_cell150", "S1_hole_d20", "S4_hole_d30"], {"S4_precrack_L30_cell150": (18.0, "H-E (Griffith 13.9)"), "S4_precrack_L40_cell150": (18.0, "H-E (Griffith 12.0)"), "S4_hole_d30": (17.5, "H-E")}),
         ("H-F: vein direction at matched porosity", ["S2_H1_p12", "S4_H2_D40_W8_veins_y", "S2_H2_D40_W8", "S4_H2_D40_W8_veins_x", "S2_H1_square_D40"], {"S4_H2_D40_W8_veins_x": (14.5, "H-F"), "S4_H2_D40_W8_veins_y": (11.5, "H-F")})]


def main():
    recs = {r["name"]: r for r in db.all_records() if r.get("status") == "completed"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for k, (title, names, preds) in enumerate(PAIRS):
        ax = axes[0, k]; names = [n for n in names if n in recs]
        x = np.arange(len(names)); obs = [M(recs[n], "strength_Nm") for n in names]
        cols = ["#d62728" if M(recs[n], "fracture_mode").startswith("abrupt") else ("#ff7f0e" if M(recs[n], "fracture_mode").startswith("stepwise") else "#1f77b4") for n in names]
        ax.bar(x, obs, color=cols, alpha=0.85)
        for j, (i, n) in enumerate([(i, n) for i, n in enumerate(names) if n in preds]):
            ax.plot([i - 0.3, i + 0.3], [preds[n][0]] * 2, "k--", lw=1.5); ax.text(i, preds[n][0] + (0.5 if j % 2 == 0 else -1.6), preds[n][1], ha="center", fontsize=5.5)
        ax.set_xticks(x); ax.set_xticklabels([n.replace("S2_", "").replace("S4_", "").replace("S1_", "") for n in names], rotation=60, fontsize=6.5); ax.set_ylabel("strength (N/m)"); ax.set_title(title, fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax2 = axes[1, k]
        for n in names:
            e, s, nb = curve(recs[n]); ax2.plot(e, s, lw=1.1, label=n.replace("S2_", "").replace("S4_", "").replace("S1_", ""))
        ax2.set_xlabel("engineering strain"); ax2.set_ylabel("2D stress (N/m)"); ax2.legend(fontsize=5.5); ax2.grid(alpha=0.3)
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color="#d62728", label="abrupt"), Patch(color="#ff7f0e", label="stepwise"), Patch(color="#1f77b4", label="progressive")], fontsize=6.5, loc="upper right")
    plt.tight_layout()
    save(fig, "discriminating_experiments", "Stage 4 discriminating experiments. Top: observed strengths (bar colour = fracture mode) with the predictions recorded before the runs (dashed lines, labelled by the hypothesis that generated them); bottom: the corresponding stress-strain curves. Left to right: pore shape at equal net section (H-A vs H-B), ligament-width dispersion (H-C), flaw-size plateau in 100 and 150 A cells (H-E), vein direction in the nested mesh (H-F).")
    print("stage4 figure written")


if __name__ == "__main__":
    main()
