"""Generates report/generated/results_body.tex and discussion_body.tex: figure environments for all
campaign figures (captions from *.caption.txt) interleaved with the interpretive text blocks stored in
report/text/*.tex (written after the analysis).  Figures that do not exist are skipped."""
from __future__ import annotations
import os, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "report", "generated"); TXT = os.path.join(ROOT, "report", "text")
os.makedirs(GEN, exist_ok=True); os.makedirs(TXT, exist_ok=True)


def fig(path_rel, label, width=1.0, caption=None):
    p = os.path.join(ROOT, "figures", path_rel)
    if not os.path.exists(p + ".pdf") and not os.path.exists(p + ".png"):
        return f"% missing figure {path_rel}\n"
    ext = ".pdf" if os.path.exists(p + ".pdf") else ".png"
    cap = caption or (open(p + ".caption.txt").read().strip() if os.path.exists(p + ".caption.txt") else path_rel)
    cap = cap.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    return f"\\begin{{figure}}[H]\\centering\\includegraphics[width={width}\\textwidth]{{{path_rel}{ext}}}\\caption{{{cap}}}\\label{{{label}}}\\end{{figure}}\n"


def text(name):
    p = os.path.join(TXT, name + ".tex")
    return open(p).read() + "\n" if os.path.exists(p) else f"% text block {name} not written yet\n"


def build():
    body = []
    body.append("\\section{Force-field validation figures}\n")
    body.append(fig("validation/agreement_with_reference", "fig:agreement"))
    body.append(fig("validation/bond_separation", "fig:bondsep", caption="Bond-separation validation (TEST 19). Top: energy, bottom: force along the pulling direction (or 2D stress for the homogeneous stretch) for the C$_2$ dimer, a bond in graphene (one atom pulled along the bond), a crack-tip atom pulled open, and the affine stretch of a zigzag sheet; black: Atomistica Rebo2Scr, red dashed: Torch cpu float64, blue dotted: Torch MPS float32; thin lines (right axes): absolute differences. Sharp features are intrinsic to the published screening/cut-off functions and are reproduced identically."))
    body.append(fig("validation/precision_fast_vs_reference", "fig:precision"))
    body.append(fig("validation/material_properties", "fig:material"))
    body.append(fig("validation/convergence", "fig:convergence"))
    body.append(fig("validation/reference_backend_aqs", "fig:refaqs"))
    body.append(fig("validation/performance", "fig:performance"))
    body.append(fig("validation/md_vs_aqs", "fig:md", 0.9))
    body.append("\\section{Design space}\n")
    body.append(fig("design/translation", "fig:translation", 0.9))
    body.append(fig("design/design_space", "fig:space"))
    body.append("\\section{Results}\n")
    body.append("\\subsection{Stage 1: baselines}\n"); body.append(text("stage1")); body.append(fig("campaign/stage1_baselines", "fig:stage1"))
    body.append("\\subsection{Stage 2: reconnaissance of the design variables}\n"); body.append(text("stage2"))
    body.append(fig("campaign/stress_strain_stage2_reconnaissance", "fig:ss2", 0.85)); body.append(fig("campaign/property_maps", "fig:maps")); body.append(fig("campaign/hierarchy_ladder", "fig:ladder"))
    body.append("\\subsection{Stage 3: competing hypotheses}\n\\label{sec:hyp}\n"); body.append("\\input{generated/hypotheses}\n"); body.append(text("stage3"))
    body.append("\\subsection{Stage 4: discriminating experiments}\n"); body.append(text("stage4"))
    for f in sorted(glob.glob(os.path.join(ROOT, "figures", "campaign", "discriminating_*.pdf"))):
        body.append(fig("campaign/" + os.path.basename(f)[:-4], "fig:" + os.path.basename(f)[:-4]))
    body.append("\\subsection{Stage 5: deep mechanism studies (hierarchy, scaling, interactions)}\n"); body.append(text("stage5"))
    for f in sorted(glob.glob(os.path.join(ROOT, "figures", "campaign", "deep_*.pdf"))):
        body.append(fig("campaign/" + os.path.basename(f)[:-4], "fig:" + os.path.basename(f)[:-4]))
    body.append("\\subsection{Stage 6: seed statistics}\n"); body.append(text("stage6")); body.append(fig("campaign/seed_statistics", "fig:seeds"))
    body.append("\\subsection{Localised versus progressive fracture}\n"); body.append(text("localisation")); body.append(fig("campaign/localisation_modes", "fig:loc"))
    body.append("\\subsection{Pareto analysis}\n"); body.append(text("pareto")); body.append(fig("campaign/pareto_fronts", "fig:pareto"))
    body.append("\\subsection{Property charts and architecture atlas}\n\\label{sec:ashby}\n"); body.append(text("ashby")); body.append(fig("campaign/ashby_overview", "fig:ashby", 0.98)); body.append(fig("campaign/architecture_atlas", "fig:atlas", 0.98))
    body.append("\\subsection{Stages 7--8: holdout predictions and evaluation}\n\\label{sec:holdout}\n"); body.append(text("holdout")); body.append("\\input{generated/holdout_table}\n"); body.append(fig("campaign/holdout_prediction_vs_observation", "fig:holdout", 0.9))
    body.append("\\subsection{Top-structure fracture progression}\n\\label{sec:top}\n"); body.append(text("top")); body.append("\\input{generated/top_structures}\n")
    for f in sorted(glob.glob(os.path.join(ROOT, "figures", "progression", "comparison_*.pdf"))):
        body.append(fig("progression/" + os.path.basename(f)[:-4], "fig:" + os.path.basename(f)[:-4]))
    open(os.path.join(GEN, "results_body.tex"), "w").write("".join(body))
    disc = ["\\section{Major mechanisms}\n\\label{sec:mech}\n", text("mechanisms"), "\\section{Rejected hypotheses and counter-examples}\n", text("rejected"),
            "\\section{Limitations}\n", text("limitations"), "\\section{Threats to validity}\n", text("threats"), "\\section{Conclusions}\n", text("conclusions"),
            "\\section{Damage and connectivity analysis}\n\\label{sec:damage}\n", text("damage_def")]
    open(os.path.join(GEN, "discussion_body.tex"), "w").write("".join(disc))
    ab = os.path.join(TXT, "abstract.tex")
    open(os.path.join(GEN, "abstract.tex"), "w").write(open(ab).read() if os.path.exists(ab) else "Abstract to be written from the results.")


if __name__ == "__main__":
    build(); print("results body generated")
