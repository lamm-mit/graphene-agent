# Figure-source map

Numbers can change between Overleaf drafts; the scientific subject is the
stable identifier. Commands and data-restoration instructions are in
[REPRODUCING.md](REPRODUCING.md).

| Figure subject | Current source in this release | Inputs |
|---|---|---|
| Original design-space charts and discovery analyses | `paper_analysis/current/make_figures.py`, `extra_figures.py`, `mechanism_figure.py` | Original `carbon_discovery` records; frozen public version remains in `paper_analysis/` |
| Old/new versions of the same panels, including Ashby hulls, slit angles, alignment, disorder and progression | `phase2_failure_extension/candidate_figures/one_to_one/scripts/build_matched.py` | Copied baseline + 45 loading/check trajectories; unchanged original figure functions |
| Comparison SI layout | `phase2_failure_extension/candidate_figures/paper_si_20260926/scripts/build_si.py` | One-to-one outputs and paired numerical tables |
| Standalone loading overview, endpoint bounds and diagnostics | `phase2_failure_extension/scripts/analyze.py` | Study records and curves; four summary figures |
| Reconnection and angle dependence | `phase2_failure_extension/candidate_figures/bond_reconnection_20260926/scripts/` | Primary saved states and strict/numerical checks, explicitly distinguished |
| 45-degree trajectory and connected lattice blowups | `phase2_failure_extension/candidate_figures/slit45_mechanism_20260926/scripts/plot.py`, `slit45_lattice_detail_20260926/scripts/build.py` | Phase-II paths; Q9 and Phase-I path remain separate |
| Hierarchy scale, flaws, seams and building-block controls | `hierarchy_followup/analysis/manuscript_figures_phase3.py` | 78 follow-up records, reference copies and predictions |
| Design-principle schematic | `phase2_failure_extension/candidate_figures/design_principles_schematic/make_figure.py` | Schematic with explicit model scope |
| Atomistic-engine schematic | `phase2_failure_extension/candidate_figures/phase1_engine_schematic/make_figure.py` | Architecture of the original engine |
| Atlas embedding, 24-group library and local neighbors | `phase2_failure_extension/candidate_figures/atlas_main_paper_20261002/code/render.py` | Bundled unchanged 256K map and 27 exact atomic examples |
| Experiment/workflow diagrams | `paper_analysis/schematics/*.tikz` | Diagram source; existing original figure source retained |

One-time manuscript mutation scripts, Overleaf snapshots, private editing
backups, working logs and rendered review duplicates are intentionally omitted
from the canonical figure workflow. The code release does not contain or edit
the current Overleaf manuscript. Historical snapshots under study inputs are
retained only where needed to regenerate the comparisons.
