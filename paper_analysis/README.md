# Phase-2 analysis: the figures and tests behind the paper

Everything in this folder was produced after the autonomous phase, in a conversation between the human author and the
same agent (see [`../prompt/phase2_requests.md`](../prompt/phase2_requests.md)). All scripts read the experiment
database in [`../carbon_discovery/`](../carbon_discovery/) and write into `figures/`; no number in the paper is typed by
hand, they enter the manuscript as LaTeX macros (`figures/numbers.tex`, `figures/facts.tex`, `figures/mech_numbers.tex`).

## Scripts

| script | output |
|---|---|
| `make_figures.py` | the six data figures of the paper (rule of mixtures, load paths, slit-angle sweep, hierarchy vs alignment index, costs of disorder, pre-registered predictions) as PDF/PNG/SVG, plus `numbers.tex`, `numbers.json`, `sweeps_table.tex`. Contains the collision resolver `resolve()` that moves labels, legends and annotations until no text overlaps data, lines or other text. |
| `check_overlaps.py` | the audit: renders each figure, tests every text element against every marker, line segment, bar and image and against all other text, and prints the conflicts. The release figures report 0 conflicts. Running it also regenerates the six figures. |
| `extra_figures.py` | the experiment-workflow figure (TikZ source `figures/fig_experiment.tikz`, standalone wrapper `figures/fig1_experiment.tex`), the design-space chart in the paper's font, the fracture-progression comparison, and the SI material (`figures/si/`: validation figures, atlas, ten progression panels, tables) together with `facts.tex` (timeline, compute and code-size facts computed from the database and the source tree). |
| `mechanism_figure.py` | the mechanism overview figure (`figures/fig_mechanisms.*`) with schematics drawn to scale from the generator parameters, and its macros and LaTeX snippet. |
| `make_movies.py` | ten synchronised multi-panel fracture movies (2560×1440 masters, 1080p versions; `movies_hq/`, with `captions.json`, `README.md`, `si_movie_captions.tex` and posters) and, with `--social`, fourteen clean single-structure square movies in two fields (`movies_social/`). Interpolated frames, slow motion around events. The MP4 files are on Hugging Face. |
| `build_figures.sh` | runs the figure scripts in order (about 10 minutes); movies are rendered separately (`python make_movies.py --all --jobs 4 --compact`, about an hour). |

The 18 pre-registered sweep simulations that these figures use (slit-angle sweep, tip-overlap controls, alignment sweep)
were designed and their predictions hashed by [`../carbon_discovery/experiments/campaign_paper_sweeps.py`](../carbon_discovery/experiments/campaign_paper_sweeps.py);
their records are in the database with `campaign = "paper"`, and the scored predictions in
[`../carbon_discovery/experiments/holdouts/paper_sweeps_evaluation.json`](../carbon_discovery/experiments/holdouts/paper_sweeps_evaluation.json).

## Definitions introduced in phase 2

- **Load-path alignment index** `A = a cos 2φ`, with `a` the anisotropy of the structure tensor of the solid phase (distance-to-pore field) and `φ` the ligament direction relative to the load; +1 for material entirely in straight paths along the load, −1 across, 0 isotropic. Implemented in `campaign_paper_sweeps.alignment_index` from the stored descriptors.
- **Slit tip overlap** `O = L cos θ − p_x/2` and **row clearance** `t = p_y − L sin θ` for slit arrays of length `L` at angle `θ` with row period `p_y` and half-period stagger; `O > 0` means the tips of adjacent rows overlap along the load.
- **Hierarchy premium**: strength of a nested design minus the single-level trend in `A` at the same `A`.

## Figures

`figures/` holds every figure as PDF, PNG and SVG, except that SVG files larger than 3 MB (atom-resolved panels) are on
Hugging Face; `../scripts/fetch_data_from_huggingface.py --what svg` puts them back in place. The manuscript itself is
not in this repository.

## Fonts

The paper figures use Arial. On systems without Arial, Matplotlib substitutes DejaVu Sans and prints a warning; the
layout and the audit are unaffected.
