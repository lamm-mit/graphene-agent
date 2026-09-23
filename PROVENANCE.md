# Provenance

This file records what was produced when, so that the autonomous phase can be told apart from the human-directed phase, and how to check the integrity of the record. Times are US Eastern (UTC−4), the local time of the machine and of the run ids; the prompt timestamps in `prompt/` are UTC.

## Timeline

| when (local) | event |
|---|---|
| 2026-09-04 19:51 | the prompt is sent (23:51 UTC), with five reference images |
| 2026-09-04 21:39 | first simulation of the agent (validation/convergence tests begin) |
| 2026-09-04 23:36 | first discovery simulation (stage 1 baselines) |
| 2026-09-05 21:20 | holdout predictions written and hashed (`experiments/predictions/holdout_predictions.json`, SHA-256 2eb2d80a…) |
| 2026-09-05 21:52 | first holdout simulation starts |
| 2026-09-06 01:50 | last discovery simulation ends (96 discovery runs + 17 convergence runs + 1 interactive app run) |
| 2026-09-06 02:16 | the agent packages the platform, database, figures, movies and report: end of phase 1 (about 30 h after the prompt; the human's messages in between were progress checks) |
| 2026-09-06 05:27 | first request of phase 2 (a design-space chart); Ashby chart and atlas added 05:30–05:55 |
| 2026-09-08 to 09-09 | slides; paper figures requested; sweep designs pre-registered 2026-09-09 04:54 (`experiments/predictions/paper_sweeps_predictions.json`, SHA-256 51891202…); 18 sweeps run 05:48–10:34; figures audited for text overlaps |
| 2026-09-13 | manuscript restructured, figures regenerated in Arial, TikZ workflow figure, SI material; validation and campaign figures regenerated |
| 2026-09-18 | mechanism overview figure |
| 2026-09-23 | fracture movies rendered from the stored trajectories (`paper_analysis/make_movies.py`); this release tree assembled |

## Simulations in the database

132 records in `carbon_discovery/experiments/database/runs/` (96 discovery, 18 pre-registered sweeps, 18 validation/convergence and interactive). Wall time is the sum over runs; two to four runs shared the GPU at any time.

| campaign | stage | runs | first | last | wall time (h) |
|---|---|---|---|---|---|
| validation | convergence | 17 | 2026-09-04 21:39 | 2026-09-05 05:07 | 20.3 |
| discovery | stage1_baselines | 9 | 2026-09-04 23:36 | 2026-09-05 02:24 | 4.7 |
| discovery | stage2_reconnaissance | 39 | 2026-09-05 02:04 | 2026-09-05 14:37 | 105.7 |
| discovery | stage4_discriminating | 10 | 2026-09-05 12:30 | 2026-09-05 17:25 | 19.9 |
| discovery | stage6_seeds | 13 | 2026-09-05 15:39 | 2026-09-05 19:15 | 35.2 |
| discovery | stage5_deep | 13 | 2026-09-05 15:42 | 2026-09-05 21:18 | 43.5 |
| discovery | stage7_holdouts | 12 | 2026-09-05 21:52 | 2026-09-06 01:50 | 36.7 |
| interactive | app | 1 | 2026-09-06 05:53 | 2026-09-06 05:53 | 0.1 |
| paper | paper_sweeps | 18 | 2026-09-09 05:48 | 2026-09-09 10:34 | 20.7 |

**Reconstructed records.** 13 records carry the flag `reconstructed_from_trajectory`: on 2026-09-05 the agent deleted their `record.json` files by mistake and rebuilt them from the stored trajectories and batch logs with `scripts/rebuild_records.py`; the trajectories, structures and stress–strain data were never lost. Affected: `S1_precrack_L10`, `S1_precrack_L20`, `S1_pristine_ac`, `S1_pristine_zz`, `S1_vac_clustered_2pct`, `S1_vac_random_2pct`, `mesh64_ds0.0025_ref1`, `mesh64_ds0.005_ref0`, `mesh64_ds0.005_ref1`, `mesh64_ds0.01_ref1`, `mesh64_fmax0.01`, `mesh64_fmax0.02`, `mesh64_fmax0.05`.

## What the agent produced in phase 1

Everything under `carbon_discovery/` except the items listed in the next section: the force engine and its parameter tables, neighbour list, batched FIRE minimiser, quasi-static loading driver, MD, structure generators, descriptors, exporters, the validation suite with its results and figures, campaign design and job queue, the database and its records, prediction and holdout evaluation files, analysis and figure scripts, fracture panels and movies, the browser app, the LaTeX report and its compiled PDF, `README.md`, `requirements.txt`, `environment.yml`, tests. The archive the agent produced at the end of phase 1 has SHA-256 checksums of every file in its `manifest.json`; the release manifest (`carbon_discovery/manifest.json`) was regenerated for this tree.

## Files inside `carbon_discovery/` that post-date phase 1

Added on 2026-09-06 between the first phase-2 request (05:27) and the archive of 05:55:

- `analysis/ashby_chart.py`
- `figures/campaign/ashby_overview.{png,pdf,svg,caption.txt}`
- `figures/campaign/architecture_atlas.{png,pdf,svg,caption.txt}`
- `report/text/ashby.tex, report/results_body_template.py (new section), report/report.pdf and report/generated/* (rebuilt)`
- `README.md (section "Property charts and atlas")`
- `figures/progression/*.caption.txt and figures/validation/bond_separation.caption.txt (caption files added so that every figure has one)`
- figures/_gallery_test.png and figures/_stage2_gallery.png removed (unused test renders)

New or changed after the archive of 2026-09-06 05:55 (from a comparison of the archive manifests; trajectories listed separately):

- `experiments/campaign_paper_sweeps.py` (new)
- `experiments/database/`: 108 files (records, structures and curves of the 18 sweep runs; rebuilt `index.jsonl`, `summary.csv`)
- `experiments/holdouts/paper_sweeps_evaluation.json` (new)
- `experiments/predictions/paper_sweeps_predictions.json` (new)
- `experiments/specs/paper/`: 37 files (18 sweep batch specifications, queue and run logs)
- `figures/campaign/`: 6 files regenerated (Arial font)
- `figures/validation/`: 15 files regenerated on 2026-09-13 (Arial font; content unchanged)
- `report/generated/numbers.tex` (changed)
- `report/generated/results_body.tex` (changed)
- `report/report.pdf` (changed)
- `report/text/ashby.tex` (changed)
- `scripts/package.sh` (changed)
- `trajectories/`: 18 sweep trajectories added (all 132 trajectories are on Hugging Face; 17 are tracked here)

`paper_analysis/`, `slides/` and `prompt/` are entirely phase 2 (the prompt text itself is phase 1). Everything in `paper_analysis/` is regenerated from the database by its scripts.

## Integrity

- `carbon_discovery/manifest.json`: SHA-256 of every file in `carbon_discovery/` in this release.
- The Hugging Face dataset carries `SHA256SUMS.txt` for the trajectories and large SVG figures.
- Prediction files are timestamped and carry the SHA-256 of their own content (`sha256_of_content`); the evaluation scripts (`experiments/stage_tools.evaluate_predictions`, `experiments/campaign_paper_sweeps.py`) recompute the hash and compare predictions with the records.
- Every record stores the force-field parameter checksum (`engine.parameter_checksum`), device and precision, so that a record can be rerun with `scripts/reproduce_run.py <run_id>`; a documented reproduction check is in `validation/results/reproduce_check_S1_pristine_zz.txt`.

## What is not in the repository

- 115 of the 132 raw trajectories (17 tracked: the ones read by the figure, movie and slide scripts); all are on Hugging Face.
- 47 SVG files larger than 3 MB (atom-resolved fracture panels), listed in `.gitignore`; PNG and PDF versions are tracked, the SVGs are on Hugging Face.
- The manuscript source (edited elsewhere), LaTeX build products, page renders used for visual QC, caches, `node_modules`.
- The agent's replies and the screenshots exchanged in the conversation (`prompt/README.md`).
