# Reproducing results and starting new work

Run commands from the repository root unless indicated. Python 3.12 is the
release-validation interpreter. Use separate Python processes for different
study trees: historical modules deliberately retain names such as `simulation`
and `experiments`, and importing several study snapshots in one interpreter can
mix implementations.

## Install and test the instrument

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest carbon_discovery/tests -q
python -m pytest hierarchy_followup/tests/test_followup.py tests_release -q
```

For independent reference comparisons, install Atomistica separately in the
environment. Three reference comparisons are skipped if it is unavailable;
those skips are not evidence that reference validation passed. The release was
tested with Atomistica present. It is not redistributed in this repository.
`requirements.txt` retains the original full-platform dependency list, including
the web application. `requirements-analysis.txt` adds figure/report dependencies.
`requirements-atlas.txt` covers design generation and embedding; historical atlas
pins are retained next to the corresponding source snapshots.

## Restore saved simulation inputs

The release handoff includes four archives in a sibling `data/` folder:
`hierarchy-data.tar.gz`, `loading-data.tar.gz`, `diagnostic-data.tar.gz`, and
`explorer-data.tar.gz`. Their complete member inventories and checksums are in
`release/data_bundles.json`.

```sh
python scripts/restore_release_data.py --bundle-dir ../data --all
```

Restoration verifies the archive and each file, rejects unsafe paths, skips
identical existing files and refuses to overwrite different files. These new
study bundles have been prepared locally; no public download URL is asserted
for them. Small records, curves and parameters are in the code release. The
original public trajectory collection remains available through
`scripts/fetch_data_from_huggingface.py --what trajectories`.

## Loading-limit comparisons

After restoring `loading-data`:

```sh
python phase2_failure_extension/candidate_figures/scripts/build_data.py
python scripts/validate_saved_analysis.py
python phase2_failure_extension/scripts/analyze.py
python phase2_failure_extension/candidate_figures/one_to_one/scripts/build_matched.py
python phase2_failure_extension/candidate_figures/one_to_one/scripts/export_panel_changes.py
python phase2_failure_extension/candidate_figures/paper_si_20260926/scripts/build_si.py
```

The data builder checks the frozen input hashes, recalculates the A/B/C endpoint
metrics, preserves 96 discovery designs and the original 40-mesh alignment
population, and reconstructs the log-space family hulls. `build_matched.py`
calls the same original figure functions with the original and rerun data;
it creates its own runtime under `one_to_one/`. Figure and table exports go into
this release, including `generated/loading_si_exports/`.

Here A means the original recorded trajectory, B the rerun truncated by the
original stop rules, and C the same rerun with the longer loading limits. A–B
differences are reproduction differences; B–C measures added loading on the
same path. Strict numerical checks are separate branches, not replacements for
the primary data. Censoring and force/transverse-stress residual flags remain
part of the reported results. Work is an integrated stress–strain response,
not fracture toughness. The follow-up retains the operational failure criterion
while changing the strain limits.

`source_original/`, `code/` and `code_verified/` preserve distinct scientific
snapshots. Historical source checksums document the executed campaigns;
release wrapper changes are separately documented in
`release/portability_changes.json`. Do not edit an old record to relabel a new
run. Historical schedulers and job manifests are provided for provenance;
they are not a command to repeat the entire campaign automatically.

The release keeps frozen derived inputs in `inputs/frozen_analysis/` and uses
`candidate_figures/data/release_inventory.json` to verify them. The original
inventory is retained. This allows the standalone analysis to regenerate its
outputs without invalidating the frozen comparison inputs.

## Mechanisms and reproduction diagnostics

```sh
python phase2_failure_extension/candidate_figures/bond_reconnection_20260926/scripts/build_figures.py
python phase2_failure_extension/candidate_figures/slit45_mechanism_20260926/scripts/plot.py
python phase2_failure_extension/candidate_figures/slit45_lattice_detail_20260926/scripts/build.py
```

The 45-degree default plot uses the two Phase-II paths. Its optional Phase-I
comparison and Q9 diagnostics remain separate. Atomic connections in these
visualizations are geometric neighbor criteria, not a quantum bond-order
measurement. See `diagnostics/slit45_reproduction_Q9/README.md` and its final
review for the batch/single reproduction evidence and limitations. Q9 requires
the `diagnostic-data` bundle for full saved runtime checkpoints and trajectories.
Historical MPS/float32 settings are not silently changed to CPU/float64.

## Hierarchy follow-up

After restoring `hierarchy-data`:

```sh
python hierarchy_followup/analysis/hierarchy_followup_analysis.py --no-panels
python hierarchy_followup/analysis/manuscript_figures_phase3.py scale flaw
```

The historical internal name “phase3” refers to later collaborative Phase-II
work in the paper. This local campaign contains 78 records, with tested 120 Å
and 300 Å cells. Do not reinterpret these as 120 nm and 300 nm simulations.
Predictions, registry variants, negative results and endpoint qualifications are
retained. Exported paper assets go to `generated/hierarchy_exports/`, never to
the author's manuscript. The original 132-record platform collection and its
earlier Phase-II subset are not merged into the follow-up population.

## The 256K atlas

```sh
python -m pip install -r requirements-atlas.txt
python scripts/validate_atlas_release.py
python phase2_failure_extension/candidate_figures/atlas_main_paper_20261002/code/validate.py
python phase2_failure_extension/candidate_figures/atlas_main_paper_20261002/code/render.py
python atlas/graphene_design_dataset_256k/tools/load_structure.py --help
python atlas/graphene_design_dataset_256k/tools/regenerate_design.py 65248 design.extxyz
```

The last command retrieves the pinned public source/recipe and refuses an
existing output. Standard extended XYZ includes carbon coordinates, cell and
periodic boundaries. The atlas contains **unrelaxed geometries without computed
mechanical properties**. Its descriptor/PCA/UMAP space describes geometry
similarity, not energy or a learned force field. The release validation
regenerates 27 saved examples covering all 24 design groups and compares their
full geometry digests, coordinates, cells and periodicity.

Full historical generation uses sibling `graphene_design_dataset_64k` and
`graphene_design_dataset_256k` workdirs. Source alone is insufficient: use each
dataset's `tools/restore_workdir.py` to restore the published masks, recipes,
sprites and embeddings. This is a substantial download/reconstruction. Exact
mask retrieval, recipe regeneration and a new stochastic generation campaign
are different operations; retain the distinction. FFmpeg/ffprobe are optional
external dependencies for movie encoding.

Restore a full atlas workdir into a **new or empty directory**, separate from
the curated source. The release tool refuses to overwrite existing work.

For the explorer, restore `explorer-data`, then:

```sh
python -m http.server 8000 --directory atlas/explorer
```

Open `http://localhost:8000`. Structure/image retrieval uses the public dataset.
The static explorer has no integrated mechanical simulation engine.

## New simulations

Use the relevant study's `scripts/run_batch.py` with a new JSON spec and an
explicit device/dtype. For example, inspect
`python carbon_discovery/scripts/run_batch.py --help` and the existing spec
format before defining a new case. Test a short CPU/float64 run first. Full
campaigns can take substantial compute time. Changing device, precision,
batching, minimization tolerances or strain increments creates a new numerical
experiment; preserve its settings and outputs under a new run identity.

The release checks do not rerun all discovery simulations or all 256,000
designs and do not establish convergence of every historical trajectory.
