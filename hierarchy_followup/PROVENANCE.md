# Provenance of the hierarchy follow-up tree (phase 3)

Written 2026-09-26 09:01:20 by `scripts/write_provenance.py`. The tree was created on 2026-09-23 from `RESULTS_nano_v2/carbon_discovery/` (the phase-1/2 tree, archived as `carbon_discovery_20260923.zip`; code identical to the public release `lamm-mit/graphene-agent`). Only platform code was copied; no phase-1 run, trajectory, figure or report is duplicated except the 132 `record.json` + `stress_strain.csv` pairs under `reference/phase1_runs/` (read-only inputs of the assessment and the rules).

Environment of the follow-up: python 3.12.14, torch 2.14.0, numpy 2.5.3, scipy 1.18.1, ase 3.29.0, atomistica 1.2.7 (conda env `graphene-agent`, `environment.yml`).

## Byte-identical to the phase-1 manifest (64 files)

| file | SHA-256 |
|---|---|
| `PLATFORM_README.md` | `d25a424697b21e55…` |
| `requirements.txt` | `6892bd06c2cb7042…` |
| `potentials/__init__.py` | `e3b0c44298fc1c14…` |
| `potentials/rebo2scr/__init__.py` | `e3b0c44298fc1c14…` |
| `potentials/rebo2scr/tests/__init__.py` | `e3b0c44298fc1c14…` |
| `potentials/rebo2scr/tests/quick_compare.py` | `bac0acde6bb2ac64…` |
| `potentials/rebo2scr/pytorch/__init__.py` | `e3b0c44298fc1c14…` |
| `potentials/rebo2scr/pytorch/ase_calculator.py` | `70de7c3ecc501fb7…` |
| `potentials/rebo2scr/pytorch/neighborlist.py` | `80bce1bb17e200d0…` |
| `potentials/rebo2scr/pytorch/torch_rebo2scr.py` | `9b5f5be13bf4c2ba…` |
| `analysis/__init__.py` | `e3b0c44298fc1c14…` |
| `analysis/ashby_chart.py` | `e489d6e06c65dd5e…` |
| `analysis/comparison_panels.py` | `7c9411aaab2acdcf…` |
| `analysis/design_space_figure.py` | `858921f2820ade0f…` |
| `analysis/holdout_figure.py` | `dcab59a07be2255a…` |
| `analysis/image_interpretation.json` | `2586ccf6029eba34…` |
| `analysis/metrics.py` | `91ee799b0c160758…` |
| `analysis/physics_description.json` | `0d4222d8b3e66f5b…` |
| `analysis/seed_statistics.py` | `c3450519e408443b…` |
| `analysis/stage1_figures.py` | `68021a4c10074bd5…` |
| `analysis/stage2_figures.py` | `71bfd8b94d3039e9…` |
| `analysis/stage4_figure.py` | `f0b292536e50c9c4…` |
| `analysis/stage5_figures.py` | `ff133adff4bb6fb1…` |
| `analysis/stage7_panels.py` | `633aff817a846c51…` |
| `experiments/__init__.py` | `e3b0c44298fc1c14…` |
| `experiments/campaign_paper_sweeps.py` | `8ab3d6e4132d70ea…` |
| `experiments/campaign_stage3_hypotheses.py` | `57e05dd5eeb156e5…` |
| `experiments/campaign_stage456.py` | `0f37c3a3da41369f…` |
| `experiments/campaign_stage7_holdouts.py` | `98d237e49dc21964…` |
| `tests/__init__.py` | `e3b0c44298fc1c14…` |
| `tests/test_all.py` | `8d0595d98ff2d8b8…` |
| `atomistics/__init__.py` | `e3b0c44298fc1c14…` |
| `atomistics/io.py` | `17769ff545dd5e39…` |
| `atomistics/structures/__init__.py` | `e3b0c44298fc1c14…` |
| `atomistics/structures/graphene.py` | `150bd538fe18c2cb…` |
| `atomistics/descriptors/__init__.py` | `e3b0c44298fc1c14…` |
| `atomistics/descriptors/descriptors.py` | `7e5d480540dd5e2d…` |
| `atomistics/generators/__init__.py` | `e3b0c44298fc1c14…` |
| `simulation/__init__.py` | `e3b0c44298fc1c14…` |
| `simulation/system.py` | `e66c56b4b8208940…` |
| `simulation/topology.py` | `7d8a13e1a19a34d0…` |
| `simulation/aqs/__init__.py` | `e3b0c44298fc1c14…` |
| `simulation/aqs/aqs.py` | `a23c6b4381b66d9d…` |
| `simulation/md/__init__.py` | `e3b0c44298fc1c14…` |
| `simulation/md/md.py` | `f3d1d2539bcb2a32…` |
| `simulation/minimization/__init__.py` | `e3b0c44298fc1c14…` |
| `simulation/minimization/fire.py` | `5fff9194ebaf4344…` |
| `simulation/neighborlist/__init__.py` | `e3b0c44298fc1c14…` |
| `scripts/__init__.py` | `e3b0c44298fc1c14…` |
| `scripts/make_manifest.py` | `d8ff518da8e286f7…` |
| `scripts/rebuild_records.py` | `ee60babaa0926536…` |
| `scripts/reproduce_run.py` | `37c10372463fa9f3…` |
| `scripts/status.py` | `cc46d1109afeb05d…` |
| `validation/bond_separation.py` | `b66d855e600f4c0d…` |
| `validation/convergence_summary.py` | `5ef5c2c6f67aeeed…` |
| `validation/force_tests.py` | `1002680ab7f53e35…` |
| `validation/material_properties.py` | `da4ea45ffd4b2df3…` |
| `validation/md_demo.py` | `d2d002ce04c46581…` |
| `validation/performance.py` | `c60621f3dcd2b7c0…` |
| `validation/precision_tests.py` | `d47e596faef8552f…` |
| `validation/reference_aqs.py` | `80e4d50d7fbe5ab8…` |
| `validation/reference_aqs_rerun_torch.py` | `4047f255b4ddee91…` |
| `validation/suite.py` | `8d71f7b26c3ab8e7…` |
| `validation/validation_figures.py` | `dfbdaf40fd98d266…` |

## Modified relative to the phase-1 manifest (12 files)

| file | SHA-256 (this tree) | change |
|---|---|---|
| `HIERARCHY_FOLLOWUP_MEMO.md` | `4bcce2e9b0fd195e…` | copied from experiments/ of the phase-1 tree (the memo may have been edited after the manifest) |
| `README.md` | `11b98e88be21f0bb…` | copied after the manifest was written |
| `atomistica.log` | `cc9d6267e1dfd361…` | copied after the manifest was written |
| `environment.yml` | `20dc3c0ee6cf9025…` | numeric packages from pip (single OpenMP runtime), see the file header |
| `analysis/campaign_analysis.py` | `2a73e956d30eac83…` | `curve` reads trajectories through `db.trajectory_path` |
| `analysis/fracture_viz.py` | `925bcfe7cc5b1e81…` | trajectory path through `db.trajectory_path` |
| `experiments/campaign_design.py` | `10ea502bbe3682f8…` | identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit |
| `experiments/db.py` | `a391e8c4377e036a…` | read-only access to the phase-1 records (`reference_records`, `stress_strain`, `trajectory_path`); `load_record` falls back to the reference copy |
| `experiments/stage_tools.py` | `267db65abc09a249…` | identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit |
| `atomistics/structures/design_space.py` | `d23e323a0be746b5…` | phase-3 additions: `add_crack`, family `seam_pores`, `generate` handles crack keys, `offset` for `slit_array` and for the fine lattice of `nanomesh_hier` (defaults reproduce phase 1; tested) |
| `scripts/run_batch.py` | `482c9a3b3c5d42d2…` | identical to the 2026-09-23 12:58 version of the phase-1 tree (device `auto`); the manifest predates that edit |
| `scripts/run_queue.py` | `bb84308a3eaf7023…` | identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit |

## New in this tree (15 files)

| file | SHA-256 | purpose |
|---|---|---|
| `ASSESSMENT_phase1.md` | `6f1abb30f89a81a3…` | generated by analysis/assess_phase1.py |
| `analysis/advantage_figure.py` | `f4734f3955239420…` |  |
| `analysis/assess_phase1.py` | `8c0fdd959a8de4cf…` | assessment of the phase-1 database; defines the two-stage metrics; writes reference/phase1_reference_numbers.json |
| `analysis/design_gallery.py` | `f1db853ee32186ed…` |  |
| `analysis/hierarchy_followup_analysis.py` | `c39d58c8b90d57ee…` | evaluation against the pre-registration, tables, figures, fracture panels |
| `analysis/mechanism_T3.py` | `f3e2caee1802d91c…` |  |
| `experiments/campaign_T5_composites.py` | `a173c5a241bd561e…` |  |
| `experiments/campaign_hierarchy_followup.py` | `a61a00876aeb733f…` | the follow-up campaign: designs, geometry analysis, pre-registered predictions, specs |
| `tests/test_followup.py` | `e4c71f68fc7bd994…` | tests of the additions (phase-1 regeneration, cracks, seams, registry offsets, curve metrics) |
| `atomistics/structures/composite.py` | `c2ed7fd49c49bdc8…` |  |
| `atomistics/descriptors/columns.py` | `03b464b4aadc07b8…` |  |
| `scripts/merge_runs_from.py` | `42dd58d93b6bff03…` | copied from the phase-1 tree (post-dates its manifest) |
| `scripts/run_T5.sh` | `87e26cb45c52256d…` |  |
| `scripts/run_campaign.sh` | `19391a2d9bd4e435…` | launches the two queues (T1/T4/T2, then T3) with nohup |
| `scripts/write_provenance.py` | `232297ad18dc9c9a…` | this file |
