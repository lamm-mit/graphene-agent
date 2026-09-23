# Carbon architecture & fracture discovery platform (screened REBO2, PyTorch, ASE)

A browser-based generative-design and scientific-discovery platform for atomically resolved graphene-derived
carbon architectures. The mechanics come exclusively from the published **screened second-generation REBO
potential (REBO2+S / Rebo2Scr)** — implemented exactly in PyTorch (`TorchRebo2Scr`, validated to 1e-13 eV
against Atomistica) and run on Apple MPS / CUDA / CPU — with athermal quasi-static (AQS) tensile loading.

## Layout
```
app/                 FastAPI backend + browser frontend (three.js viewer)          -> scripts/launch_app.sh
atomistics/          graphene builders, image-derived generator families, descriptors, exporters (ASE)
potentials/rebo2scr/ reference parameters/tables, PyTorch engine, neighbour lists, ASE calculator, tests
simulation/          batched system, FIRE minimiser, AQS driver, MD, topology/damage analysis
validation/          TESTS 1-20 (scripts + results/*.json), validation figures
experiments/         campaign design, specs (job queue), database (one dir per run), predictions, holdouts
analysis/            metrics, campaign analysis, fracture visualisation (panels, movies), image interpretation
figures/ movies/     publication figures (png/svg/pdf + captions), MP4 fracture movies
final_designs/       top-structure selection + structures
report/              LaTeX report (report.tex -> report.pdf), bibliography, generated tables
trajectories/        raw AQS trajectories (npz: positions, cells, per-atom energy/virial, bonds per frame); 17 in git,
                     all 132 on Hugging Face -> ../scripts/fetch_data_from_huggingface.py
tests/               pytest suite
```

## Quick start
```bash
# environment: see ../environment.yml or ../requirements.txt (the campaign used a conda env with torch 2.7, ase 3.27, atomistica 1.2.7)
PY=python
$PY -m pytest -q tests/                      # automated tests (~2 min)
$PY validation/suite.py                      # validation table (fast tests) -> validation/results/validation_table.json
./scripts/launch_app.sh                      # http://127.0.0.1:8766
$PY scripts/run_batch.py experiments/specs/stage1/stage1_b0.json    # run one batch spec
$PY scripts/run_queue.py experiments/specs/stage2 --workers 3       # run a queue of specs concurrently (MPS)
$PY scripts/reproduce_run.py <run_id>        # reproduce a stored run from its record
$PY scripts/merge_runs_from.py <other_tree> --apply   # copy-only merge of runs made in another copy of the tree (never overwrites)
$PY analysis/campaign_analysis.py            # campaign figures from the database
$PY final_designs/select_top.py              # top structures, progression panels, movies
$PY report/results_body_template.py && $PY report/build_report.py   # compile the report
./scripts/package.sh                         # manifest + ZIP
```

## Force engine
`potentials/rebo2scr/pytorch/torch_rebo2scr.py` — energy, forces (autograd), virial, per-atom energies/virials;
`device="auto"` selects CUDA > MPS > CPU. Fast mode: MPS/CUDA float32. Reference mode: CPU float64 (bit-level
agreement with Atomistica `Rebo2Scr()`). The Fortran reference (Atomistica 1.2.7) is used as the numerical oracle
in `validation/`.

## Units
eV, Å; 2D stress in N/m (1 eV/Å² = 16.0218 N/m). No implicit conversion to GPa.

## Performance (measured; `validation/results/performance.json`, Fig. `figures/validation/performance`)

Apple M4 Max, torch 2.7.0, energy + forces + virial per evaluation: Torch MPS float32 about 9 us/atom above 3000 atoms (launch-bound below);
Torch CPU float32 about 17 us/atom; Torch CPU float64 about 26 us/atom; Atomistica Fortran reference (CPU, float64) about 3 us/atom.
The GPU port is therefore *not* faster than the Fortran reference per evaluation on this machine. It was used because it provides
batched evaluation of many structures in one call, per-atom energies and virials, autograd-exact forces/stresses from the same energy code,
and concurrent use of the GPU by several simulation processes (1 / 2 / 4 processes: 34 / 59 / 73 evaluations per second at about 3000 atoms),
which is how the campaign job queue ran. No CUDA device was available; the CUDA code path is identical but unbenchmarked.

## Reproduction check

`scripts/reproduce_run.py <run_id>` regenerates a stored run from its record. For the pristine zigzag baseline (float32, MPS) the
regenerated run reproduced the modulus to five digits (250.07 N/m), the strength to 0.8 % (38.47 vs 38.79 N/m), the fracture mode,
and the failure strain to within one strain increment (0.261 vs 0.271); the residual comes from non-deterministic float32 reductions
on the GPU (TEST 17) amplified at the intrinsic instability of the perfect lattice (`validation/results/reproduce_check_S1_pristine_zz.txt`).

## Property charts and atlas

- `analysis/ashby_chart.py` -- Ashby-style property charts (strength/modulus vs relative areal density with family envelopes and specific-property guidelines, strength vs work to failure, strength vs failure strain, thumbnails of representative structures) and an atlas of all simulated architectures -> `figures/campaign/ashby_overview.*`, `figures/campaign/architecture_atlas.*`.
