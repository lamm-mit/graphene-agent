"""Audit the user's question about the black Phase-1 curve, seed and rerun setup."""
from pathlib import Path
import json,hashlib,shutil,csv
import numpy as np
from ase.io import read
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];BASE=STUDY.parent
parent=STUDY/'inputs/baseline/runs/run_20260905_091359_4df2c2'
a=json.loads((parent/'record.json').read_text());x=read(parent/'initial.extxyz')
original=BASE/'carbon_discovery'; originaltraj=original/a['trajectory']; copied=STUDY/'inputs/baseline/trajectories'/originaltraj.name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(originaltraj)==sha(copied)
az=np.load(copied);EV=16.0217663
spec=original/'experiments/specs/stage2/done/stage2_b06.json';shutil.copy2(spec,ROOT/'data/original_batch_spec.json');batch=json.loads(spec.read_text())
plotted=list(csv.DictReader((ROOT/'data/curves.csv').open()));branches=[]
for name,label in [('ext__S2_slit_45deg','Phase 2 primary extension'),('res__S2_slit_45deg','Phase 2 resolution check')]:
 p=STUDY/'runs'/name;b=json.loads((p/'record.json').read_text());y=read(p/'initial.extxyz');z=np.load(p/'trajectory.npz')
 assert np.array_equal(x.positions,y.positions) and np.array_equal(x.cell.array,y.cell.array)
 assert b['initial_geometry_sha256']==sha(parent/'initial.extxyz')
 assert b['engine']==a['engine']
 rows=[r for r in plotted if r['branch']==label]
 assert np.array_equal(np.array([float(r['strain']) for r in rows]),z['eps_x'])
 assert np.array_equal(np.array([float(r['stress_N_m']) for r in rows]),z['sigma_xx']*EV)
 branches.append(dict(case=name,initial_geometry_exactly_matches=True,engine_record_exactly_matches=True,
  settings_changes={k:[a['aqs_config'].get(k),v] for k,v in b['aqs_config'].items() if a['aqs_config'].get(k)!=v},
  relaxed_initial_max_coordinate_difference_A=float(np.max(abs(az['positions'][0]-z['positions'][0]))),
  relaxed_initial_max_cell_difference_A=float(np.max(abs(az['cells'][0]-z['cells'][0]))),
  first_damage_strain=b['paired']['C_extended']['first_damage_strain'],
  maximum_recorded_stress_N_m=float(max(z['sigma_xx'])*EV),
  source_environment=b['environment'],plotted_curve_exact=True))
rows=[r for r in plotted if r['branch']=='Phase 1 single shot']
assert np.array_equal(np.array([float(r['strain']) for r in rows]),az['eps_x'])
assert np.array_equal(np.array([float(r['stress_N_m']) for r in rows]),az['sigma_xx']*EV)
unchanged=[]
for folder in ['potentials/rebo2scr/pytorch','simulation/minimization']:
 for p in (STUDY/'source_original'/folder).rglob('*.py'):
  rel=p.relative_to(STUDY/'source_original');q=STUDY/'code'/rel
  assert sha(p)==sha(q);unchanged.append(str(rel))
report=dict(original_name=a['name'],parent_run_id=a['run_id'],original_seed=a['seed'],atoms=len(x),
 original_curve_matches_source_file=True,plotted_original_curve_exact=True,
 original_stop=a['termination'],nominal_original_total_strain_limit=a['aqs_config']['max_strain'],
 actual_original_endpoint=a['metrics']['failure_strain'],original_failure_observed=a['metrics']['failure_observed'],
 original_batch_size=len(batch['structures']),original_batch_names=[s['name'] for s in batch['structures']],rerun_batch_size=1,
 original_environment=a['environment'],original_maximum_recorded_stress_N_m=a['metrics']['strength_Nm'],
 original_first_damage_strain=a['first_damage_strain'],force_kernel_and_minimizer_unchanged=unchanged,
 rerun_randomized_initial_conditions=False,branches=branches,
 cause_assessment='Numerical/path sensitivity is observed, including different relaxed initial states and later connection histories. Batch layout and OS/backend environment differ; float32 arithmetic, finite minimization tolerance and first-damage step refinement are relevant candidates. Their individual causal contributions have not been isolated. This is not an intentional different-seed experiment, and neither original nor rerun curves are established as converged truth.',
 continuity='Reruns begin at the original initial geometry; neither is a continuation from the saved Phase-1 endpoint.',
 source_hashes={str(p.relative_to(STUDY)):sha(p) for p in [parent/'record.json',parent/'initial.extxyz',copied,ROOT/'data/curves.csv',ROOT/'data/original_batch_spec.json']})
(ROOT/'qa/reproduction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'REPRODUCTION_NOTE.md').write_text('''# Why the 45° curves differ

The black curve is the original Phase 1 single-shot path, verified against the source trajectory and plotted data. It reached the prescribed total-strain limit of 0.32, with the discrete endpoint at 0.32375, still carrying load. It did not terminate at the 0.12 post-peak rule or an observed failure.

Both Phase 2 curves are new runs from exactly the same saved initial coordinates and cell (3,088 atoms; original geometry seed 0). They are not continuations from the last black-curve configuration. The recorded force-field checksum, float32 MPS engine and PyTorch version agree. The primary settings change only the stopping limits; the resolution branch also halves the post-damage increment. No new randomized initial conditions or thermal noise are introduced by this AQS run path.

The original case ran in a four-structure batch; each rerun ran individually. The original record lists macOS 26.6.2, the reruns macOS 27.0. Relaxed starting coordinates already differ: the largest component differences are about 0.000206 Å for the primary and 0.01065 Å for the half-step run. These are observations, not proof that one particular backend or batching difference caused the later divergence.

All paths form the first four persistent new pairs near strain 0.17. They then follow different reconnection and damage histories. By the last sample at or below 0.20, qualifying distinct new pairs number 128, 66 and 52 for the original, primary and half-step paths. Recorded maxima are 18.54, 12.20 and 11.45 N/m, respectively. The original-versus-primary peak difference is already present inside the original observation window; loading the primary farther does not change its maximum.

The plotting and input identity are verified; quantitative numerical convergence is not. AQS trajectories contain minimizer/residual qualifications, and these comparisons do not isolate a unique cause. It is inappropriate to call the difference a different random seed, a measured physical randomness distribution, or an effect of longer loading alone. Isolating the cause would require controlled repeatability, batch-layout and stronger numerical-convergence tests; none were silently launched for this review.

See qa/reproduction_audit.json and scripts/check_reproduction.py for reproducible checks.
''')
print(json.dumps({k:report[k] for k in ['original_seed','atoms','original_curve_matches_source_file','plotted_original_curve_exact','original_batch_size','rerun_batch_size','actual_original_endpoint']},indent=2))
