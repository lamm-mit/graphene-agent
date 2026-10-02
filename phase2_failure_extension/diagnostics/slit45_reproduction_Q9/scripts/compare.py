"""Read-only, repeatable comparisons of saved paths; no simulation or manuscript writes."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parents[1]
PARENT = 'run_20260905_091359_4df2c2'
EV = 16.0217663

def edges(x):
    return {tuple(sorted(map(int, pair))) for pair in x}

def regular_path(name, folder, original=False):
    z = np.load(folder / 'trajectory.npz')
    if original:
        audit = json.loads((STUDY / 'diagnostics/archived_pilot_forces' / (PARENT + '.json')).read_text())
        force = np.array([r['max_force_eV_A'] for r in audit['rows']])
        bond = [edges(z[f'bonds_{i}']) for i in range(len(z['eps_x']))]
    else:
        force = z['max_force_eV_A']
        bond = []
        for i in range(len(z['eps_x'])):
            with np.load(folder / 'frames' / f'{i:05d}.npz') as f:
                bond.append(edges(f['bonds']))
    return dict(name=name, status='completed', strain=z['eps_x'], stress=z['sigma_xx'] * EV,
                force=force, transverse=z['sigma_yy'] * EV, broken=z['n_broken_cum'],
                positions=z['positions'], cells=z['cells'], bonds=bond)

def replay_path(job):
    folder = ROOT / 'runs' / job['id']
    if not (folder / 'curve.json').exists():
        return None
    # Atomic curve replacement means every referenced frame was fully persisted first.
    curve = json.loads((folder / 'curve.json').read_text())
    record = json.loads((folder / 'record.json').read_text())
    control_file = ROOT / 'results/execution_control.json'
    overrides = json.loads(control_file.read_text()).get('case_status_overrides', {}) if control_file.exists() else {}
    status = overrides.get(job['id'], record['status'])
    positions, cells, bonds = [], [], []
    for row in curve:
        with np.load(folder / 'frames' / f"{row['frame']:05d}.npz") as f:
            positions.append(f['positions'].copy())
            cells.append(f['cell'].copy())
            bonds.append(edges(f['bonds']))
    return dict(name=job['id'], status=status, strain=np.array([x['eps_x'] for x in curve]),
                stress=np.array([x['sigma'][0] * EV for x in curve]),
                transverse=np.array([x['sigma'][1] * EV for x in curve]),
                force=np.array([x['max_force_eV_A'] for x in curve]),
                broken=np.array([x['n_broken_cum'] for x in curve]),
                positions=np.array(positions), cells=np.array(cells), bonds=bonds)

def matched(a, b):
    result = []
    for i, strain in enumerate(a['strain']):
        matches = np.flatnonzero(np.isclose(b['strain'], strain, atol=1e-10, rtol=0))
        if not len(matches):
            continue
        assert len(matches) == 1
        j = int(matches[0])
        # Remove homogeneous cell deformation, minimum-image wrap in b's cell,
        # then remove rigid translation. Atom identities are preserved.
        ca, cb = a['cells'][i].astype(float), b['cells'][j].astype(float)
        assert np.array_equal(ca, np.diag(np.diag(ca))) and np.array_equal(cb, np.diag(np.diag(cb)))
        la, lb = np.diag(ca), np.diag(cb)
        assert np.all(la > 0) and np.all(lb > 0)
        delta = b['positions'][j].astype(float) - a['positions'][i].astype(float) * (lb / la)
        fractional = delta / lb
        fractional[:, :2] -= np.round(fractional[:, :2])
        delta = fractional * lb
        delta -= delta.mean(axis=0)
        assert np.isfinite(delta).all()
        result.append(dict(reference=a['name'], comparison=b['name'], strain=float(strain),
                           reference_frame=i, comparison_frame=j,
                           reference_stress_N_m=float(a['stress'][i]), comparison_stress_N_m=float(b['stress'][j]),
                           stress_difference_N_m=float(b['stress'][j] - a['stress'][i]),
                           rms_nonaffine_difference_A=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
                           maximum_nonaffine_difference_A=float(np.linalg.norm(delta, axis=1).max()),
                           edge_symmetric_difference=len(a['bonds'][i] ^ b['bonds'][j]),
                           reference_force_eV_A=float(a['force'][i]), comparison_force_eV_A=float(b['force'][j]),
                           reference_cumulative_broken=int(a['broken'][i]), comparison_cumulative_broken=int(b['broken'][j])))
    return result

def main():
    original = regular_path('Phase 1 single shot', ROOT / 'inputs' / PARENT, original=True)
    primary = regular_path('Existing Phase 2 primary', STUDY / 'runs/ext__S2_slit_45deg')
    half = regular_path('Existing Phase 2 half step', STUDY / 'runs/res__S2_slit_45deg')
    jobs = json.loads((ROOT / 'jobs.json').read_text())
    control_file = ROOT / 'results/execution_control.json'
    control = json.loads(control_file.read_text()) if control_file.exists() else {}
    required = [control['active_case']] if control else [j['id'] for j in jobs]
    paths = [original, primary, half]
    states = []
    for job in jobs:
        p = replay_path(job)
        record_file = ROOT / 'runs' / job['id'] / 'record.json'
        raw_status = json.loads(record_file.read_text())['status'] if record_file.exists() else 'pending'
        effective_status = control.get('case_status_overrides', {}).get(job['id'], raw_status)
        states.append(dict(id=job['id'], status=p['status'] if p else effective_status, frames=len(p['strain']) if p else 0))
        if p:
            paths.append(p)
    rows = []
    for p in paths[1:]:
        rows.extend(matched(original, p))
    # Also compare repeated and batched paths directly, without assuming one is truth.
    for i, p in enumerate(paths[3:]):
        for q in paths[4+i:]:
            rows.extend(matched(p, q))
    (ROOT / 'results/common_strain_comparisons.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    with (ROOT / 'results/common_strain_comparisons.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    plotted = []
    for p in paths:
        s = p['stress']; first = np.flatnonzero(p['broken'] > 0)
        summaries.append(dict(name=p['name'], status=p['status'], frames=len(s),
                              last_recorded_strain=float(p['strain'][-1]),
                              maximum_recorded_stress_N_m=float(s.max()),
                              strain_at_recorded_maximum=float(p['strain'][s.argmax()]),
                              first_recorded_damage_strain=float(p['strain'][first[0]]) if len(first) else None,
                              frames_above_force_target=int((p['force'] > .02).sum()),
                              frames_above_transverse_target=int((abs(p['transverse']) > .005 * EV).sum())))
        for k in range(len(s)):
            plotted.append(dict(path=p['name'], status=p['status'], frame=k, strain=float(p['strain'][k]),
                                stress_N_m=float(s[k]), cumulative_broken=int(p['broken'][k]),
                                max_force_eV_A=float(p['force'][k]), transverse_stress_N_m=float(p['transverse'][k])))
    with (ROOT / 'results/curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(plotted[0])); writer.writeheader(); writer.writerows(plotted)
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), scheduled_runs=states, summaries=summaries,
                  required_case_ids=required,
                  scheduled_runs_complete=all(s['status'] == 'completed' for s in states if s['id'] in required),
                  original_three_run_schedule_complete=all(s['status'] == 'completed' for s in states),
                  note='Incomplete-path maxima and last saved strains are progress values, not final strengths or failure strains. Force audits of Phase 1 use CPU float64 saved states; Phase 2 and Q9 use recorded MPS residuals. No interpolation of configurations or graph events.',
                  displacement_definition='Map reference positions affinely into comparison cell, apply xy minimum image, remove mean translation; RMS norm over atoms in Angstrom.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT / 'results/comparison_status.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'scheduled_runs': states, 'matched_rows': len(rows), 'all_scheduled_runs_complete': result['scheduled_runs_complete']}, indent=2))

if __name__ == '__main__':
    main()
