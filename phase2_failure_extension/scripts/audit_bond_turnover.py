"""Audit geometric bond turnover in saved trajectories; does not run simulations."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from ase.geometry import find_mic

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'results' / ('bond_turnover_audit_' + stamp)
    out.mkdir(exist_ok=False)
    report = {'created_utc': stamp, 'criterion': 'Geometric atom pairs at distance < 2 Angstrom; not an independent bond-order or chemical-validation calculation.',
              'completed_slit_totals': [], 'sources_sha256': {}}

    def source(p):
        report['sources_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

    for p in sorted((ROOT / 'runs').glob('ext__*slit*/trajectory.npz')):
        with np.load(p) as t:
            report['completed_slit_totals'].append({'case_id': p.parent.name,
                'broken_events': int(t['n_broken_new'].sum()), 'formed_events': int(t['n_formed_new'].sum())})
        source(p)
    run = ROOT / 'runs/ext__S7_slit_20deg'
    t = np.load(run / 'trajectory.npz')
    files = sorted((run / 'frames').glob('*.npz'))
    assert len(files) == len(t['eps_x'])
    states = []
    for k, p in enumerate(files):
        assert p.stem == f'{k:05d}'
        with np.load(p) as f:
            states.append({tuple(sorted(map(int, b))) for b in f['bonds'].tolist()})
        source(p)
    initial = states[0]
    seen = set(initial)
    events = []
    for k in range(1, len(states)):
        formed = states[k] - states[k-1]
        broken = states[k-1] - states[k]
        assert len(formed) == int(t['n_formed_new'][k])
        assert len(broken) == int(t['n_broken_new'][k])
        for pair in sorted(formed):
            events.append({'index': k, 'strain': float(t['eps_x'][k]), 'pair': list(pair),
                           'type': 'reappearance' if pair in seen else 'new_pair'})
        seen |= states[k]
    report['20deg_formation_events'] = events
    report['20deg_totals'] = {'new_pairs': sum(e['type'] == 'new_pair' for e in events),
                            'reappearance_events': sum(e['type'] == 'reappearance' for e in events),
                            'noninitial_pairs_at_endpoint': len(states[-1] - initial)}
    report['20deg_selected_states'] = []
    for k in [35, 51, 160]:
        pairs = np.array(sorted(states[k] - initial), dtype=int)
        row = {'index': k, 'strain': float(t['eps_x'][k]), 'noninitial_pairs_present': len(pairs),
               'cumulative_broken_events': int(t['n_broken_new'][:k+1].sum()),
               'cumulative_formed_events': int(t['n_formed_new'][:k+1].sum())}
        for state, label in [(0, 'initial'), (k, 'current')]:
            R = t['positions'][state]
            _, d = find_mic(R[pairs[:, 1]] - R[pairs[:, 0]], t['cells'][state], pbc=[True, True, False])
            row[label + '_distances_A'] = dict(zip(['minimum', 'median', 'maximum'], map(float, np.quantile(d, [0, .5, 1]))))
        report['20deg_selected_states'].append(row)
    report['interpretation'] = ('The 20-degree rerun has new neighboring atom pairs already by the virtual original stop. '
        'Distances at that state show these are not merely pairs flickering about the counting cutoff. '
        'Counts concern the new rerun, not a separate direct audit of the archived original trajectory. '
        'Formation events include repeated appearances and must not be described as independent healing events. '
        'The causal contribution of rebonding to peak stress or integrated work has not been isolated; '
        'the planar-path and numerical-convergence limitations remain.')
    source(Path(__file__))
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps({'output': str(out), 'totals': report['completed_slit_totals'], '20deg': report['20deg_totals']}, indent=2))


if __name__ == '__main__':
    main()
