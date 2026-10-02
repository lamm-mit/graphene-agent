"""Audit completed loading cases without changing simulations or releasing the gate.

Recomputes endpoint integrals and stopping chronology independently of the
paired-metric helpers. Optional static checks use the existing CPU audit driver.
Every case review and static attempt must have a new output path.
"""
import argparse
import csv
import json
import math
import subprocess
import sys

from study_common import ROOT, EV, now, sha, write_json


def read_curve(path):
    with path.open() as stream:
        rows = [{k.split('#')[0].strip(): float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    # The archived pristine control has no per-frame wall-time observations.
    # Preserve that missing timing metadata; physical/stop fields must be finite.
    assert rows and all(math.isfinite(v) for r in rows for k, v in r.items() if k != 'wall_time')
    assert all(b['eps_x'] > a['eps_x'] for a, b in zip(rows, rows[1:]))
    return rows


def first_stop(rows, cfg):
    peak, peak_strain = 0., 0.
    for i, row in enumerate(rows[1:], 1):
        step = row.get('attempted_step', i)
        if step > cfg['max_steps']:
            return i - 1, 'max steps reached'
        stress, strain = row['sigma_xx'], row['eps_x']
        if stress > peak:
            peak, peak_strain = stress, strain
        damaged = row['n_broken_cum'] > 0
        if cfg['stop_when_not_spanning'] and not row['spanning']:
            return i, 'not spanning (fractured)'
        if damaged and stress < cfg['stop_stress_fraction'] * peak:
            return i, 'stress collapsed'
        if strain >= cfg['max_strain'] - 1e-12:
            return i, 'max strain reached'
        cap = cfg.get('post_peak_strain_limit')
        if cap is not None and damaged and strain >= peak_strain + cap - 1e-12:
            return i, 'post-peak strain limit reached (load still carried)'
        if damaged and stress < 0 and strain > peak_strain:
            return i, 'compressive after fracture'
        if step >= cfg['max_steps']:
            return i, 'max steps reached'
    return None


def scalar_check(rows, endpoint):
    prefix = rows[:endpoint['endpoint_index'] + 1]
    integral = math.fsum((b['eps_x'] - a['eps_x']) *
                         (a['sigma_xx'] + b['sigma_xx']) * .5 * EV
                         for a, b in zip(prefix, prefix[1:]))
    peak = max(prefix, key=lambda r: r['sigma_xx'])
    actual = dict(end_strain=prefix[-1]['eps_x'],
                  maximum_recorded_stress_N_m=peak['sigma_xx'] * EV,
                  strain_at_recorded_maximum=peak['eps_x'],
                  stress_strain_integral_J_m2=integral)
    for key, value in actual.items():
        assert math.isclose(value, endpoint[key], rel_tol=1e-10, abs_tol=1e-10), key
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    assert '/' not in args.stamp
    for case in args.cases:
        assert '/' not in case
        output = ROOT / 'results' / f'review_{case}.json'
        assert not output.exists(), f'Preserve existing review: {output}'
        run = ROOT / 'runs' / case
        record = json.loads((run / 'record.json').read_text())
        assert record['status'] == 'completed'
        base = ROOT / 'inputs/baseline/runs' / record['parent_run_id']
        assert sha(base / 'record.json') == record['parent_record_sha256']
        parent = json.loads((base / 'record.json').read_text())
        archived, rows = read_curve(base / 'stress_strain.csv'), read_curve(run / 'stress_strain.csv')
        paired = record['paired']
        a, b, c = [paired[k] for k in ['A_archived', 'B_original_protocol_on_rerun', 'C_extended']]
        independent = {name: scalar_check(curve, endpoint) for name, curve, endpoint in
                       [('A', archived, a), ('B', rows, b), ('C', rows, c)]}
        for cfg, endpoint in [(parent['aqs_config'], b), (record['aqs_config'], c)]:
            assert first_stop(rows, cfg) == (endpoint['endpoint_index'], endpoint['termination_reason'])
        assert c['endpoint_index'] == len(rows) - 1
        cfg = record['aqs_config']
        fbad = [r['max_force_eV_A'] > cfg['fmax'] for r in rows]
        tbad = [abs(r['sigma_yy']) > cfg['sigma_t_tol'] for r in rows]
        assert sum(fbad) == record['quality']['frames_above_force_tolerance']
        assert sum(tbad) == record['quality']['frames_above_transverse_tolerance']
        peak_index = max(range(len(rows)), key=lambda i: rows[i]['sigma_xx'])
        audit_path = None
        if args.static:
            attempt = f'original_setting_{case}_static_{args.stamp}'
            audit_path = ROOT / 'diagnostics' / attempt / 'record.json'
            assert not audit_path.parent.exists()
            indices = sorted({b['endpoint_index'], peak_index, c['endpoint_index']})
            with (ROOT / 'logs' / (attempt + '.log')).open('x') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts/audit_verified_loading_frames.py'),
                                case, attempt, *map(str, indices)], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads(audit_path.read_text())['status'] == 'completed'
        percentages = {}
        for label, start, end in [('A_to_B', a, b), ('B_to_C', b, c), ('A_to_C', a, c)]:
            percentages[label] = {key: 100 * (end[key] / start[key] - 1) for key in
                                  ['maximum_recorded_stress_N_m', 'stress_strain_integral_J_m2']}
        review = dict(reviewed_utc=now(), case_id=case, completed_utc=record['finished_utc'],
                      source_record_sha256=sha(run / 'record.json'), script_sha256=sha(__file__),
                      paired=paired, quality=record['quality'], independent_endpoint_checks=independent,
                      independent_stopping_chronology_pass=True, relative_changes_percent=percentages,
                      endpoint_force_eV_A=rows[-1]['max_force_eV_A'],
                      endpoint_transverse_stress_N_m=rows[-1]['sigma_yy'] * EV,
                      endpoint_residual_checks_pass=not (fbad[-1] or tbad[-1]),
                      residual_flagged_frames_union=sum(f or t for f, t in zip(fbad, tbad)),
                      independent_static_audit=str(audit_path.relative_to(ROOT)) if audit_path else None,
                      increment_convergence_established=False, production_approved=False,
                      assessment='Paired values and stopping chronology agree with independent recomputation. '
                      'Driver termination is distinct from residual convergence. Reproduction differences '
                      'are separated from added loading; no unique converged material-failure value is inferred.')
        write_json(output, review)
        print(json.dumps(dict(case=case,changes=percentages,endpoint_residual_checks_pass=review['endpoint_residual_checks_pass'],
                              residual_flagged_frames_union=review['residual_flagged_frames_union'])), flush=True)


if __name__ == '__main__':
    main()
