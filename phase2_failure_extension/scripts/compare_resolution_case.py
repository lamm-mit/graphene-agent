"""Compare a completed scheduled resolution check with its primary extension.

Read-only with respect to simulations. Each review gets a new output filename.
This is a sensitivity comparison; two rerun paths do not establish convergence.
"""
import argparse
import json
from pathlib import Path

from study_common import ROOT, now, sha
from review_loading_cases import read_curve, scalar_check, first_stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case_id')
    args = parser.parse_args()
    case = args.case_id
    assert case.startswith('res__') and '/' not in case
    primary = 'ext__' + case.removeprefix('res__')
    output = ROOT / 'results' / f'resolution_comparison_{case}.json'
    assert not output.exists(), f'Preserve existing review: {output}'
    records = {}
    hashes = {}
    for label, case_id in [('primary', primary), ('half_post_damage_step', case)]:
        folder = ROOT / 'runs' / case_id
        record = json.loads((folder / 'record.json').read_text())
        assert record['status'] == 'completed'
        rows = read_curve(folder / 'stress_strain.csv')
        for endpoint in ['B_original_protocol_on_rerun', 'C_extended']:
            scalar_check(rows, record['paired'][endpoint])
        c = record['paired']['C_extended']
        assert first_stop(rows, record['aqs_config']) == (c['endpoint_index'], c['termination_reason'])
        records[label] = record
        hashes[label] = {name: sha(folder / name) for name in ['record.json', 'stress_strain.csv', 'initial.extxyz']}
    coarse, fine = records.values()
    assert coarse['parent_run_id'] == fine['parent_run_id']
    assert coarse['paired']['A_archived'] == fine['paired']['A_archived']
    assert hashes['primary']['initial.extxyz'] == hashes['half_post_damage_step']['initial.extxyz']
    differences = {k: [v, fine['aqs_config'][k]] for k, v in coarse['aqs_config'].items()
                   if v != fine['aqs_config'][k]}
    assert set(differences) == {'d_strain'}
    assert fine['aqs_config']['d_strain'] * 2 == coarse['aqs_config']['d_strain']
    metrics = ['end_strain', 'maximum_recorded_stress_N_m',
               'strain_at_recorded_maximum', 'stress_strain_integral_J_m2']
    comparisons = {}
    for endpoint in ['B_original_protocol_on_rerun', 'C_extended']:
        a, b = [r['paired'][endpoint] for r in [coarse, fine]]
        comparisons[endpoint] = {
            k: {'primary': a[k], 'half_post_damage_step': b[k], 'difference': b[k] - a[k],
                'relative_change_percent': 100 * (b[k] / a[k] - 1) if a[k] else None}
            for k in metrics}
        comparisons[endpoint]['termination_reason'] = [a['termination_reason'], b['termination_reason']]
        comparisons[endpoint]['spanning_at_end'] = [a['spanning_at_end'], b['spanning_at_end']]
    result = dict(reviewed_utc=now(), resolution_case=case, primary_case=primary,
                  parent_run_id=coarse['parent_run_id'], source_hashes=hashes,
                  script_sha256=sha(Path(__file__)), initial_geometry_hash_equal=True,
                  configuration_differences=differences, comparisons=comparisons,
                  paired={k: r['paired'] for k, r in records.items()},
                  quality={k: r['quality'] for k, r in records.items()},
                  independently_recomputed_metrics_and_stop_pass=True,
                  increment_convergence_established=False,
                  interpretation='Independent reruns with only the post-damage strain increment changed. '
                  'Elastic and minimum refinement increments are unchanged. Differences include path '
                  'sensitivity and potentially floating-point variability; this is not a deterministic '
                  'error estimate or a converged replacement. Retain residual flags and both branches.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(output.relative_to(ROOT)),
                      'extended_endpoint_comparison': comparisons['C_extended']}, indent=2))


if __name__ == '__main__':
    main()
