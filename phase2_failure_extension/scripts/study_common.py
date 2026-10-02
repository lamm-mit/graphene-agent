"""Local paths, chronological stopping logic and censor-aware paired metrics."""
from __future__ import annotations
from pathlib import Path
import csv, datetime, hashlib, json, math, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
EV = 16.0217663

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()

def load_curve(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [{k.split('#')[0].strip(): float(v) for k, v in r.items()} for r in reader]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

def stop_event(c, cfg):
    """First chronological stop under cfg; max_steps includes rejected attempts.

    The driver checks physical failure before administrative limits. Its initial
    frame is recorded before the loading-loop peak bookkeeping begins.
    """
    e, s = np.asarray(c['eps_x']), np.asarray(c['sigma_xx'])
    peak, peak_e = 0.0, 0.0
    steps = c.get('attempted_step', np.arange(len(e)))
    for i in range(1, len(e)):
        if steps[i] > cfg['max_steps']:
            return dict(index=i-1, reason='max steps reached', failure_observed=False)
        if s[i] > peak: peak, peak_e = float(s[i]), float(e[i])
        damaged = c['n_broken_cum'][i] > 0
        if cfg['stop_when_not_spanning'] and not bool(c['spanning'][i]):
            return dict(index=i, reason='not spanning (fractured)', failure_observed=True)
        if damaged and s[i] < cfg['stop_stress_fraction'] * peak:
            return dict(index=i, reason='stress collapsed', failure_observed=True)
        if e[i] >= cfg['max_strain'] - 1e-12:
            return dict(index=i, reason='max strain reached', failure_observed=False)
        cap = cfg.get('post_peak_strain_limit')
        if cap is not None and damaged and e[i] >= peak_e + cap - 1e-12:
            return dict(index=i, reason='post-peak strain limit reached (load still carried)', failure_observed=False)
        if damaged and s[i] < 0 and e[i] > peak_e:
            return dict(index=i, reason='compressive after fracture', failure_observed=False)
        if steps[i] >= cfg['max_steps']:
            return dict(index=i, reason='max steps reached', failure_observed=False)
    return None

def endpoint_metrics(c, event=None, reason='observation in progress'):
    k = event['index'] if event else len(c['eps_x'])-1
    e = np.asarray(c['eps_x'][:k+1]); s = np.asarray(c['sigma_xx'][:k+1])*EV
    ip = int(np.argmax(s)); observed = bool(event and event['failure_observed'])
    first = np.flatnonzero(np.asarray(c['n_broken_cum'][:k+1]) > 0)
    out = dict(endpoint_index=k, end_strain=float(e[-1]), end_stress_N_m=float(s[-1]),
        failure_observed=observed, failure_strain_if_observed=float(e[-1]) if observed else None,
        last_nonfailed_strain=float(e[-2]) if observed and len(e)>1 else None,
        termination_reason=event['reason'] if event else reason,
        maximum_recorded_stress_N_m=float(s[ip]), strain_at_recorded_maximum=float(e[ip]),
        peak_at_last_frame=ip == k, stress_strain_integral_J_m2=float(np.trapezoid(s,e)),
        end_stress_fraction_of_maximum=float(s[-1]/s[ip]) if s[ip] else None,
        spanning_at_end=bool(c['spanning'][k]), first_damage_strain=float(e[first[0]]) if len(first) else None,
        recorded_frames=k+1, cumulative_broken_bond_events=int(c['n_broken_cum'][k]))
    return out

def paired_results(parent, original, new, extended_cfg, termination):
    b_event = stop_event(new, parent['aqs_config'])
    c_event = stop_event(new, extended_cfg)
    # A is deliberately the archived endpoint and convention, not silently redefined.
    a = endpoint_metrics(original, reason=parent['termination'])
    a.update(failure_observed=parent['metrics']['failure_observed'],
             failure_strain_if_observed=parent['metrics']['failure_strain'] if parent['metrics']['failure_observed'] else None,
             archived_reported_failure_strain=parent['metrics']['failure_strain'],
             archived_reported_work_to_failure_J_m2=parent['metrics']['work_to_failure_J_m2'])
    b = endpoint_metrics(new,b_event)
    c = endpoint_metrics(new,c_event,termination)
    pe = parent['aqs_config']['d_strain']
    shared = min(a['end_strain'], b['end_strain'])
    grid = original['eps_x'][original['eps_x'] <= shared+1e-12]
    observed_s = np.interp(grid, new['eps_x'], new['sigma_xx'])*EV
    original_s = original['sigma_xx'][:len(grid)]*EV
    norm = max(a['maximum_recorded_stress_N_m'],1e-12)
    err = dict(maximum_stress_relative=(b['maximum_recorded_stress_N_m']/a['maximum_recorded_stress_N_m']-1),
               integral_relative=(b['stress_strain_integral_J_m2']/a['stress_strain_integral_J_m2']-1),
               stop_strain_difference=b['end_strain']-a['end_strain'],
               first_damage_strain_difference=(b['first_damage_strain']-a['first_damage_strain']) if b['first_damage_strain'] is not None and a['first_damage_strain'] is not None else None,
               shared_interval_normalized_stress_rmse=float(np.sqrt(np.mean((observed_s-original_s)**2))/norm),
               curve_rmse_note='Linear interpolation is a diagnostic over the shared strain interval; event locations are also compared separately.')
    flags=[]
    for key in ['maximum_stress_relative','integral_relative']:
        if abs(err[key])>.02: flags.append(key+' exceeds 2% diagnostic trigger')
    for key in ['stop_strain_difference','first_damage_strain_difference']:
        if err[key] is not None and abs(err[key])>pe+1e-10: flags.append(key+' exceeds one parent increment')
    d = dict(additional_strain_B_to_C=c['end_strain']-b['end_strain'],
             additional_integral_B_to_C_J_m2=c['stress_strain_integral_J_m2']-b['stress_strain_integral_J_m2'],
             maximum_stress_change_B_to_C_N_m=c['maximum_recorded_stress_N_m']-b['maximum_recorded_stress_N_m'],
             integral_change_A_to_C_J_m2=c['stress_strain_integral_J_m2']-a['stress_strain_integral_J_m2'])
    return dict(A_archived=a,B_original_protocol_on_rerun=b,C_extended=c,
                virtual_stop_found=b_event is not None, reproduction=err,reproduction_flags=flags,changes=d)
