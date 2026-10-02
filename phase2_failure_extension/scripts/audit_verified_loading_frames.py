"""Independent CPU/float64 static checks of selected accepted Q4 frames.

No minimization, restart or replacement of the loading path is performed.
Each invocation creates a new attempt folder and copies the exact source frames.
"""
import argparse
import json
import shutil
import sys

from study_common import ROOT, EV, now, sha, write_json
from analyze import load_live_curve
sys.path.insert(0, str(ROOT / 'code_verified'))
import numpy as np
import torch
from ase import Atoms
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case_id')
    parser.add_argument('attempt')
    parser.add_argument('indices', nargs='+', type=int)
    args = parser.parse_args()
    assert '/' not in args.attempt and '/' not in args.case_id
    outdir = ROOT / 'diagnostics' / args.attempt
    outdir.mkdir(exist_ok=False)
    run = ROOT / 'runs' / args.case_id
    r = json.loads((run / 'record.json').read_text())
    c = load_live_curve(run / 'stress_strain.csv')
    shutil.copy2(run / 'record.json', outdir / 'source_record.json')
    audit = dict(created_utc=now(), case_id=args.case_id, status='running',
        method='Static CPU float64 reevaluation of saved float32 positions and cells. No relaxation.',
        script_sha256=sha(__file__), selected_frame_indices=args.indices, rows=[])
    write_json(outdir / 'record.json', audit)
    torch.set_num_threads(2)
    eng = TorchRebo2Scr(device='cpu', dtype=torch.float64)
    assert eng.info.parameter_checksum == r['engine']['parameter_checksum']
    try:
        for i in args.indices:
            assert 0 <= i < len(c['eps_x'])
            src = run / 'frames' / f'{i:05d}.npz'
            saved = outdir / src.name
            shutil.copy2(src, saved)
            d = np.load(saved)
            cell = d['cell'].astype(float)
            atoms = Atoms('C' * len(d['positions']), positions=d['positions'].astype(float), cell=cell, pbc=[True,True,False])
            ev, _ = eng.evaluate_atoms(atoms, skin=r['aqs_config']['skin'])
            fm = float(torch.linalg.vector_norm(ev['forces'], dim=1).max())
            stress = ev['dE_deps'].detach().cpu().numpy()[0] / abs(np.linalg.det(cell[:2,:2])) * EV
            row = dict(frame_index=i, strain=float(c['eps_x'][i]), source_sha256=sha(saved),
                recorded_force_eV_A=float(c['max_force_eV_A'][i]), cpu_force_eV_A=fm,
                recorded_axial_stress_N_m=float(c['sigma_xx'][i]*EV), cpu_axial_stress_N_m=float(stress[0,0]),
                recorded_transverse_stress_N_m=float(c['sigma_yy'][i]*EV), cpu_transverse_stress_N_m=float(stress[1,1]),
                cpu_force_pass=bool(fm<r['aqs_config']['fmax']),
                cpu_transverse_pass=bool(abs(stress[1,1])<=r['aqs_config']['sigma_t_tol']*EV))
            audit['rows'].append(row)
            write_json(outdir / 'record.json', audit)
            print(json.dumps(row), flush=True)
        audit.update(status='completed', finished_utc=now())
    except BaseException as error:
        audit.update(status='execution_error', finished_utc=now(), error=repr(error))
        raise
    finally:
        write_json(outdir / 'record.json', audit)


if __name__ == '__main__':
    main()
