"""Reevaluate saved pilot/original configurations in CPU float64; no relaxation.

This diagnostic distinguishes real residual forces from an instrumentation or
precision artifact. It does not certify trajectory convergence or change runs.
"""
from pathlib import Path
import sys,json
from study_common import ROOT,CODE,EV,now,write_json,load_curve
sys.path.insert(0,str(CODE))
import numpy as np,torch
from ase import Atoms
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr

def main():
    torch.set_num_threads(2)
    eng=TorchRebo2Scr(device='cpu',dtype=torch.float64)
    rows=[]
    for name in ['ext__S2_slit_45deg','ext__S7_slit_20deg','ext__S7_slit_0deg_armchair']:
        run=ROOT/'runs'/name;rec=json.loads((run/'record.json').read_text());rid=rec['parent_run_id']
        base=ROOT/'inputs/baseline/runs'/rid
        orig=np.load(ROOT/'inputs/baseline/trajectories'/f'{rid}.npz')
        old_curve=load_curve(base/'stress_strain.csv');new_curve=load_curve(run/'stress_strain.csv')
        new_indices=sorted(set([int(np.argmax(new_curve['max_force_eV_A'])),len(new_curve['eps_x'])-1]))
        selections=[]
        for i in new_indices:
            frame=run/'frames'/f'{i:05d}.npz'
            if frame.exists():
                d=np.load(frame);selections.append(('rerun',i,d['positions'],d['cell'],new_curve))
        for i in sorted(set([len(old_curve['eps_x'])-1,int(np.argmin(abs(old_curve['eps_x']-new_curve['eps_x'][-1])))])):
            selections.append(('archived',i,orig['positions'][i],orig['cells'][i],old_curve))
        for source,i,positions,cell,c in selections:
            atoms=Atoms('C'*len(positions),positions=positions.astype(float),cell=cell,pbc=[True,True,False])
            out,_=eng.evaluate_atoms(atoms,skin=.3)
            force=out['forces'].detach().cpu().numpy();sigma=out['dE_deps'].detach().cpu().numpy()[0]/abs(np.linalg.det(cell[:2,:2]))*EV
            fmax=float(np.sqrt(np.sum(force**2,axis=1)).max())
            row=dict(case_id=name,source=source,frame_index=i,strain=float(c['eps_x'][i]),cpu_float64_max_force_eV_A=fmax,
                cpu_float64_sigma_xx_N_m=float(sigma[0,0]),cpu_float64_sigma_yy_N_m=float(sigma[1,1]),
                recorded_sigma_xx_N_m=float(c['sigma_xx'][i]*EV),recorded_sigma_yy_N_m=float(c['sigma_yy'][i]*EV),
                recorded_force_residual_eV_A=float(c['max_force_eV_A'][i]) if 'max_force_eV_A' in c else None,
                recorded_fire_converged=bool(c['fire_converged'][i]),force_tolerance_eV_A=rec['aqs_config']['fmax'])
            rows.append(row);print(json.dumps(row),flush=True)
    write_json(ROOT/'results/saved_frame_residual_checks.json',dict(created_utc=now(),engine=eng.info.__dict__,
        method='Static CPU float64 reevaluation of saved positions/cells; no relaxation. Selected worst-force and latest available rerun frames plus archived endpoint/matching-strain frames. Saved positions have float32 precision.',rows=rows))

if __name__=='__main__':main()
