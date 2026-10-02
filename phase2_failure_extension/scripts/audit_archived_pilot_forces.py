"""Static CPU/float64 force audit of every saved frame in the eight pilot parents."""
import json,sys,os,dataclasses
from study_common import ROOT,CODE,EV,now,write_json,sha,load_curve
sys.path.insert(0,str(CODE))
import numpy as np,torch
from ase import Atoms
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr

def main():
    dest=ROOT/'diagnostics/archived_pilot_forces';dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);eng=TorchRebo2Scr(device='cpu',dtype=torch.float64)
    jobs=[j for j in json.loads((ROOT/'protocol/jobs.json').read_text()) if j['stage']=='pilot']
    summaries=[]
    for job in jobs:
        rid=job['parent_run_id'];base=ROOT/'inputs/baseline/runs'/rid
        parent=json.loads((base/'record.json').read_text());source=ROOT/'inputs/baseline/trajectories'/f'{rid}.npz'
        data=np.load(source);curve=load_curve(base/'stress_strain.csv');rows=[]
        for i,(positions,cell) in enumerate(zip(data['positions'],data['cells'])):
            atoms=Atoms('C'*len(positions),positions=positions.astype(float),cell=cell,pbc=[True,True,False])
            out,_=eng.evaluate_atoms(atoms,skin=.3)
            force=out['forces'].detach().cpu().numpy();fmax=float(np.linalg.norm(force,axis=1).max())
            rows.append(dict(frame_index=i,strain=float(curve['eps_x'][i]),max_force_eV_A=fmax,
                archived_fire_converged=bool(curve['fire_converged'][i]),above_nominal_tolerance=fmax>parent['aqs_config']['fmax']))
        summary=dict(case_id=job['case_id'],parent_run_id=rid,n_frames=len(rows),
            frames_above_nominal_force_tolerance=sum(r['above_nominal_tolerance'] for r in rows),
            frames_above_5x_force_tolerance=sum(r['max_force_eV_A']>5*parent['aqs_config']['fmax'] for r in rows),
            maximum_force_residual_eV_A=max(r['max_force_eV_A'] for r in rows),
            force_tolerance_eV_A=parent['aqs_config']['fmax'],source_sha256=sha(source))
        write_json(dest/f'{rid}.json',dict(summary=summary,rows=rows));summaries.append(summary)
        write_json(dest/'summary.json',dict(updated_utc=now(),status='completed' if len(summaries)==len(jobs) else 'running',pid=os.getpid(),
            method='CPU float64 static reevaluation of archived float32 saved coordinates, without relaxation. Near-tolerance differences can reflect rounding; large exceedances are not a precision artifact.',
            engine=dataclasses.asdict(eng.info),script_sha256=sha(__file__),cases=summaries))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
