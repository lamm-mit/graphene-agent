"""Q2: fixed-cell relaxation probes, never a resumed loading trajectory."""
import json,sys,dataclasses,os,traceback
from pathlib import Path
from study_common import ROOT,CODE,EV,now,write_json,sha
sys.path.insert(0,str(CODE))
import numpy as np, torch
from ase import Atoms
from ase.io import write
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.minimization.fire import fire_minimize

def main():
    manifest=json.loads((ROOT/'protocol/amendment_Q2.json').read_text())
    dest=ROOT/'diagnostics/frame_relaxation_Q2';dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    eng=TorchRebo2Scr(device='mps',dtype=torch.float32)
    meta=dict(started_utc=now(),status='running',pid=os.getpid(),engine=dataclasses.asdict(eng.info),
              script_sha256=sha(__file__),amendment=manifest)
    write_json(dest/'record.json',meta)
    rows=[]
    try:
        for selection in manifest['frames']:
            source=ROOT/'runs'/selection['case_id']/'frames'/f"{selection['frame_index']:05d}.npz"
            d=np.load(source);atoms=Atoms('C'*len(d['positions']),positions=d['positions'],cell=d['cell'],pbc=[True,True,False])
            outdir=dest/f"{selection['case_id']}__frame{selection['frame_index']:05d}";outdir.mkdir()
            write(outdir/'initial.extxyz',atoms)
            sys_=BatchedSystem([atoms],eng);nl=sys_.build_neighbor_list(skin=.3)
            def evaluate():
                nonlocal nl
                if eng.needs_rebuild(sys_.positions,sys_.cell,sys_.sid,nl):nl=sys_.build_neighbor_list(skin=.3)
                out=eng.evaluate(sys_.positions,sys_.cell,sys_.sid,nl,compute_stress=True)
                sigma=out['dE_deps'][0].detach().cpu().numpy()/float(sys_.areas()[0].cpu())*EV
                return dict(max_force_eV_A=float(torch.sqrt((out['forces']**2).sum(1)).max().cpu()),
                    sigma_xx_N_m=float(sigma[0,0]),sigma_yy_N_m=float(sigma[1,1]),energy_eV=float(out['energy'][0].cpu()))
            before=evaluate()
            nl,res=fire_minimize(eng,sys_,nl,fmax=.02,max_steps=8000,skin=.3,stall_iters=10**9)
            after=evaluate();write(outdir/'final.extxyz',sys_.to_atoms(0))
            np.savez_compressed(outdir/'relaxed.npz',positions=sys_.positions_numpy(0),cell=sys_.cells_numpy()[0])
            row=dict(**selection,source_sha256=sha(source),before=before,after=after,
                force_converged=after['max_force_eV_A']<.02,minimizer=dataclasses.asdict(res))
            rows.append(row);write_json(outdir/'result.json',row)
            write_json(dest/'progress.json',dict(updated_utc=now(),completed=len(rows),planned=len(manifest['frames']),rows=rows))
            print(json.dumps(dict(case=selection['case_id'],frame=selection['frame_index'],before=before,after=after)),flush=True)
        meta.update(status='completed',finished_utc=now(),rows=rows)
    except BaseException as error:
        meta.update(status='execution_error',finished_utc=now(),error=repr(error),traceback=traceback.format_exc(),rows=rows)
        raise
    finally:write_json(dest/'record.json',meta)

if __name__=='__main__':main()
