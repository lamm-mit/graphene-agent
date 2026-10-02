"""Q5: fixed-cell convergence probe of the first completed pilot endpoint."""
import dataclasses,json,os,sys,traceback
from study_common import ROOT,CODE,EV,now,write_json,sha
sys.path.insert(0,str(CODE))
import numpy as np,torch
from ase import Atoms
from ase.io import write
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.minimization.fire import fire_minimize
from simulation.aqs.aqs import AQSRunner,AQSConfig
from simulation.topology import spanning_x

def main():
    amendment=json.loads((ROOT/'protocol/amendment_Q5.json').read_text())
    dest=ROOT/'diagnostics/frame_relaxation_Q5';dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);eng=TorchRebo2Scr(device='mps',dtype=torch.float32)
    record=dict(started_utc=now(),status='running',pid=os.getpid(),engine=dataclasses.asdict(eng.info),script_sha256=sha(__file__),amendment=amendment,rows=[])
    write_json(dest/'record.json',record)
    try:
        for selection in amendment['frames']:
            source=ROOT/selection['source'];d=np.load(source)
            atoms=Atoms('C'*len(d['positions']),positions=d['positions'],cell=d['cell'],pbc=[True,True,False])
            folder=dest/f"{selection['case_id']}__frame{selection['frame_index']:05d}";folder.mkdir()
            write(folder/'initial.extxyz',atoms)
            system=BatchedSystem([atoms],eng);nl=system.build_neighbor_list(skin=.3)
            probe=AQSRunner(eng,system,AQSConfig(),log=None)
            def evaluate():
                nonlocal nl
                probe.nl=nl
                out=probe._evaluate(per_atom=False);nl=probe.nl
                sigma=out['sigma2d'][0]*EV
                i,j,shift=probe._bonds(out)[0]
                return dict(max_force_eV_A=float(torch.linalg.vector_norm(out['forces'],dim=1).max().cpu()),
                    sigma_xx_N_m=float(sigma[0]),sigma_yy_N_m=float(sigma[1]),energy_eV=float(out['energy'][0].cpu()),
                    spanning=bool(spanning_x(i,j,shift,len(atoms))),
                    below_frozen_running_peak_threshold=bool(sigma[0]<amendment['running_peak_stress_N_m']*amendment['stress_fraction_threshold']))
            before=evaluate()
            def log(message):
                print(now(),selection['case_id'],message,flush=True)
                write_json(dest/'progress.json',dict(updated_utc=now(),case_id=selection['case_id'],message=message,completed=len(record['rows']),planned=len(amendment['frames'])))
            nl,result=fire_minimize(eng,system,nl,log=log,**amendment['minimizer_settings'])
            after=evaluate();write(folder/'final.extxyz',system.to_atoms(0))
            np.savez_compressed(folder/'relaxed.npz',positions=system.positions_numpy(0),cell=system.cells_numpy()[0])
            row=dict(**selection,source_sha256=sha(source),before=before,after=after,force_converged=after['max_force_eV_A']<.02,minimizer=dataclasses.asdict(result))
            write_json(folder/'result.json',row);record['rows'].append(row);write_json(dest/'record.json',record)
            print(json.dumps(dict(case=selection['case_id'],before=before,after=after)),flush=True)
        record.update(status='completed',finished_utc=now())
    except BaseException as error:
        record.update(status='execution_error',finished_utc=now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(dest/'record.json',record)

if __name__=='__main__':main()
