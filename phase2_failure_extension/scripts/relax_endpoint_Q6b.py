"""Q6b: independent CPU/float64 L-BFGS-B fixed-cell endpoint diagnostic."""
import json,sys,os,dataclasses,time,traceback
from study_common import ROOT,CODE,EV,now,write_json,sha
sys.path.insert(0,str(CODE))
import numpy as np,torch,scipy
from scipy.optimize import minimize
from ase import Atoms
from ase.io import write
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner,AQSConfig
from simulation.topology import spanning_x

def main():
    amendment=json.loads((ROOT/'protocol/amendment_Q6b.json').read_text())
    dest=ROOT/'diagnostics/endpoint_lbfgs_Q6b';dest.mkdir(parents=True,exist_ok=False)
    source=ROOT/amendment['source'];d=np.load(source)
    atoms=Atoms('C'*len(d['positions']),positions=d['positions'].astype(float),cell=d['cell'],pbc=[True,True,False])
    write(dest/'initial.extxyz',atoms)
    torch.set_num_threads(2);eng=TorchRebo2Scr(device='cpu',dtype=torch.float64)
    system=BatchedSystem([atoms],eng);probe=AQSRunner(eng,system,AQSConfig(),log=None)
    probe.nl=system.build_neighbor_list(skin=.3)
    record=dict(status='running',started_utc=now(),pid=os.getpid(),amendment=amendment,source_sha256=sha(source),
                script_sha256=sha(__file__),engine=dataclasses.asdict(eng.info),scipy_version=scipy.__version__)
    write_json(dest/'record.json',record)
    counter=0;iterations=0;cache={};history=[];start=time.time()
    def evaluate(x):
        nonlocal counter
        system.positions=torch.as_tensor(x.reshape(-1,3).copy(),dtype=torch.float64)
        out=probe._evaluate(per_atom=False)
        energy=float(out['energy'][0].cpu());force=out['forces'].cpu().numpy();sigma=out['sigma2d'][0]*EV
        i,j,shift=probe._bonds(out)[0]
        values=dict(energy_eV=energy,max_force_eV_A=float(np.linalg.norm(force,axis=1).max()),
                    sigma_xx_N_m=float(sigma[0]),sigma_yy_N_m=float(sigma[1]),spanning=bool(spanning_x(i,j,shift,len(atoms))),
                    below_frozen_running_peak_threshold=bool(sigma[0]<amendment['running_peak_stress_N_m']*amendment['stress_fraction_threshold']))
        counter+=1;cache.update(x=x.copy(),metrics=values)
        return energy,-force.ravel().copy()
    def callback(x):
        nonlocal iterations
        iterations+=1
        if not np.array_equal(x,cache['x']):evaluate(x)
        if iterations==1 or iterations%10==0:
            row=dict(iteration=iterations,evaluations=counter,elapsed_s=time.time()-start,**cache['metrics'])
            history.append(row);write_json(dest/'progress.json',dict(updated_utc=now(),**row))
            print(json.dumps(row),flush=True)
        if iterations%50==0:
            np.savez_compressed(dest/f'checkpoint_{iterations:04d}.npz',positions=x.reshape(-1,3),cell=d['cell'])
    try:
        x0=atoms.positions.ravel().copy();energy,gradient=evaluate(x0);record['before']=cache['metrics'].copy()
        direction=gradient/np.linalg.norm(gradient);checks=[]
        for h in [1e-6,1e-5,1e-4]:
            ep,_=evaluate(x0+h*direction);em,_=evaluate(x0-h*direction)
            fd=(ep-em)/(2*h);analytic=float(gradient@direction)
            checks.append(dict(step_A=h,finite_difference=fd,analytic=analytic,relative_error=abs(fd-analytic)/max(abs(analytic),1e-12)))
        write_json(dest/'directional_gradient_check.json',checks)
        if min(c['relative_error'] for c in checks)>.01:raise RuntimeError('Directional gradient check failed at all tested displacements')
        result=minimize(evaluate,x0,method='L-BFGS-B',jac=True,callback=callback,options=amendment['optimizer_options'])
        evaluate(result.x);after=cache['metrics'].copy()
        np.savez_compressed(dest/'relaxed.npz',positions=result.x.reshape(-1,3),cell=d['cell'])
        write(dest/'final.extxyz',system.to_atoms(0))
        record.update(status='completed',finished_utc=now(),after=after,history=history,
            force_converged=after['max_force_eV_A']<.02,transverse_converged=abs(after['sigma_yy_N_m'])<=.005*EV,
            optimizer=dict(success=bool(result.success),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),
                           nfev_including_validation=counter,fun=float(result.fun)),
            interpretation='Optimizer success is not substituted for force/transverse residual checks; this fixed-cell configuration is not a repaired loading trajectory.')
    except BaseException as error:
        record.update(status='execution_error',finished_utc=now(),error=repr(error),traceback=traceback.format_exc(),history=history);raise
    finally:write_json(dest/'record.json',record)

if __name__=='__main__':main()
