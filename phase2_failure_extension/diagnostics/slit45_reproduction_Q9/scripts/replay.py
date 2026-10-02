"""Q9 replay of the recorded Phase-I protocol with observation-only instrumentation."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,inspect,traceback,dataclasses
import numpy as np
import torch
from ase.io import read
import ase
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code_snapshot'))
from atomistics.structures import design_space as D
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner,AQSConfig
EV=16.0217663

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 return v

def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()

def main(job_id):
 jobs=json.loads((ROOT/'jobs.json').read_text());job=next(j for j in jobs if j['id']==job_id)
 for rel,h in json.loads((ROOT/'source_checksums.json').read_text()).items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
 dest=ROOT/'runs'/job_id;dest.mkdir(exist_ok=False);(dest/'frames').mkdir();(dest/'checkpoints').mkdir()
 rec={'id':job_id,'pid':os.getpid(),'status':'running','started_utc':now(),'job':job,'environment':{'python':sys.version,'torch':torch.__version__,'ase':ase.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'device':'mps','precision':'float32'},'instrumentation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 write(dest/'record.json',rec)
 spec=json.loads((ROOT/'inputs/stage2_b06.json').read_text());ids=json.loads((ROOT/'protocol.json').read_text())['original_batch_parent_ids'];target=job['target_index']
 selected=list(zip(spec['structures'],ids)) if job['layout']=='original_batch4' else [(spec['structures'][2],ids[2])]
 atoms=[];names=[];parents=[]
 for s,rid in selected:
  a=D.generate(s['family'],**dict(s['params'],seed=s['seed']));b=read(ROOT/'inputs'/rid/'initial.extxyz');parent=json.loads((ROOT/'inputs'/rid/'record.json').read_text())
  assert np.array_equal(a.positions.astype(np.float32),b.positions.astype(np.float32))
  assert np.array_equal(a.cell.array.astype(np.float32),b.cell.array.astype(np.float32))
  atoms.append(a);names.append(s['name']);parents.append(parent)
 cfg=parents[target]['aqs_config'];assert all(p['aqs_config']==cfg for p in parents)
 rec.update(config=cfg,names=names,atom_counts=[len(a) for a in atoms],parent_ids=[p['run_id'] for p in parents]);write(dest/'record.json',rec)
 trace=open(dest/'minimizer_calls.jsonl','x',buffering=1);neighbors=open(dest/'neighbor_rebuilds.jsonl','x',buffering=1);runner_ref=[]
 class EngineTrace(TorchRebo2Scr):
  def build_neighbor_list(self,*args,**kwargs):
   nl=super().build_neighbor_list(*args,**kwargs)
   r=runner_ref[0] if runner_ref else None
   neighbors.write(json.dumps({'time':now(),'target_eps':float(r.eps_x[target]) if r else None,'M':nl.M,'Mb':nl.Mb,'n_pairs':nl.n_pairs,'cell':np.asarray(args[1]).tolist()})+'\n')
   return nl
 engine=EngineTrace(device='mps',dtype=torch.float32);assert engine.info.parameter_checksum==parents[target]['engine']['parameter_checksum']
 system=BatchedSystem(atoms,engine,names=names)
 # Match the original pre-run initial-energy evaluation before constructing the driver.
 nl0=system.build_neighbor_list(skin=cfg['skin']);engine.evaluate(system.positions,system.cell,system.sid,nl0,compute_stress=False)
 started=time.time();frame_rows=[];checkpointed=set()
 def log(message):
  print(message,flush=True)
  if message.startswith('step ') and runner_ref:
   r=runner_ref[0]
   for limit in [.16,.17,.18,.20,.32]:
    if limit in checkpointed or r.eps_x[target]<limit-1e-10:continue
    # Capture the full runtime state after the inherited accepted-step bookkeeping.
    state={k:v for k,v in r.__dict__.items() if k not in ['engine','sys','log','on_done']}
    payload={'runner':state,'system':{'positions':system.positions.detach().clone(),'cell':system.cell.clone(),'cell0':system.cell0.clone()},'names':names,'target':target,'note':'Observation checkpoint after inherited bookkeeping; not spliced into any trajectory.'}
    torch.save(payload,dest/'checkpoints'/f'after_{limit:.2f}.pt');checkpointed.add(limit)
 class TraceRunner(AQSRunner):
  def _minimize(self,active,max_steps=None,stall_iters=500):
   site=inspect.currentframe().f_back.f_lineno
   before=self.sys.positions_numpy(target).copy()
   res=super()._minimize(active,max_steps=max_steps,stall_iters=stall_iters)
   after=self.sys.positions_numpy(target)
   trace.write(json.dumps(clean({'time':now(),'callsite_line':site,'target_eps':float(self.eps_x[target]),'target_eps_y':float(self.eps_y[target]),'active':np.asarray(active),'budget':max_steps or self.cfg.max_fire_steps,'iterations':res.iterations,'converged':res.converged,'soft':res.soft,'fmax':res.fmax,'target_max_component_motion_A':float(np.max(abs(after-before)))}))+'\n')
   return res
  def _record(self,out,bonds,n_new_broken,n_new_formed,fire_it,fire_conv,n_t,active,t_start):
   super()._record(out,bonds,n_new_broken,n_new_formed,fire_it,fire_conv,n_t,active,t_start)
   if not active[target]:return
   fr=self.frames[target][-1];k=len(self.frames[target])-1;sl=self.sys.slice(target)
   F=out['forces'][sl].detach().cpu().numpy();row=dataclasses.asdict(fr);row['max_force_eV_A']=float(np.linalg.norm(F,axis=1).max());row['frame']=k;row['wall_time_s']=time.time()-started;row=clean(row);frame_rows.append(row)
   np.savez_compressed(dest/'frames'/f'{k:05d}.npz',positions=self.positions_frames[target][-1],cell=self.cell_frames[target][-1],bonds=self.bond_frames[target][-1],force=F,peratom_energy=self.peratom_energy_frames[target][-1])
   write(dest/'progress.json',{'time':now(),'pid':os.getpid(),'frame':k,'strain':fr.eps_x,'stress_N_m':float(fr.sigma[0])*EV,'force_residual_eV_A':row['max_force_eV_A'],'elapsed_s':time.time()-started,'batch_eps':self.eps_x.tolist()})
   write(dest/'curve.json',frame_rows)
 try:
  runner=TraceRunner(engine,system,AQSConfig(**cfg),log=log);runner_ref.append(runner)
  values=runner.run();r=values[target];ep=np.array(r['eps_x']);sigma=np.array(r['sigma_xx'])*EV
  np.savez_compressed(dest/'trajectory.npz',positions=np.stack(r['positions']),cells=np.stack(r['cells']),eps_x=ep,eps_y=r['eps_y'],sigma_xx=r['sigma_xx'],sigma_yy=r['sigma_yy'],energy=r['energy'],n_broken_cum=r['n_broken_cum'],max_force_eV_A=[x['max_force_eV_A'] for x in frame_rows])
  rec.update(status='completed',finished_utc=now(),wall_time_s=time.time()-started,termination=r['termination'],frames=len(ep),maximum_recorded_stress_N_m=float(sigma.max()),peak_strain=float(ep[sigma.argmax()]),end_strain=float(ep[-1]),first_damage_strain=float(r['first_damage_strain']) if np.isfinite(r['first_damage_strain']) else None,stats=r['stats'],frames_above_force_target=sum(x['max_force_eV_A']>cfg['fmax'] for x in frame_rows),frames_above_transverse_target=sum(abs(x['sigma'][1])>cfg['sigma_t_tol'] for x in frame_rows));write(dest/'record.json',rec)
 except BaseException as ex:
  rec.update(status='execution_error',finished_utc=now(),error=repr(ex),traceback=traceback.format_exc());write(dest/'record.json',rec);raise
 finally:
  trace.close();neighbors.close()
if __name__=='__main__':main(sys.argv[1])
