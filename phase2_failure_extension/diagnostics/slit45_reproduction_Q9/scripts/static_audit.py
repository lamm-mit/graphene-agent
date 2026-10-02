"""Q9 static checks: input serialization, batch/precision repeatability and neighbor rebuilds."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
from ase import Atoms
from ase.io import read
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1]
sys.path.insert(0,str(STUDY/'source_original'))
from atomistics.structures import design_space as D
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
EV=16.0217663
spec=json.loads((ROOT/'inputs/stage2_b06.json').read_text())
atoms=[D.generate(s['family'],**dict(s['params'],seed=s['seed'])) for s in spec['structures']]
target=2;parent='run_20260905_091359_4df2c2'
saved=read(STUDY/'inputs/baseline/runs'/parent/'initial.extxyz')
assert np.array_equal(atoms[target].positions.astype(np.float32),saved.positions.astype(np.float32))
assert np.array_equal(atoms[target].cell.array.astype(np.float32),saved.cell.array.astype(np.float32))
Z=np.load(STUDY/'inputs/baseline/trajectories'/f'{parent}.npz')
result={'input_serialization_float32_exact':True,'static_cases':[],'affine_neighbor_checks':[]}

def eval_state(engine,group,tidx,repeats):
 system=BatchedSystem(group,engine);nl=system.build_neighbor_list(skin=.3)
 arr=[]
 for rep in range(repeats):
  out=engine.evaluate(system.positions,system.cell,system.sid,nl,compute_stress=True)
  F=out['forces'][system.slice(tidx)].detach().cpu().numpy().copy()
  stress=(out['dE_deps'][tidx,:2,:2]/system.areas()[tidx]).detach().cpu().numpy()*EV
  arr.append((F,stress,float(out['energy'][tidx].detach().cpu())))
 return arr,nl
for frame in [0,16,18,21,40,51]:
 at=Atoms('C'+str(len(saved)),positions=Z['positions'][frame],cell=Z['cells'][frame],pbc=[1,1,0]);group=list(atoms);group[target]=at
 entry={'frame':frame,'strain':float(Z['eps_x'][frame]),'evaluations':{}}
 raw={}
 for device,precision,repeats in [('mps','float32',5),('cpu','float64',1)]:
  engine=TorchRebo2Scr(device=device,dtype=getattr(torch,precision))
  for layout,g,tidx in [('single',[at],0),('batch4',group,target)]:
   outputs,nl=eval_state(engine,g,tidx,repeats);raw[(device,layout)]=outputs
   entry['evaluations'][device+'_'+layout]={'stress_N_m':outputs[0][1].tolist(),'max_force_eV_A':float(np.linalg.norm(outputs[0][0],axis=1).max()),'energy_eV':outputs[0][2],'neighbor_M':nl.M,'neighbor_Mb':nl.Mb,'repeat_max_force_component_delta':max(float(np.max(abs(x[0]-outputs[0][0]))) for x in outputs),'repeat_max_stress_delta_N_m':max(float(np.max(abs(x[1]-outputs[0][1]))) for x in outputs)}
  entry[device+'_batch_vs_single_max_force_component_delta']=float(np.max(abs(raw[(device,'single')][0][0]-raw[(device,'batch4')][0][0])))
  entry[device+'_batch_vs_single_max_stress_delta_N_m']=float(np.max(abs(raw[(device,'single')][0][1]-raw[(device,'batch4')][0][1])))
 entry['mps_vs_cpu_single_max_force_component_delta']=float(np.max(abs(raw[('mps','single')][0][0]-raw[('cpu','single')][0][0])))
 result['static_cases'].append(entry);(ROOT/'results/static_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('static frame',frame,entry['mps_batch_vs_single_max_force_component_delta'],entry['mps_batch_vs_single_max_stress_delta_N_m'],flush=True)
# Pure affine diagnostic: current needs_rebuild removes affine motion completely.
engine=TorchRebo2Scr(device='cpu',dtype=torch.float64)
for frame in [0,16,18]:
 at=Atoms('C'+str(len(saved)),positions=Z['positions'][frame],cell=Z['cells'][frame],pbc=[1,1,0]);system=BatchedSystem([at],engine);old=system.build_neighbor_list(skin=.3)
 for ey in [-.005,-.01,-.025,-.05,-.1]:
  factors=torch.tensor([1.01,1+ey,1.],dtype=torch.float64)
  R=system.positions*factors;C=system.cell.clone();C[:,0,:]*=1.01;C[:,1,:]*=1+ey
  needs=engine.needs_rebuild(R,C,system.sid,old)
  new=engine.build_neighbor_list([R.numpy()],C.numpy(),[1,1,0],skin=.3)
  a=engine.evaluate(R,C,system.sid,old,compute_stress=True);b=engine.evaluate(R,C,system.sid,new,compute_stress=True)
  fd=float(torch.max(abs(a['forces']-b['forces'])));sd=float(torch.max(abs(a['dE_deps']-b['dE_deps']))/(C[0,0,0]*C[0,1,1])*EV)
  row={'source_frame':frame,'affine_x_increment':.01,'affine_y_increment':ey,'driver_requests_rebuild':needs,'force_component_difference_eV_A':fd,'stress_difference_N_m':sd,'energy_difference_eV':float(a['energy_total']-b['energy_total'])}
  result['affine_neighbor_checks'].append(row);print('affine',row,flush=True)
( ROOT/'results/static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
