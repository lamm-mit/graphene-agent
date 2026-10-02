"""Read-only angle audit of geometric bond reconnection in the same 12 slit designs."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1]
D=json.loads((STUDY/'candidate_figures/one_to_one/data/designs.json').read_text())
names=['S2_slit_0deg','P1_slit_05deg','P1_slit_10deg','P1_slit_15deg','S7_slit_20deg','P1_slit_25deg','P1_slit_30deg','P1_slit_35deg','S2_slit_45deg','P1_slit_60deg','P1_slit_75deg','S2_slit_90deg']
rows=[];curves=[];sources={}
for name in names:
 d=next(v for v in D if v['name']==name)
 if d['C']:
  run=STUDY/'runs'/d['extension_case'];p=run/'trajectory.npz'
 else:
  run=None;p=STUDY/'inputs/baseline/trajectories'/f"{d['parent_run_id']}.npz"
  if not p.exists():p=STUDY/'candidate_figures/one_to_one/inputs/trajectories'/p.name
 z=np.load(p);P=z['positions'].astype(float);C=z['cells'];eps=z['eps_x'];N=P.shape[1]
 states=[]
 for k in range(len(eps)):
  if run:
   with np.load(run/'frames'/f'{k:05d}.npz') as f:states.append({tuple(sorted(map(int,b))) for b in f['bonds']})
  else:
   L=np.diag(C[k])[:2];pos=P[k,:,:2]%L
   states.append({tuple(map(int,b)) for b in cKDTree(pos,boxsize=L).query_pairs(2,output_type='ndarray')})
 initial=states[0];seen=set(initial);unique=set();reappeared=0;first=[];formed=[]
 for k,S in enumerate(states):
  F=S-states[k-1] if k else set()
  unique|=F-initial
  new=F-seen;seen|=S
  reappeared+=len(F-new)
  for pair in sorted(new):
   last=k
   while last+1<len(states) and pair in states[last+1]:last+=1
   L=np.diag(C[k])[:2];dr=P[k,pair[1],:2]-P[k,pair[0],:2];dr-=np.round(dr/L)*L
   persistent=(last>=k+2)
   first.append((k,pair,last,persistent,float(np.linalg.norm(dr))))
  curves.append(dict(name=name,angle=d['params']['angle_deg'],strain=float(eps[k]),n_atoms=N,unique_new_pairs=len(unique),retained_noninitial_pairs=len(S-initial),cumulative_formation_events=sum(formed)+len(F),residual_flag=d['residual_flag']))
  formed.append(len(F))
 persistence=[v for v in first if v[3]]
 for k,point in enumerate(curves[-len(states):]):
  point['persistent_unique_new_pairs']=sum(v[0]<=k for v in persistence)
 row=dict(name=name,angle=d['params']['angle_deg'],source_phase=d['source_phase'],trajectory='primary extension' if d['C'] else 'original retained control',porosity=d['porosity'],n_atoms=N,ending_strain=float(eps[-1]),initial_bonds=len(initial),new_unique_pairs_total=len(unique),reappearance_events_total=reappeared,first_new_strain=float(eps[first[0][0]]) if first else None,first_persistent_new_strain=float(eps[persistence[0][0]]) if persistence else None,residual_flag=d['residual_flag'])
 for target in [.10,.15,.20,.195,.275,.30]:
  key=f'{target:.3f}';ix=np.where(eps<=target+1e-10)[0][-1]
  supported=eps[-1]>=target-1e-10
  rr=[v for v in first if v[0]<=ix]
  row['frame_strain_at_'+key]=float(eps[ix]) if supported else None
  row['unique_new_at_'+key]=len(rr) if supported else None
  row['retained_new_at_'+key]=len(states[ix]-initial) if supported else None
  row['persistent_new_at_'+key]=sum(v[3] for v in rr) if supported else None
  row['persistent_new_per1000_at_'+key]=1000*sum(v[3] for v in rr)/N if supported else None
  row['unique_new_per1000_at_'+key]=1000*len(rr)/N if supported else None
 rows.append(row);sources[str(p.relative_to(STUDY))]=hashlib.sha256(p.read_bytes()).hexdigest()
 print(name,'angle',row['angle'],'first persistent',row['first_persistent_new_strain'],'new/persistent at .2',row['unique_new_at_0.200'],row['persistent_new_at_0.200'],'end total',row['new_unique_pairs_total'],flush=True)
(ROOT/'data/angle_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
for fn,records in [('angle_comparison.csv',rows),('angle_bond_curves.csv',curves)]:
 with (ROOT/'data'/fn).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(ROOT/'data/angle_source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
