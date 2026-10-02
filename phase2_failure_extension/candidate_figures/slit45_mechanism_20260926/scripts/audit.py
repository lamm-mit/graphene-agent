"""Read-only 45-degree slit mechanism comparison; no simulations or paper writes."""
from pathlib import Path
import json,csv,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];EV=16.0217663
D=json.loads((STUDY/'candidate_figures/one_to_one/data/designs.json').read_text())
design=next(x for x in D if x['name']=='S2_slit_45deg')
sources={};rows=[];curves=[];all_events={};sets={};traj={}
paths={'Phase 1 single shot':STUDY/'inputs/baseline/trajectories'/f"{design['parent_run_id']}.npz",
 'Phase 2 primary extension':STUDY/'runs/ext__S2_slit_45deg/trajectory.npz',
 'Phase 2 resolution check':STUDY/'runs/res__S2_slit_45deg/trajectory.npz'}
def digest(p):sources[str(p.relative_to(STUDY))]=hashlib.sha256(p.read_bytes()).hexdigest()
for label,path in paths.items():
 z=np.load(path);digest(path);traj[label]=z;P=z['positions'].astype(float);L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2];eps=z['eps_x'];states=[]
 for k in range(len(eps)):
  if label=='Phase 1 single shot':B=z[f'bonds_{k}']
  else:
   fp=path.parent/'frames'/f'{k:05d}.npz'
   with np.load(fp) as f:B=f['bonds']
   digest(fp)
  states.append({tuple(sorted(map(int,b))) for b in B})
 sets[label]=states
 def dist(pair,k):
  v=P[k,pair[1],:2]-P[k,pair[0],:2];v-=np.round(v/L[k])*L[k]
  return float(np.linalg.norm(v))
 seen=set(states[0]);events=[];reappear=0;broken_total=0
 for k,S in enumerate(states):
  fresh=S-seen;formation=(S-states[k-1]) if k else set();broken=(states[k-1]-S) if k else set();broken_total+=len(broken)
  reappear+=len(formation-fresh)
  for pair in sorted(fresh):
   last=k
   while last+1<len(states) and pair in states[last+1]:last+=1
   events.append(dict(frame=k,strain=float(eps[k]),pair=list(pair),last_continuous_frame=last,persistent=last>=k+2,initial_distance_A=dist(pair,0),first_distance_A=dist(pair,k)))
  seen|=S
  curves.append(dict(branch=label,frame=k,strain=float(eps[k]),stress_N_m=float(z['sigma_xx'][k]*EV),transverse_strain=float(z['eps_y'][k]),retained_new_pairs=len(S-states[0]),distinct_new_pairs=len(seen-states[0]),formation_events=len(formation),broken_events=len(broken),cumulative_broken_events=broken_total,noninitial_edges_first_seen=len(fresh)))
 persistent=[e for e in events if e['persistent']]
 for pt in curves[-len(states):]:pt['persistent_distinct_new_pairs']=sum(e['frame']<=pt['frame'] for e in persistent)
 all_events[label]=events
 row=dict(branch=label,frames=len(eps),endpoint_strain=float(eps[-1]),peak_strain=float(eps[np.argmax(z['sigma_xx'])]),peak_stress_N_m=float(max(z['sigma_xx'])*EV),endpoint_transverse_strain=float(z['eps_y'][-1]),first_new_strain=events[0]['strain'],first_persistent_strain=persistent[0]['strain'],distinct_new_pairs_total=len(events),retained_new_pairs_at_endpoint=len(states[-1]-states[0]),reappearance_events_total=reappear,cumulative_broken_events=broken_total)
 for target in [.15,.17,.20,.26,.32]:
  k=int(np.where(eps<=target+1e-9)[0][-1]);new=states[k]-states[0]
  key=f'{target:.2f}'
  row['at_'+key]=dict(actual_strain=float(eps[k]),transverse_strain=float(z['eps_y'][k]),retained_new_pairs=len(new),persistent_distinct_new_pairs=sum(e['frame']<=k for e in persistent),stress_N_m=float(z['sigma_xx'][k]*EV))
  if new:row['at_'+key].update(initial_distance_quantiles_A=np.quantile([dist(b,0) for b in new],[0,.5,1]).tolist(),current_distance_quantiles_A=np.quantile([dist(b,k) for b in new],[0,.5,1]).tolist())
 if 'max_force_eV_A' in z:
  row['force_flagged_frames']=int(sum(z['max_force_eV_A']>.02));row['transverse_flagged_frames']=int(sum(abs(z['sigma_yy'])>.005))
 rows.append(row)
 print(json.dumps(row,indent=2))

# Confirm a primary and resolution-branch neighbor exchange: new partner follows
# a removed connection and persists >=3 saved frames. Prefer interior events.
exchanges=[]
for label in ['Phase 2 primary extension','Phase 2 resolution check']:
 S=sets[label];z=traj[label];P=z['positions'].astype(float);L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2]
 events=all_events[label]
 for e in events:
  k=e['frame']
  if k<2 or not e['persistent']:continue
  i,j=e['pair']
  for p,new in [(i,j),(j,i)]:
   old_pairs=[b for b in S[k-1]-S[k] if p in b]
   for b in old_pairs:
    old=b[0] if b[1]==p else b[1]
    if b in S[0]:continue
    def dd(a,b,frame):
     v=P[frame,b,:2]-P[frame,a,:2];v-=np.round(v/L[frame])*L[frame]
     return float(np.linalg.norm(v))
    exchanges.append(dict(branch=label,frame=k,strain=float(z['eps_x'][k]),previous_strain=float(z['eps_x'][k-1]),atom=p,old_partner=old,new_partner=new,position=P[k,p,:2].tolist(),old_distance_before_A=dd(p,old,k-1),old_distance_after_A=dd(p,old,k),new_distance_before_A=dd(p,new,k-1),new_distance_after_A=dd(p,new,k),initial_new_distance_A=dd(p,new,0),new_connection_persists_through_strain=float(z['eps_x'][e['last_continuous_frame']]),interior=bool(np.all(P[k,p,:2]>15)&np.all(P[k,p,:2]<L[k]-15))))

(ROOT/'data/summary.json').write_text(json.dumps({'design':design,'branches':rows,'neighbor_exchanges':exchanges},indent=2)+'\n')
(ROOT/'data/new_pair_events.json').write_text(json.dumps(all_events,indent=2)+'\n')
(ROOT/'data/source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
with (ROOT/'data/curves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
print('NEW-TO-NEW NEIGHBOR EXCHANGES',len(exchanges));print(json.dumps([e for e in exchanges if e['interior']][:8],indent=2))
