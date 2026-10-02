"""Reproduce candidate tables from immutable study inputs; no manuscript writes."""
from pathlib import Path
import json, csv, hashlib, math, sys
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'candidate_figures'; DATA=OUT/'data'
sys.path.insert(0,str(ROOT/'scripts'))
from review_loading_cases import read_curve, scalar_check, first_stop
EV=16.0217663
FAMCOL={'pristine':'k','vacancies':'0.5','precrack':'#8c564b','ring_around_hole':'#e377c2','nanomesh_single':'#1f77b4','nanomesh_hier':'#d62728','voronoi_network':'#2ca02c','slit_array':'#ff7f0e','strut_lattice':'#9467bd','graded_pores':'#17becf'}
FAMNAME={'pristine':'pristine','vacancies':'vacancies','precrack':'precracked','ring_around_hole':'hole + rings','slit_array':'slit array','strut_lattice':'strut lattice','graded_pores':'graded pores','nanomesh_single':'single-level mesh','nanomesh_hier':'hierarchical mesh','voronoi_network':'Voronoi network'}
FAMORDER=['pristine','vacancies','precrack','ring_around_hole','slit_array','strut_lattice','graded_pores','nanomesh_single','nanomesh_hier','voronoi_network']
THUMBS=['S1_pristine_zz','S1_precrack_L20','S1_vac_random_2pct','S5_slit0_phi0.1','S2_slit_90deg','S7_slit_20deg','S7_H2_veinsX_W12_ellipseX','S2_hole20_rings2','S2_voronoi_n25_reg0.6','S2_strut_090','S2_H1_p16','S2_graded_x']
REF=['S2_H1_p16','S1_mesh_p16_phi0.2','S6_mesh_p16_offset1','S6_mesh_p16_offset2','S6_mesh_p16_offset3']
GROUPS=[('Positional disorder (Å)',{'0':REF,'1':['S2_mesh_jitter1.0'],'2':['S2_mesh_jitter2.0','S6_mesh_jitter2.0_s2','S6_mesh_jitter2.0_s3'],'3':['S2_mesh_jitter3.0']}),('Pore-size dispersion',{'0':REF,'0.15':['S2_mesh_sizedis0.15','S6_mesh_sizedis0.15_s2','S6_mesh_sizedis0.15_s3'],'0.3':['S2_mesh_sizedis0.3']}),('Voronoi regularity',{'0':['S2_voronoi_n25_reg0.0','S6_voronoi_n25_reg0.0_s2','S6_voronoi_n25_reg0.0_s3','S2_voronoi_n50_reg0.0'],'0.3':['S5_voronoi_n25_reg0.3'],'0.6':['S2_voronoi_n25_reg0.6','S6_voronoi_n25_reg0.6_s2','S6_voronoi_n25_reg0.6_s3'],'0.8':['S5_voronoi_n25_reg0.8','S7_voronoi_n35_reg0.8_s7'],'1':['S2_voronoi_n25_reg1.0']}),('Pore-size gradient',{'uniform':REF,'along load':['S2_graded_x'],'radial':['S2_graded_radial'],'weak band':['S7_graded_x_weak']}),('Flaw halo',{'hole 20':['S1_hole_d20'],'+2 rings':['S2_hole20_rings2'],'+far pores':['S2_hole20_background'],'hole 30':['S4_hole_d30'],'+3 rings':['S7_hole30_rings3']})]
def load(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):
 (DATA/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 if not rows:return
 with (DATA/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ligament(r):
 f,p,d=r['family'],r['params'],r['descriptors']
 if f=='slit_array':
  a=p.get('angle_deg',0)%180;return 'straight' if a<30 or a>150 else ('bridges' if abs(a-90)<30 else 'oblique')
 if f=='nanomesh_single':
  if p.get('aspect',1)>1.2:return 'straight' if abs(p.get('angle_deg',0)-90)<30 else 'fine_round'
  return 'coarse_round' if d['ligament_width_mean_A']>=18 else 'fine_round'
 if f=='nanomesh_hier':return 'fine_round'
 if f=='voronoi_network':return 'random'
 if f=='strut_lattice':return 'straight' if any(abs(a%180)<1 for a in p.get('angles_deg',[])) else 'random'
 return 'flawed_sheet' if f in ['precrack','ring_around_hole','vacancies','graded_pores'] else 'fine_round'
def ep(e):return dict(strength=e['maximum_recorded_stress_N_m'],integral=e['stress_strain_integral_J_m2'],end_strain=e['end_strain'],peak_strain=e['strain_at_recorded_maximum'],failure_observed=e['failure_observed'],failure_strain=e['failure_strain_if_observed'],stop=e['termination_reason'])
def value(r,state,key):
 if key in ['rho','alignment','min_section','modulus']:return r[key]
 return (r[state] if r[state] is not None else r['A'])[key]
def correlation(xs,ys):return dict(pearson=float(np.corrcoef(xs,ys)[0,1]),spearman=float(spearmanr(xs,ys).statistic))
def main():
 inv=load('candidate_figures/data/release_inventory.json')
 assert all(sha(ROOT/p)==h for p,h in inv['source_sha256'].items())
 base=[json.loads(p.read_text()) for p in sorted((ROOT/'inputs/baseline/runs').glob('*/record.json'))]
 base=[r for r in base if r.get('status')=='completed' and r.get('metrics')]
 runs={p.parent.name:json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/record.json'))}
 extensions={r['parent_run_id']:r for r in runs.values() if r['role']=='extension'}
 assert len(extensions)==34
 rows=[];curves=[];moduli=[]
 for r in base:
  m,d=r['metrics'],r['descriptors'];name=r['name'];run=extensions.get(r['run_id'])
  a=dict(strength=m['strength_Nm'],integral=m['work_to_failure_J_m2'],end_strain=m['failure_strain'],peak_strain=m['strain_at_peak'],failure_observed=m['failure_observed'],failure_strain=m['failure_strain'] if m['failure_observed'] else None,stop=r.get('termination',{}))
  if not isinstance(a['stop'],str):a['stop']=str(a['stop'])
  row=dict(parent_run_id=r['run_id'],name=name,family=r['family'],source_phase={'discovery':'I','paper':'earlier II','validation':'original validation','interactive':'original interactive'}.get(r['campaign'],r['campaign']),campaign=r['campaign'],params=r['params'],porosity=r.get('porosity') or 0,n_atoms=r['n_atoms'],rho=r['n_atoms']/(d['Lx']*d['Ly'])/(4/(np.sqrt(3)*2.460177**2)),alignment=d['anisotropy_index']*math.cos(2*math.radians(d['ligament_orientation_deg'])),min_section=d['min_solid_fraction_across_x'],modulus=m['modulus_2d_Nm'],level=min(r.get('hierarchy_levels',2),3) if r['family']=='nanomesh_hier' else 1,ligament_class=ligament(r),tags=r.get('tags') or [],A=a,B=None,C=None,extension_case=None,residual_flag=None,residual_fraction=None)
  if run:
   row.update(A=ep(run['paired']['A_archived']),B=ep(run['paired']['B_original_protocol_on_rerun']),C=ep(run['paired']['C_extended']),extension_case=run['case_id'])
   new=read_curve(ROOT/'runs'/run['case_id']/'stress_strain.csv');bad=[q['max_force_eV_A']>run['aqs_config']['fmax'] or abs(q['sigma_yy'])>run['aqs_config']['sigma_t_tol'] for q in new]
   row['residual_flag']=any(bad);row['residual_fraction']=sum(bad)/len(bad)
   sel=[q for q in new if 0<=q['eps_x']<=.02];ym=float(np.polyfit([q['eps_x'] for q in sel],[q['sigma_xx']*EV for q in sel],1)[0]) if len(sel)>=3 else None
   moduli.append(dict(name=name,parent_run_id=r['run_id'],archived_modulus_N_m=m['modulus_2d_Nm'],rerun_small_strain_fit_N_m=ym,relative_change_percent=100*(ym/m['modulus_2d_Nm']-1) if ym is not None else None,fit_definition='OLS stress on strain, 0 <= strain <= 0.02; reproduction diagnostic only'))
  rows.append(row)
 assert len(rows)==132 and sum(r['campaign']=='discovery' for r in rows)==96
 by={r['name']:r for r in rows};disc=[r for r in rows if r['campaign']=='discovery'];meshes=[r for r in rows if r['family'] in ['nanomesh_single','nanomesh_hier'] and .17<=r['porosity']<=.23 and not any(k in r['name'] for k in ['jitter','sizedis']) and 'disorder' not in r['tags']]
 assert len(meshes)==40 and sum(r['C'] is not None for r in meshes)==8 and sum(r['C'] is not None for r in disc)==20
 dump('designs.json',rows);csvout('modulus_reproduction.csv',moduli)
 flat=[]
 for r in rows:
  for state in ['A','B','C']:
   e=r[state] if r[state] is not None else r['A']
   flat.append({k:r[k] for k in ['parent_run_id','name','family','source_phase','rho','alignment','min_section','modulus','level']}|dict(display_state=state,coordinate_source=state if r[state] is not None else 'archived A carried forward; no rerun',extension_case=r['extension_case'],new_run_residual_flag=r['residual_flag'] if state!='A' and r['extension_case'] else None)|e)
 csvout('plotted_design_coordinates.csv',flat)
 # Every saved curve is exported with phase, trajectory branch, residual flags and B/C indices.
 for case,run in runs.items():
  rr=read_curve(ROOT/'runs'/case/'stress_strain.csv');a=read_curve(ROOT/'inputs/baseline/runs'/run['parent_run_id']/'stress_strain.csv')
  for key,crv,endpoint in [('A',a,run['paired']['A_archived']),('C',rr,run['paired']['C_extended'])]:
   scalar_check(crv,endpoint)
   for i,q in enumerate(crv):curves.append(dict(case_id=case,role=run['role'],source_phase=run['source_phase'],trajectory=key,index=i,strain=q['eps_x'],stress_N_m=q['sigma_xx']*EV,cumulative_broken=q['n_broken_cum'],spanning=bool(q['spanning']),residual_flag=(q['max_force_eV_A']>run['aqs_config']['fmax'] or abs(q['sigma_yy'])>run['aqs_config']['sigma_t_tol']) if key=='C' else None,B_index=run['paired']['B_original_protocol_on_rerun']['endpoint_index'] if key=='C' else None,C_index=endpoint['endpoint_index']))
  scalar_check(rr,run['paired']['B_original_protocol_on_rerun'])
  assert first_stop(rr,run['aqs_config'])==(run['paired']['C_extended']['endpoint_index'],run['paired']['C_extended']['termination_reason'])
 csvout('stress_strain_curves.csv',curves)
 # Family envelopes: exact positive-coordinate log-space hull method, explicit degenerate handling.
 hulls=[]
 for panel,xkey,ykey in [('strength_density','rho','strength'),('modulus_density','rho','modulus'),('strength_integral','integral','strength'),('strength_endpoint','end_strain','strength')]:
  for state in ['A','B','C']:
   for fam in FAMORDER:
    rr=[r for r in disc if r['family']==fam and value(r,state,xkey)>0 and value(r,state,ykey)>0]
    if not rr:continue
    xy=np.array([[value(r,state,xkey),value(r,state,ykey)] for r in rr]);p=np.log10(xy);unique=np.unique(p,axis=0,return_index=True)[1];u=p[unique]
    if len(u)>=3 and np.linalg.matrix_rank(u-u[0])==2:ids=unique[ConvexHull(u).vertices];mode='log-space convex hull';area=float(ConvexHull(u).volume)
    else:ids=unique[np.argsort(u[:,np.argmax(np.ptp(u,axis=0))])][[0,-1]] if len(u)>1 else unique;mode='degenerate segment or point';area=0.
    verts=[]
    for i in ids:
     r=rr[int(i)];e=r[state] if r[state] else r['A'];verts.append(dict(parent_run_id=r['parent_run_id'],name=r['name'],x=float(xy[i,0]),y=float(xy[i,1]),log10_x=float(p[i,0]),log10_y=float(p[i,1]),endpoint_censored=not e['failure_observed'],residual_flag=r['residual_flag'] if state!='A' and r['C'] else None,coordinate_source=state if r[state] else 'archived A'))
    hulls.append(dict(panel=panel,state=state,family=fam,population_n=len(rr),mode=mode,log_area=area,vertices=verts))
 dump('boundary_vertices.json',hulls);csvout('boundary_vertices.csv',[dict(panel=h['panel'],state=h['state'],family=h['family'],vertex_order=i,mode=h['mode'])|v for h in hulls for i,v in enumerate(h['vertices'])])
 summary={'populations':dict(discovery=96,earlier_phase_II=18,targeted_phase_I=20,targeted_earlier_II=14,alignment=40,alignment_replacements=8,primary_extensions=34,controls=2,resolution=4,additional_loading_checks=5),'cohorts':{},'alignment':{},'strength_audit':{},'slit_sweep':{},'predictions':{},'family_best':{},'ligament_medians':{}}
 for phase in ['I','II']:
  rr=[r for r in runs.values() if r['role']=='extension' and r['source_phase']==phase];continued=[r for r in rr if r['paired']['C_extended']['endpoint_index']>r['paired']['B_original_protocol_on_rerun']['endpoint_index']]
  c={};c['n']=len(rr);c['continued_n']=len(continued);c['B_equals_C']=len(rr)-len(continued);c['censored']=sum(not r['paired']['C_extended']['failure_observed'] for r in rr);c['residual_flagged']=sum(r['quality']['frames_above_force_tolerance']+r['quality']['frames_above_transverse_tolerance']>0 for r in rr)
  for label,subset in [('all',rr),('continued',continued)]:
   for key in ['maximum_recorded_stress_N_m','stress_strain_integral_J_m2','end_strain']:
    for s0,s1 in [('A','B'),('B','C'),('A','C')]:
     keys={'A':'A_archived','B':'B_original_protocol_on_rerun','C':'C_extended'};delta=[r['paired'][keys[s1]][key]-r['paired'][keys[s0]][key] for r in subset];pct=[100*(r['paired'][keys[s1]][key]/r['paired'][keys[s0]][key]-1) for r in subset]
     c[f'{label}_{s0}_to_{s1}_{key}']=dict(n=len(subset),median_abs=float(np.median(delta)),median_percent=float(np.median(pct)),min_percent=float(min(pct)),max_percent=float(max(pct)))
  summary['cohorts'][phase]=c
 s0=value(by['S1_pristine_zz'],'A','strength');snet=float(np.median([value(r,'A','strength')/r['min_section'] for r in disc if r['family']=='slit_array' and r['params'].get('angle_deg',0)==0 and r['params'].get('orientation','zigzag')=='zigzag']))
 slit=[r for r in rows if r['family']=='slit_array' and r['params'].get('orientation','zigzag')=='zigzag' and r['params'].get('period_x')==30];sweep=sorted([r for r in slit if .17<=r['porosity']<=.23],key=lambda r:r['params']['angle_deg']);ctrl=sorted([r for r in slit if r['params']['angle_deg']==20],key=lambda r:r['porosity'])
 preds=load('inputs/reference/paper_sweeps_predictions.json');pr={r['name']:r for r in preds['predictions']}
 slitdata=[]
 for r in slit:
  raw=next(b for b in base if b['run_id']==r['parent_run_id']);des=raw.get('design') or raw['descriptors'];p=r['params'];theta=p['angle_deg'];L=p['slit_len'];px=des['Lx']/max(1,round(des['Lx']/p['period_x']));py=des['Ly']/max(1,round(des['Ly']/p['period_y']));ov=L*math.cos(math.radians(theta))-px/2;clear=py-L*math.sin(math.radians(theta));rule=pr[r['name']]['predicted']['strength_Nm_rule'] if r['name'] in pr else r['min_section']*snet;mech=pr[r['name']]['predicted']['strength_Nm_mechanism'] if r['name'] in pr else rule*(max(.38,1-.62*min(1,theta/20)) if ov>0 and theta>=5 else .9)
  slitdata.append(dict(name=r['name'],parent_run_id=r['parent_run_id'],source_phase=r['source_phase'],theta=theta,porosity=r['porosity'],overlap_A=ov,clearance_A=clear,rule=rule,mechanism=mech,prediction_source='frozen preregistration' if r['name'] in pr else 'original plotting formula; not separately preregistered',in_angle_sweep=r in sweep,A=value(r,'A','strength'),B=value(r,'B','strength'),C=value(r,'C','strength'),B_C_are_archived_carry_forward=r['C'] is None))
 csvout('slit_angle_predictions_and_responses.csv',slitdata)
 nested=sorted([r for r in meshes if r['level']>1],key=lambda r:r['alignment']);premrows=[]
 for state in ['A','B','C']:
  fits={};res={}
  for label,rs in [('all',meshes),('H1',[r for r in meshes if r['level']==1]),('H2',[r for r in meshes if r['level']==2])]:
   x=[r['alignment'] for r in rs];y=[value(r,state,'strength') for r in rs];coef=np.polyfit(x,y,1);fits[label]=coef.tolist();res[label]=dict(n=len(rs),slope=float(coef[0]),intercept=float(coef[1]),**correlation(x,y))
  pre=[value(r,state,'strength')-np.polyval(fits['H1'],r['alignment']) for r in nested]
  for r,v in zip(nested,pre):premrows.append(dict(name=r['name'],parent_run_id=r['parent_run_id'],state=state,alignment=r['alignment'],level=r['level'],premium_N_m=float(v)))
  res['premium_mean']=float(np.mean(pre));res['premium_sd']=float(np.std(pre));res['integral_correlation']=correlation([r['alignment'] for r in meshes],[value(r,state,'integral') for r in meshes]);summary['alignment'][state]=res
  porous=[r for r in disc if r['porosity']>.03];band=[r for r in disc if .77<=r['rho']<=.83];bs=[value(r,state,'strength')/r['rho'] for r in band]
  med={c:float(np.median([value(r,state,'strength')/r['min_section'] for r in porous if r['ligament_class']==c])) for c in set(r['ligament_class'] for r in porous)}
  summary['strength_audit'][state]=dict(porous_n=len(porous),density=correlation([r['rho'] for r in porous],[value(r,state,'strength') for r in porous]),min_section=correlation([r['min_section'] for r in porous],[value(r,state,'strength') for r in porous]),band_n=len(band),band_specific_min=min(bs),band_specific_max=max(bs),band_specific_ratio=max(bs)/min(bs),retrospective_ligament_class_medians=med)
  summary['family_best'][state]={f:dict(name=(best:=max([r for r in disc if r['family']==f],key=lambda r:value(r,state,'strength')/r['rho']))['name'],efficiency=value(best,state,'strength')/best['rho']/s0) for f in FAMORDER}
  summary['slit_sweep'][state]={'minimum':min([dict(name=r['name'],theta=r['params']['angle_deg'],strength=value(r,state,'strength')) for r in sweep],key=lambda q:q['strength']),'minimum_0_to_45_degrees':min([dict(name=r['name'],theta=r['params']['angle_deg'],strength=value(r,state,'strength')) for r in sweep if r['params']['angle_deg']<=45],key=lambda q:q['strength']),'ranked_names':[r['name'] for r in sorted(sweep,key=lambda r:value(r,state,'strength'))],'n_records':len(sweep),'n_angles':len(set(r['params']['angle_deg'] for r in sweep))}
  ligs=[r for r in rows if r['campaign'] in ['discovery','paper'] and r['porosity']>.03 and r['ligament_class'] in ['straight','coarse_round','fine_round','random'] and not (r['family']=='slit_array' and r['params'].get('angle_deg',0)!=0)]
  summary['ligament_medians'][state]={cl:dict(n=len(rr:=[r for r in ligs if r['ligament_class']==cl]),median=float(np.median([value(r,state,'strength')/r['min_section'] for r in rr]))) for cl in ['straight','coarse_round','fine_round','random']}
 csvout('hierarchy_premiums.csv',premrows);dump('alignment_population.json',[r['parent_run_id'] for r in meshes]);assert sum(x['n'] for x in summary['ligament_medians']['A'].values())==77
 groupdata=[]
 for title,levels in GROUPS:
  for lev,names in levels.items():
   for state in ['A','B','C']:
    for key in ['strength','integral']:
     vals=[value(by[n],state,key) for n in names];groupdata.append(dict(group=title,level=lev,state=state,metric=key,n=len(names),mean=float(np.mean(vals)),sd=float(np.std(vals)),names=';'.join(names),rerun_n=sum(by[n]['C'] is not None for n in names)))
 csvout('disorder_groups.csv',groupdata)
 prediction_rows=[]
 for cohort,n in [('Phase-I holdouts','holdout_predictions'),('Earlier Phase-II sweeps','paper_sweeps_predictions')]:
  obj=load(f'inputs/reference/{n}.json');out=[]
  for p in obj['predictions']:
   r=by[p['name']]
   for method,k in ([('registered','strength_Nm')] if n=='holdout_predictions' else [('net-section','strength_Nm_rule'),('mechanism','strength_Nm_mechanism')]):
    pred=p['predicted'][k]
    for state in ['A','B','C']:
     actual=value(r,state,'strength');q=dict(cohort=cohort,method=method,name=r['name'],parent_run_id=r['parent_run_id'],state=state,prediction=pred,observed=actual,relative_error_percent=100*(pred/actual-1),retrospective=state!='A',extended_case=r['extension_case'],original_file_sha256=sha(ROOT/f'inputs/reference/{n}.json'));out.append(q);prediction_rows.append(q)
  summary['predictions'][cohort]={f'{method}_{state}':dict(n=len(rr),MAPE_percent=float(np.mean([abs(q['relative_error_percent']) for q in rr])),within_15_percent=sum(abs(q['relative_error_percent'])<=15 for q in rr)) for method in sorted(set(q['method'] for q in out)) for state in ['A','B','C'] if (rr:=[q for q in out if q['method']==method and q['state']==state])}
 csvout('retrospective_prediction_comparison.csv',prediction_rows)
 # Primary effects in one machine-readable A/B/C table.
 effects=[]
 for r in rows:
  if not r['C']:continue
  item={k:r[k] for k in ['parent_run_id','name','family','source_phase','extension_case','residual_flag','residual_fraction']}
  for state in ['A','B','C']:
   item.update({state+'_'+k:v for k,v in r[state].items()})
  for s,t in [('A','B'),('B','C'),('A','C')]:
   for k in ['strength','integral','end_strain']:item[f'{s}_{t}_{k}_percent']=100*(r[t][k]/r[s][k]-1)
  effects.append(item)
 csvout('primary_A_B_C_effects.csv',effects);dump('quantitative_summary.json',summary)
 # Verify archived coordinate reconstruction agrees with every targeted A.
 assert all(math.isclose(r['metrics']['strength_Nm'],by[r['name']]['A']['strength'],rel_tol=1e-10) for r in base)
 dump('data_qa.json',dict(all_source_hashes_pass=True,all_45_loading_metrics_and_final_stops_recomputed=True,one_to_one_primary_parents=34,discovery_population=96,discovery_replacements=20,alignment_population=40,alignment_replacements=8,predictions_unchanged=True,baseline_descriptors_and_moduli_preserved=True,physical_convergence_claimed=False))
 print(json.dumps(summary,indent=2)[:900])
if __name__=='__main__':main()
