"""Re-render the original figure functions with paired A/C data; never write to the paper.
Run from any directory with the study's PyTorch Python environment.
"""
from pathlib import Path
import sys, json, copy, shutil, hashlib, difflib, csv, pickle
import numpy as np
OUT=Path(__file__).resolve().parents[1]; STUDY=OUT.parents[1]
REF=STUDY/'inputs/reference/paper_candidate_baseline_20260925'
RT=OUT/'runtime'; PAPER=RT/'paper'; FIG=OUT/'figures'; DATA=OUT/'data'; QA=OUT/'qa'
for p in [PAPER,FIG,DATA,QA]:p.mkdir(parents=True,exist_ok=True)
def dump(p,o):p.write_text(json.dumps(o,indent=2,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else float(o))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for src in sorted((STUDY/'source_original').rglob('*.py')):
 if '__pycache__' in str(src):continue
 dst=RT/src.relative_to(STUDY/'source_original');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sources[str(src.relative_to(STUDY))]=sha(src)
for src in (REF/'figure_scripts').glob('*.py'):
 shutil.copy2(src,PAPER/src.name);sources[str(src.relative_to(STUDY))]=sha(src)
pred=RT/'experiments/predictions';pred.mkdir(parents=True,exist_ok=True)
shutil.copy2(STUDY/'inputs/reference/paper_sweeps_predictions.json',pred/'paper_sweeps_predictions.json')
sources['inputs/reference/paper_sweeps_predictions.json']=sha(pred/'paper_sweeps_predictions.json')
# Keep the same named event in each progression column. Original A determines the column order.
ep=PAPER/'extra_figures.py';old=ep.read_text();new=old.replace('pairs = sorted(zip(keys, [frame_for(k) for k in keys]), key=lambda kf: kf[1]);', 'pairs = matched_event_order(name, list(zip(keys, [frame_for(k) for k in keys])));')
assert new!=old;ep.write_text(new)
(OUT/'runtime_patch.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen/extra_figures.py',tofile='runtime/paper/extra_figures.py')))
dump(DATA/'source_hashes.json',sources)
sys.path[:0]=[str(PAPER),str(RT)]
from experiments import db
BASE=[json.loads(p.read_text()) for p in sorted((STUDY/'inputs/baseline/runs').glob('*/record.json'))]
BASE=[r for r in BASE if r.get('status')=='completed' and r.get('metrics')]
DES=json.loads((STUDY/'candidate_figures/data/designs.json').read_text());DD={d['parent_run_id']:d for d in DES}
assert len(BASE)==132 and len(DD)==132
RUNS={r['parent_run_id']:r for p in (STUDY/'runs').glob('*/record.json') if (r:=json.loads(p.read_text()))['role']=='extension'}
STATES={}
for state in ['A','C']:
 rr=copy.deepcopy(BASE)
 for r in rr:
  rid=r['run_id'];r['run_dir']=str(STUDY/'inputs/baseline/runs'/rid);r['trajectory']=str(STUDY/'inputs/baseline/trajectories'/f'{rid}.npz')
  if state=='C' and rid in RUNS:
   run=RUNS[rid];d=DD[rid]['C'];r['metrics'].update(strength_Nm=d['strength'],work_to_failure_J_m2=d['integral'],failure_strain=d['end_strain'],strain_at_peak=d['peak_strain'],failure_observed=d['failure_observed'])
   r['trajectory']=str(STUDY/'runs'/run['case_id']/'trajectory.npz');r['broken_bond_events']=json.loads((STUDY/'runs'/run['case_id']/'damage_events.json').read_text())
 STATES[state]=rr
ACTIVE=STATES['A'];db.all_records=lambda:ACTIVE;db.load_record=lambda rid:next(r for r in ACTIVE if r['run_id']==rid)
import make_figures as MF
import extra_figures as EX
import mechanism_figure as ME
from analysis import ashby_chart as ASH
from analysis import fracture_viz as FV
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from matplotlib.collections import PathCollection
from matplotlib.text import Text
FROZEN_CAL=MF.sigma_net_straight();MF.sigma_net_straight=lambda:FROZEN_CAL
CURRENT='A';EVENTS={};TRAJCACHE={};COLOR={};EVENT_EXPORT=[]
def loadtraj(path):
 if path not in TRAJCACHE:
  original_path=path
  if not Path(path).exists():
   src=STUDY.parent/'carbon_discovery/trajectories'/Path(path).name
   target=OUT/'inputs/trajectories'/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
   assert sha(src)==sha(target);sources[str(src.relative_to(STUDY.parent))]='external-copy:'+sha(src)
   path=str(target)
  with np.load(path) as d:
   T={k:d[k] for k in d.files if not k.startswith('bonds_')}
   if 'bonds_0' in d.files:T['bonds']=[d[f'bonds_{k}'] for k in range(len(T['positions']))]
   else:
    T['bonds']=[]
    for k in range(len(T['positions'])):
     with np.load(Path(path).parent/'frames'/f'{k:05d}.npz') as fr:T['bonds'].append(fr['bonds'])
  TRAJCACHE[original_path]=T
  path=original_path
 return TRAJCACHE[path]
EX.load_traj=loadtraj;ME.load_traj=loadtraj
def eventframes(T):
 sels=FV.select_event_frames(T)
 return {k:next((i for lab,i in sels if k in lab.split(' = ')),0 if k=='relaxed' else len(T['eps_x'])-1) for k in ['relaxed','first_damage','peak_load','post_peak','final']}
ROWS=['S2_slit_0deg','S7_slit_20deg','S5_H2_D40_W8_ellipseX','S7_H2_veinsX_W12_ellipseX']
for name in ROWS:
 vals=[]
 for state in ['A','C']:
  r=next(r for r in STATES[state] if r['name']==name);T=loadtraj(r['trajectory']);ev=eventframes(T)
  for key,i in ev.items():
   vals.append(FV.atom_colors(T,i,'energy_rel')[0]);EVENT_EXPORT.append(dict(state=state,name=name,event=key,frame=i,strain=float(T['eps_x'][i]),stress=float(T['sigma_xx'][i]*16.0217663)))
 COLOR[name]=(min(0,min(np.percentile(v,1) for v in vals)),max(np.percentile(v,99) for v in vals))
def eventorder(name,pairs):
 global DRAW_NAME
 DRAW_NAME=name
 if CURRENT=='A':EVENTS[name]=[k for k,_ in sorted(pairs,key=lambda z:z[1])]
 return sorted(pairs,key=lambda z:EVENTS[name].index(z[0]))
EX.matched_event_order=eventorder
def frame(ax,T,f,field,vmin,vmax,**kw):return FV.draw_frame(ax,T,f,field,*COLOR[DRAW_NAME],**kw)
EX.draw_frame=frame
dump(DATA/'progression_events.json',EVENT_EXPORT)
ORIG_LSA=ASH.linear_sum_assignment;SLOTS=None
def slots(cost):
 global SLOTS
 if SLOTS is None:SLOTS=ORIG_LSA(cost)
 return SLOTS
ASH.linear_sum_assignment=slots
# Cache the identical relaxed thumbnails without changing the rendering function.
ORIG_TH=ASH.thumbnail;THUMBCACHE={}
def thumb(r,px=240):
 key=(r['run_id'],px)
 if key not in THUMBCACHE:THUMBCACHE[key]=ORIG_TH(r,px)
 return THUMBCACHE[key]
ASH.thumbnail=MF.thumbnail=thumb
CAPTURE={}
def capture(fig,name,*a,**kw):CAPTURE[name]=fig
MF.save=EX.save=capture
class Captured(Exception):pass
def captureash(fig,name,*a):
 CAPTURE['fig2_design_space']=fig
 raise Captured()
ASH.save=captureash
SPECS=[('fig2_design_space','Figure 3','Ashby charts'),('fig3_alignment_cliff','Figure 6','Slit-angle and overlap controls'),('fig4_hierarchy_alignment','Figure 8','Alignment and hierarchy'),('fig5_costs','SI: costs','Disorder, gradients and flaw halos'),('fig7_progression','Figure 7','Fracture progression'),('fig_mechanisms','Figure 9','Mechanism overview'),('fig1_rule_of_mixtures','Figure 4','Density and efficiency'),('fig2_load_paths','Figure 5','Load paths')]
FIGURES={};NUMBERS={};AUDIT=[];ARTISTS=[];CLIPS={}
def relabel(fig):
 for t in fig.findobj(Text):
  s=t.get_text()
  if s.startswith('slits across load\n'):s=s.replace('slits across load\n','slits across\nload\n')
  if s.startswith('slits along load\n'):s=s.replace('slits along load\n','slits along\nload\n')
  s=s.replace('gray band: single-level scatter','gray band: all-mesh residual s.d.')
  t.set_text(s)
 for ax in fig.axes:
  for direction in ['x','y']:
   label=getattr(ax,'get_'+direction+'label')()
   if 'work to failure' in label:label=label.replace('work to failure','endpoint integral').replace(', proxy','')
   if 'failure strain' in label:label='recorded ending strain $\\varepsilon_{\\mathrm{end}}$'
   getattr(ax,'set_'+direction+'label')(label)
def marks(fig,name,state):
 """Point-linked censoring + residual markers on the principal affected panels only."""
 by={r['name']:r for r in STATES[state]};pairs=[]
 if name=='fig2_design_space':
  for r in [r for r in STATES[state] if r['campaign']=='discovery']:
   pairs.extend([(fig.axes[2],r,MF.W(r),MF.sig(r),True),(fig.axes[3],r,MF.M(r,'failure_strain'),MF.sig(r),True)])
 elif name=='fig4_hierarchy_alignment':
  ids=set(json.loads((STUDY/'candidate_figures/data/alignment_population.json').read_text()))
  pairs=[(fig.axes[3],r,MF.Aidx(r),MF.W(r),True) for r in STATES[state] if r['run_id'] in ids]
 elif name=='fig3_alignment_cliff':
  for r in STATES[state]:
   if r['family']=='slit_array' and r['params'].get('period_x')==30 and r['params'].get('orientation','zigzag')=='zigzag' and .17<=(r.get('porosity') or 0)<=.23:pairs.append((fig.axes[1],r,r['params']['angle_deg'],MF.sig(r),False))
 for ax,r,x,y,endpoint in pairs:
  d=DD[r['run_id']];e=d[state] or d['A'];flag=state=='C' and d['C'] and d['residual_flag']
  if endpoint and not e['failure_observed']:ax.scatter([x],[y],marker='>',s=29,facecolors='white',edgecolors=MF.famcol(r),linewidths=.65,zorder=6)
  if flag:ax.scatter([x],[y],marker='x',s=9,color='0.25',linewidths=.42,zorder=7)
def paired_export(name,fa,fc):
 assert len(fa.axes)==len(fc.axes)
 relabel(fa);relabel(fc)
 for i,(a,c) in enumerate(zip(fa.axes,fc.axes)):
  # Apply the same view to both; preserve every original panel and marker population.
  for direction in ['x','y']:
   va=getattr(a,'get_'+direction+'lim')();vc=getattr(c,'get_'+direction+'lim')()
   lim=(min(va[0],vc[0]),max(va[1],vc[1]))
   if name=='fig2_design_space' and i==3 and direction=='x':lim=(.08,.90)
   if name=='fig2_design_space' and i==2 and direction=='x':lim=(.27,5.8)
   if name=='fig4_hierarchy_alignment' and i==3 and direction=='y':lim=(.3,3.25)
   getattr(a,'set_'+direction+'lim')(lim);getattr(c,'set_'+direction+'lim')(lim)
  if name=='fig2_design_space' and i==3:
   for ax in [a,c]:ax.xaxis.set_major_locator(FixedLocator([.1,.2,.3,.5,.8]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
  # Freeze the axes allocation; equal-aspect snapshot boxes now show the same physical field.
  c.set_position(a.get_position(original=True))
 for f in [fa,fc]:f.canvas.draw()
 # Retain each source's original text-placement strategy; no figure function is redesigned.
 if name not in ['fig2_design_space','fig7_progression','fig_mechanisms']:
  MF.resolve(fa);MF.resolve(fc)
 for st,f in [('A',fa),('C',fc)]:marks(f,name,st)
 for f in [fa,fc]:f.canvas.draw()
 bb=Bbox.union([f.get_tightbbox(f.canvas.get_renderer()) for f in [fa,fc]]).padded(.09)
 for state,f in [('A',fa),('C',fc)]:
  for ext in ['pdf','svg','png']:f.savefig(FIG/f'{name}__{state}.{ext}',bbox_inches=bb,facecolor='white',dpi=220)
  for i,ax in enumerate(f.axes):
   AUDIT.append(dict(figure=name,state=state,panel_axis=i,bounds=list(ax.get_position().bounds),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),xscale=ax.get_xscale(),yscale=ax.get_yscale()))
   for k,coll in enumerate(ax.collections):
    if isinstance(coll,PathCollection):
     for j,(x,y) in enumerate(coll.get_offsets()):ARTISTS.append(dict(figure=name,state=state,axis=i,artist='scatter',series=k,point=j,x=float(x),y=float(y)))
   for k,line in enumerate(ax.lines):
    for j,(x,y) in enumerate(zip(line.get_xdata(),line.get_ydata())):
     try:ARTISTS.append(dict(figure=name,state=state,axis=i,artist='line',series=k,point=j,x=float(x),y=float(y)))
     except (TypeError,ValueError):pass
   for k,patch in enumerate(ax.patches):
    if hasattr(patch,'get_height') and hasattr(patch,'get_x'):ARTISTS.append(dict(figure=name,state=state,axis=i,artist='bar_or_rectangle',series=k,point=0,x=float(patch.get_x()+patch.get_width()/2),y=float(patch.get_height())))
  # PDF-point clips enclosing the original panel boxes, for matched magnified views.
  rc=f.canvas.get_renderer();rects=[]
  for ax in f.axes:
   ab=ax.get_tightbbox(rc).transformed(f.dpi_scale_trans.inverted())
   rects.append([72*(ab.x0-bb.x0),72*(bb.y1-ab.y1),72*(ab.x1-bb.x0),72*(bb.y1-ab.y0)])
  CLIPS.setdefault(name,{})[state]=rects
 if name=='fig2_design_space':
  hs=json.loads((STUDY/'candidate_figures/data/boundary_vertices.json').read_text())
  for i,(panel,xkey,ykey) in enumerate([('strength_density','rho','strength'),('modulus_density','rho','modulus'),('strength_integral','integral','strength'),('strength_endpoint','end_strain','strength')]):
   ax=fc.axes[i]
   for h in hs:
    if h['state']!='A' or h['panel']!=panel:continue
    v=h['vertices'];xy=np.array([[q['x'],q['y']] for q in v]);xy=np.vstack([xy,xy[:1]]) if len(xy)>2 else xy
    ax.plot(xy[:,0],xy[:,1],'--',lw=.95,color=MF.FAMCOL[h['family']],zorder=5)
   if i==1:continue
   for d in DES:
    if d['campaign']!='discovery' or d['C'] is None:continue
    xa=d[xkey] if xkey=='rho' else d['A'][xkey];xc=d[xkey] if xkey=='rho' else d['C'][xkey];ya=d['A'][ykey];yc=d['C'][ykey]
    ax.scatter([xa],[ya],s=15,facecolors='white',edgecolors=MF.FAMCOL[d['family']],linewidths=.6,zorder=6)
    if abs(xa-xc)>1e-10 or abs(ya-yc)>1e-10:ax.annotate('',xy=(xc,yc),xytext=(xa,ya),arrowprops=dict(arrowstyle='->',color=MF.FAMCOL[d['family']],lw=.55,alpha=.7,shrinkA=2,shrinkB=2),zorder=5)
  for ext in ['pdf','svg','png']:fc.savefig(FIG/f'{name}__overlay.{ext}',bbox_inches=bb,facecolor='white',dpi=220)
 print('EXPORTED',name,flush=True)
for state in ['A','C']:
 CURRENT=state;ACTIVE=STATES[state];MF.ALL=ACTIVE;MF.DISC=[r for r in ACTIVE if r['campaign']=='discovery'];MF.PAPER=[r for r in ACTIVE if r['campaign']=='paper'];MF.BY={r['name']:r for r in ACTIVE};CAPTURE.clear()
 try:ASH.main()
 except Captured:pass
 for fn in [MF.fig1,MF.fig2,MF.fig3,MF.fig4,MF.fig5,EX.fig_progression]:fn()
 CAPTURE['fig_mechanisms']=ME.fig_mechanisms()
 FIGURES[state]=dict(CAPTURE);NUMBERS[state]=dict(figure_numbers=copy.deepcopy(MF.NUM),mechanism_numbers=copy.deepcopy(ME.NUMS))
 print('BUILT',state,flush=True)
for name,_,_ in SPECS:paired_export(name,FIGURES['A'][name],FIGURES['C'][name])
dump(DATA/'axes_correspondence.json',AUDIT);dump(DATA/'source_figure_numbers.json',NUMBERS);dump(DATA/'panel_clips.json',CLIPS);dump(DATA/'figure_list.json',SPECS)
with (DATA/'rendered_artist_coordinates.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(ARTISTS[0]));w.writeheader();w.writerows(ARTISTS)
for fname in ['designs.json','primary_A_B_C_effects.csv','plotted_design_coordinates.csv','boundary_vertices.json','boundary_vertices.csv','quantitative_summary.json','disorder_groups.csv']:
 shutil.copy2(STUDY/'candidate_figures/data'/fname,DATA/fname)
# Validate matched geometry and retained populations after all adaptations.
for name,_,_ in SPECS:
 a=[x for x in AUDIT if x['figure']==name and x['state']=='A'];c=[x for x in AUDIT if x['figure']==name and x['state']=='C']
 assert len(a)==len(c)
 for aa,cc in zip(a,c):
  for key in ['bounds','xlim','ylim']:assert np.allclose(aa[key],cc[key]),(name,key,aa[key],cc[key])
  for key in ['xlabel','ylabel','xscale','yscale']:assert aa[key]==cc[key]
assert len(MF.DISC)==96 and MF.NUM['align_n']==40 and ME.NUMS['lig_n']==77
assert all(r['metrics']['modulus_2d_Nm']==a['metrics']['modulus_2d_Nm'] for r,a in zip(STATES['C'],STATES['A']))
assert all(sha((STUDY.parent if h.startswith('external-copy:') else STUDY)/p)==h.split(':')[-1] for p,h in sources.items())
dump(DATA/'source_hashes.json',sources)
dump(QA/'validation.json',dict(same_panels_axes_limits_labels_and_scales=True,same_96_discovery=True,same_40_alignment=True,same_77_ligament=True,frozen_predictions=True,source_files_unchanged=True,pairs=8,primary_replacements=34,preregistered_figure_excluded=True))
plt.close('all')
