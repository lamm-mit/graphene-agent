"""Candidate figures only. All reads/writes stay inside the isolated study."""
from pathlib import Path
import sys,json,csv,math,hashlib,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter,FixedLocator,NullFormatter
from ase.io import read as ase_read
from build_data import ROOT,OUT,DATA,FAMCOL,FAMNAME,FAMORDER,THUMBS,GROUPS,REF,value,load,sha
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
ROWS=load('candidate_figures/data/designs.json');BY={r['name']:r for r in ROWS};DISC=[r for r in ROWS if r['campaign']=='discovery'];SUM=load('candidate_figures/data/quantitative_summary.json');HULL=load('candidate_figures/data/boundary_vertices.json');MESH=[r for r in ROWS if r['parent_run_id'] in load('candidate_figures/data/alignment_population.json')]
RUNS={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/record.json')}
CURVE_ROWS=list(csv.DictReader((DATA/'stress_strain_curves.csv').open()));CURVES={}
for q in CURVE_ROWS:CURVES.setdefault((q['case_id'],q['trajectory']),[]).append(q)
REG=[];POINTS=[];FRAMEINFO=[]
A_COLOR='.55';B_COLOR='#087e8b';C_COLOR='#cf5d11'
def style(ax,title=None):
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16,lw=.4);ax.set_axisbelow(True)
 if title:ax.set_title(title,loc='left',pad=7)
def title(fig,text):fig.suptitle(text,fontsize=11,y=.985)
def footer(fig,text='A: archive   B: rerun at original stop   C: extended rerun   Open: endpoint censored   +: residual exceedance'):
 fig.text(.5,.012,text,ha='center',va='bottom',fontsize=6.5)
def label(ax,s):ax.text(-.13,1.035,s,transform=ax.transAxes,fontweight='bold',fontsize=10)
def save(fig,name,caption,original=None,cases=None):
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}')
 (FIG/f'{name}.caption.txt').write_text(caption+'\n')
 REG.append(dict(id=name,title=fig._suptitle.get_text() if fig._suptitle else name,caption=caption,original=original,cases=cases or [],files=[f'figures/{name}.{e}' for e in ['pdf','svg','png']]))
 plt.close(fig)
def famlegend(fig,y=.05,ncol=5):
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=FAMCOL[f],label=FAMNAME[f],ms=4) for f in FAMORDER],loc='lower center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False,columnspacing=1,fontsize=6.5)
def marker(r):return 's' if r['source_phase']=='earlier II' else 'o'
def obs(ax,r,state,x,y,color=None,size=18,mark=None):
 endpoint=r[state] or r['A'];c=color or (A_COLOR if state=='A' else FAMCOL[r['family']]);m=mark or marker(r)
 if m=='_':ax.scatter([x],[y],s=size,marker=m,color=c,linewidths=1.0,zorder=4)
 else:ax.scatter([x],[y],s=size,marker=m,facecolors=c if endpoint['failure_observed'] else 'white',edgecolors=c,linewidths=.6,zorder=4)
 if state!='A' and r['C'] and r['residual_flag']:ax.scatter([x],[y],marker='+',s=size*1.2,color='k',linewidths=.55,zorder=5)
 POINTS.append(dict(figure=CURRENT,panel=ax.get_label(),name=r['name'],parent_run_id=r['parent_run_id'],state=state,coordinate_source=state if r[state] else 'A carried forward',x=float(x),y=float(y),endpoint_censored=not endpoint['failure_observed'],new_run_residual_flag=r['residual_flag'] if state!='A' and r['C'] else None))
def paired(ax,rr,xkey,ykey,states=('A','C')):
 for r in rr:
  x=[value(r,s,xkey) for s in states];y=[value(r,s,ykey) for s in states]
  if r['C'] and states[0]!=states[1]:ax.plot(x,y,color=FAMCOL[r['family']],alpha=.55,lw=.65,zorder=2)
  for i,s in enumerate(states):
   if i==0 or r['C']:obs(ax,r,s,x[i],y[i],size=15)
def thumbnail(ax,r):
 p=ROOT/'inputs/baseline/runs'/r['parent_run_id']/'relaxed.extxyz';a=ase_read(p);pos=a.positions;L=a.cell.lengths();ax.scatter(pos[:,0]%L[0],pos[:,1]%L[1],s=.12,c='.18',linewidths=0,rasterized=False);ax.set(xlim=(0,L[0]),ylim=(0,L[1]));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set(color=FAMCOL[r['family']],linewidth=.8)
def hullpanel(ax,panel,xkey,ykey,states=('A','C')):
 ax.set_label(panel);style(ax);ax.set_xscale('log');ax.set_yscale('log')
 for h in HULL:
  if h['panel']!=panel or h['state'] not in states:continue
  xy=np.array([[v['x'],v['y']] for v in h['vertices']]);c=FAMCOL[h['family']];s=h['state'];closed=np.vstack([xy,xy[0]]) if len(xy)>2 else xy
  if len(xy)>2 and s==states[-1]:ax.fill(xy[:,0],xy[:,1],color=c,alpha=.045,zorder=0)
  ax.plot(closed[:,0],closed[:,1],color=c,ls='--' if s==states[0] else '-',lw=.85,alpha=.85,zorder=1)
 paired(ax,DISC,xkey,ykey,states)
 ax.set_xlabel({'rho':'Relative areal density','integral':'Integral to stated endpoint (J/m²)','end_strain':'Recorded ending strain'}[xkey]);ax.set_ylabel('2D modulus (N/m)' if ykey=='modulus' else 'Maximum recorded stress (N/m)')
 ax.yaxis.set_major_locator(FixedLocator([2,5,10,20,40,100,300]));ax.yaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
 if xkey=='rho':ax.set_xlim(.47,1.07);ax.xaxis.set_major_locator(FixedLocator([.5,.6,.8,1]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
 if ykey=='strength':ax.set_ylim(3,48)
 if panel=='modulus_density':ax.set_ylim(1.5,400)
 if xkey=='end_strain':ax.set_xlim(.07,1.0);ax.xaxis.set_major_locator(FixedLocator([.1,.2,.4,.8]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
 if xkey=='integral':ax.set_xlim(.2,8);ax.xaxis.set_major_locator(FixedLocator([.25,.5,1,2,4,8]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
def ashby():
 global CURRENT
 CURRENT='01_ashby_A_to_C';fig=plt.figure(figsize=(8.1,9.5));gs=fig.add_gridspec(2,2,left=.10,right=.97,bottom=.19,top=.76,wspace=.30,hspace=.32)
 panels=[('strength_density','rho','strength'),('modulus_density','rho','modulus'),('strength_integral','integral','strength'),('strength_endpoint','end_strain','strength')]
 for i,(p,x,y) in enumerate(panels):
  ax=fig.add_subplot(gs[i//2,i%2]);hullpanel(ax,p,x,y);label(ax,'abcd'[i])
  if x=='rho':
   for k in ([38.8,20,10,5] if y=='strength' else [245,100,30]):ax.plot([.48,1.06],np.array([.48,1.06])*k,':',color='.7',lw=.55,zorder=0)
 for i,n in enumerate(THUMBS):
  ax=fig.add_axes([.075+(i%6)*.153,.805+(1-i//6)*.075,.13,.055]);thumbnail(ax,BY[n]);ax.set_title(n.split('_',1)[1],fontsize=5.3,pad=2)
 title(fig,'Candidate Fig. 3: the same 96 discovery designs');famlegend(fig,.055)
 footer(fig,'Dashed hull: A   Solid hull: C   Paired points: 20 reruns   Open: censored endpoint   +: new-run residual flag')
 save(fig,CURRENT,'Same 96 discovery designs, with the 20 targeted Phase-I source cases paired A to C. Unrerun designs retain archived coordinates. Family envelopes use ConvexHull in log10 coordinates, as in the original plotting source; they are observed numerical response envelopes, not confidence or physical feasibility bounds. Modulus and density retain original definitions/values. Dotted density guides are constant specific properties. Thumbnails are unchanged archived relaxed geometries. No absence of a residual marker establishes convergence.','fig2_design_space.pdf',[r['name'] for r in DISC])
 CURRENT='02_ashby_B_to_C';fig,axs=plt.subplots(1,2,figsize=(8,3.9));fig.subplots_adjust(left=.085,right=.97,bottom=.27,top=.84,wspace=.32)
 for ax,(p,x,y),l in zip(axs,panels[2:],'ab'):hullpanel(ax,p,x,y,('B','C'));label(ax,l)
 title(fig,'Added-loading effects within the reruns');famlegend(fig,.06)
 footer(fig,'Dashed: B for 20 reruns + archived A for 76 others   Solid: C for 20 reruns + the same 76 A values')
 save(fig,CURRENT,'Same 96-design discovery population. This mixed-population diagnostic isolates B-to-C coordinate changes for the 20 rerun designs; the other 76 retain A in both envelopes, and no B trajectory is invented for them. Open markers identify unmet operational failure criteria, not exact failure strains. Crosses indicate new-run residual exceedances.','fig2_design_space.pdf',[r['name'] for r in DISC])
def angle():
 global CURRENT
 CURRENT='03_slit_angle';sd=list(csv.DictReader((DATA/'slit_angle_predictions_and_responses.csv').open()));sweep=sorted([q for q in sd if q['in_angle_sweep']=='True'],key=lambda q:float(q['theta']));ctrl=sorted([q for q in sd if float(q['theta'])==20],key=lambda q:float(q['porosity']))
 fig,axs=plt.subplots(1,3,figsize=(8.5,4.7),gridspec_kw={'width_ratios':[1.4,1,.9]});fig.subplots_adjust(left=.07,right=.98,bottom=.23,top=.66,wspace=.43)
 chosen=[]
 for theta in [0,10,20,30,45,90]:chosen.append(min([q for q in sweep if float(q['theta'])==theta],key=lambda q:abs(float(q['porosity'])-.2)))
 for i,q in enumerate(chosen):
  ax=fig.add_axes([.07+i*.153,.77,.125,.12]);thumbnail(ax,BY[q['name']]);ax.set_title(f"{float(q['theta']):g}°; O={float(q['overlap_A']):+.1f} Å\nφ={float(q['porosity']):.3f}",fontsize=6,pad=2)
 for i,ax in enumerate(axs):style(ax);label(ax,'abc'[i]);ax.set_label('slit_'+'abc'[i])
 th=[float(q['theta']) for q in sweep]
 for key,ls,col in [('rule','-','.65'),('mechanism','--','#c0392b')]:axs[0].plot(th,[float(q[key]) for q in sweep],ls,c=col,lw=.85,label=key+' (original)')
 for q in sweep:
  r=BY[q['name']];x=float(q['theta']);axs[0].plot([x]*3,[value(r,s,'strength') for s in ['A','B','C']],c='.65',lw=.7)
  for s,dx,col,m in [('A',-.6,A_COLOR,None),('B',0,B_COLOR,'_'),('C',.6,FAMCOL['slit_array'],None)]:obs(axs[0],r,s,x+dx,value(r,s,'strength'),color=col,mark=m,size=22)
 axs[0].set(xlabel='Slit angle to load (°)',ylabel='Maximum recorded stress (N/m)',xlim=(-4,94),ylim=(0,36));axs[0].legend(fontsize=5.8,frameon=False,loc='upper right')
 # Retain original geometric overlap/clearance panel.
 for key,col,m in [('overlap_A','#c0392b','o'),('clearance_A','#1f77b4','s')]:axs[1].plot(th,[float(q[key]) for q in sweep],marker=m,ms=3,c=col,lw=.8,label='tip overlap O' if key=='overlap_A' else 'row clearance t')
 axs[1].axhline(0,c='.5',lw=.6);axs[1].set(xlabel='Slit angle (°)',ylabel='Length (Å)');axs[1].legend(frameon=False,fontsize=6)
 for i,q in enumerate(ctrl):
  r=BY[q['name']];axs[2].plot([i]*3,[value(r,s,'strength') for s in ['A','B','C']],c='.65',lw=.7)
  for s,dx,col,m in [('A',-.12,A_COLOR,None),('B',0,B_COLOR,'_'),('C',.12,FAMCOL['slit_array'],None)]:obs(axs[2],r,s,i+dx,value(r,s,'strength'),color=col,mark=m,size=25)
  axs[2].scatter(i-.23,float(q['rule']),marker='_',c='.55',s=35);axs[2].scatter(i+.23,float(q['mechanism']),marker='_',c='#c0392b',s=35)
 axs[2].set(xticks=range(len(ctrl)),xticklabels=[f"{float(q['porosity']):.2f}" for q in ctrl],xlabel='Porosity at 20°',ylabel='Maximum stress (N/m)',ylim=(0,36))
 title(fig,'Slit-angle response: original and extended observations');footer(fig,'Gray: A   Teal dash: B   Orange: C   Circle: Phase-I source   Square: earlier Phase-II source\nOpen: censored endpoint   +: new-run residual flag; predictors retain original calibration')
 save(fig,CURRENT,'Candidate update of slit-angle and overlap-control observations, with original geometric formulas and frozen prediction values/calibration. A/B/C offsets are for visibility only. Shapes preserve source-phase attribution. The geometry panel is unchanged; all new runs are human-AI Phase-II work. Most prediction curves were preregistered for the paper sweeps; existing anchor-design values are the original plotting formula, distinguished in the exported table. Later peaks do not by themselves establish a new mechanism.','fig3_alignment_cliff.pdf',[q['name'] for q in sd])
def alignment():
 global CURRENT
 CURRENT='04_alignment';fig,axs=plt.subplots(2,2,figsize=(8,6.7));fig.subplots_adjust(left=.085,right=.97,bottom=.14,top=.90,wspace=.31,hspace=.38)
 for ax,l in zip(axs.flat,'abcd'):style(ax);label(ax,l);ax.set_label('alignment_'+l)
 for ax,key in [(axs[0,0],'strength'),(axs[0,1],'integral')]:
  paired(ax,MESH,'alignment',key);ax.set_xlabel('Original alignment index A');ax.set_ylabel('Maximum recorded stress (N/m)' if key=='strength' else 'Integral to stated endpoint (J/m²)')
  if key=='strength':
   for lev,col,label_ in [(1,'#1f77b4','H1'),(2,'#d62728','H2')]:
    for state,ls in [('A','--'),('C','-')]:
     fit=SUM['alignment'][state][label_];x=np.linspace(-.27,.65,80);ax.plot(x,fit['slope']*x+fit['intercept'],ls,c=col,lw=.85,label=f'{label_} {state}')
   ax.legend(frameon=False,ncol=2,fontsize=6)
  else:ax.text(.03,.96,f"Pearson r: {SUM['alignment']['A']['integral_correlation']['pearson']:.3f} → {SUM['alignment']['C']['integral_correlation']['pearson']:.3f}",transform=ax.transAxes,va='top',fontsize=7)
 nested=sorted([r for r in MESH if r['level']>1],key=lambda r:r['alignment']);pr=list(csv.DictReader((DATA/'hierarchy_premiums.csv').open()));lookup={(q['name'],q['state']):float(q['premium_N_m']) for q in pr}
 for i,r in enumerate(nested):
  y=[lookup[r['name'],s] for s in ['A','C']];axs[1,0].plot([i-.12,i+.12],y,c=FAMCOL[r['family']],lw=.7);axs[1,0].scatter(i-.12,y[0],s=13,c=A_COLOR);axs[1,0].scatter(i+.12,y[1],s=13,c=FAMCOL[r['family']])
 axs[1,0].axhline(0,c='.5',lw=.6);axs[1,0].set(xlabel='Nested design, sorted by original A',ylabel='Premium over H1 OLS trend (N/m)');axs[1,0].set_title(f"Mean premium: A {SUM['alignment']['A']['premium_mean']:.2f}; C {SUM['alignment']['C']['premium_mean']:.2f} N/m",fontsize=7)
 for i,group in enumerate(['H1','H2','all']):
  axs[1,1].plot([i-.12,i+.12],[SUM['alignment'][s][group]['slope'] for s in ['A','C']],c='.6');axs[1,1].scatter(i-.12,SUM['alignment']['A'][group]['slope'],c=A_COLOR,s=25);axs[1,1].scatter(i+.12,SUM['alignment']['C'][group]['slope'],c='#8c4cbb',s=25)
 axs[1,1].set(xticks=[0,1,2],xticklabels=['H1','H2','all 40'],ylabel='OLS strength slope (N/m per A)',xlabel='Retrospective fits; same population')
 title(fig,'Alignment and hierarchy: fixed original 40-mesh population');footer(fig,'Gray / dashed: archive A   Colored / solid: C   Paired observations: eight reruns\nThe original set includes one validation mesh; no designs are added or removed. +: residual flag')
 save(fig,CURRENT,'Fixed 40-mesh population selected by the original plotting code, including validation mesh_size32. Eight source cases are replaced one-to-one. Panels a/b compare strength and endpoint integral versus the unchanged descriptor; c shows nested-design premiums relative to each state\'s H1 OLS fit; d shows retrospective slopes. No significance/convergence claim is inferred from these refits, and no separate hierarchy-follow-up data enter. Full premium IDs and population are exported.','fig4_hierarchy_alignment.pdf',[r['name'] for r in MESH])
def disorder(compact=False):
 global CURRENT
 CURRENT='06_disorder_compact' if compact else '05_disorder_full';groups=GROUPS[:3] if compact else GROUPS;n=len(groups);fig,axs=plt.subplots(2,n,figsize=(7.4 if compact else 10.4,5));fig.subplots_adjust(left=.08,right=.985,bottom=.16,top=.88,hspace=.38,wspace=.4)
 for j,(name,levels) in enumerate(groups):
  for i,key in enumerate(['strength','integral']):
   ax=axs[i,j];style(ax);ax.set_label(name+'_'+key);label(ax,'abcdefghij'[i*n+j])
   if i==0:ax.set_title(name,fontsize=8)
   for k,(level,names) in enumerate(levels.items()):
    rr=[BY[z] for z in names]
    for ri,r in enumerate(rr):
     dx=(ri-(len(rr)-1)/2)*.04;x0=k-.11+dx;x1=k+.11+dx;y0=value(r,'A',key);y1=value(r,'C',key);ax.plot([x0,x1],[y0,y1],c='.7',lw=.5)
     obs(ax,r,'A',x0,y0,size=12);obs(ax,r,'C',x1,y1,size=12)
    for state,offset,col in [('A',-.14,A_COLOR),('C',.14,'#c0392b' if i==0 else '#1f77b4')]:
     mean=np.mean([value(r,state,key) for r in rr]);ax.plot([k+offset-.10,k+offset+.10],[mean]*2,c=col,lw=1.3)
   if j<4:
    rv=[value(BY[n],'A',key) for n in REF];ax.axhspan(np.mean(rv)-np.std(rv),np.mean(rv)+np.std(rv),color='.8',alpha=.18,lw=0)
   ax.set_xticks(range(len(levels)));ax.set_xticklabels(list(levels),rotation=35 if j>=3 else 0,ha='right' if j>=3 else 'center',fontsize=6)
   if j==0:ax.set_ylabel('Maximum stress (N/m)' if i==0 else 'Endpoint integral (J/m²)')
 title(fig,'Disorder and controls: paired observations and fixed group means');footer(fig,'Left / gray: A   Right / color: C   Bars: group means   Shading: unchanged registry-reference mean ± SD\nEach seed remains one observation; + marks new-run residual exceedances.')
 save(fig,CURRENT,'Original seed/registry groups and their exact denominators are preserved. Individual A/C observations and means are shown for strength and the integral to a stated endpoint. Unrerun reference values carry forward. Gradient and flaw-halo panels are unchanged context in the full option; the compact version retains the three affected columns. Group means are neither fracture toughness nor validated failure-work means.','fig5_costs.pdf',[n for _,lev in groups for names in lev.values() for n in names])
def getcurve(case,branch):
 q=CURVES[(case,branch)];return np.array([float(r['strain']) for r in q]),np.array([float(r['stress_N_m']) for r in q]),np.array([r['residual_flag']=='True' for r in q])
def curvepanel(ax,case,branches=False):
 run=RUNS[case];x,y,bad=getcurve(case,'C');xa,ya,_=getcurve(case,'A');b=run['paired']['B_original_protocol_on_rerun'];c=run['paired']['C_extended'];style(ax);ax.set_label(case)
 ax.plot(xa,ya,'--',c=A_COLOR,lw=.9,label='archived A');ax.plot(x,y,c=FAMCOL[run['parent_family']],lw=1,label='primary C');ax.plot(x[bad],y[bad],'+',c='k',ms=3,mew=.4,alpha=.65)
 i=b['endpoint_index'];ax.fill_between(x[i:],y[i:],color=FAMCOL[run['parent_family']],alpha=.09);ax.axvline(b['end_strain'],c=B_COLOR,lw=.7,ls=':');ax.scatter([xa[-1]],[ya[-1]],facecolors='white' if not run['paired']['A_archived']['failure_observed'] else A_COLOR,edgecolors=A_COLOR,s=18,zorder=5)
 ax.scatter(x[-1],y[-1],marker='X' if c['failure_observed'] else '>',c=FAMCOL[run['parent_family']],s=27,zorder=5)
 ax.scatter(x[np.argmax(y)],max(y),marker='*',c='k',s=25,zorder=5)
 if branches:
  res=case.replace('ext__','res__')
  if res in RUNS:
   xr,yr,_=getcurve(res,'C');ax.plot(xr,yr,c='#007f86',lw=.9,label='half post-damage step')
  if '20deg' in case and 'phi' not in case:
   xr,yr,_=getcurve('quality__S7_slit_20deg_verified_Q4','C');ax.plot(xr,yr,c='#7b2e83',lw=.9,label='strict Q4')
 ax.set_xlabel('Engineering strain');ax.set_ylabel('2D stress (N/m)');ax.set_title(run['name'].split('_',1)[1],fontsize=8)
 return ax
MAINCASES=['ext__S7_slit_20deg','ext__P1_slit_35deg','ext__S2_slit_45deg','ext__P1_slit_30deg','ext__S7_slit_0deg_armchair','ext__S5_H2_D40_W8_ellipseX']
def curves():
 global CURRENT
 CURRENT='07_loading_examples';fig,axs=plt.subplots(2,3,figsize=(9,6));fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.15,hspace=.48,wspace=.38)
 for ax,case,l in zip(axs.flat,MAINCASES,'abcdef'):curvepanel(ax,case);label(ax,l)
 title(fig,'Separate archived and extended trajectories');footer(fig,'Dashed: A   Solid: primary rerun   Teal dotted: B   Shading: added integral B–C\nStar: recorded peak   X: operational failure stop   Arrow: censored endpoint   +: residual exceedance')
 save(fig,CURRENT,'Six examples show separate A and C paths, the chronological B cutoff, B-to-C added integral, peak, endpoint and force/transverse residual flags. The 30-degree slit remains censored at the extended cap. No archived/rerun trajectory is spliced. Stress-strain integrals are recorded endpoint integrals, not fracture toughness.','fig7_progression.pdf',MAINCASES)
 cases=sorted([c for c,r in RUNS.items() if r['role']=='extension'],key=lambda c:(RUNS[c]['source_phase'],c))
 for page in range(4):
  CURRENT=f'08_curve_atlas_{page+1}';fig,axs=plt.subplots(3,3,figsize=(9,8.2));fig.subplots_adjust(left=.075,right=.98,top=.90,bottom=.12,hspace=.55,wspace=.4)
  for i,ax in enumerate(axs.flat):
   if page*9+i>=len(cases):ax.axis('off');continue
   case=cases[page*9+i];curvepanel(ax,case);ax.set_title(f"{RUNS[case]['source_phase']} source | "+RUNS[case]['name'].split('_',1)[1],fontsize=7)
  title(fig,f'Complete 34-case curve atlas ({page+1}/4)');footer(fig,'A dashed / C solid; B dotted; shaded added integral; star peak; X driver stop; arrow censoring; + residual flag')
  save(fig,CURRENT,'Complete primary-cohort curves, separately labeled by source phase. All are new Phase-II extensions. The numerical legend and caveats of the compact curve comparison apply.','fig7_progression.pdf',cases[page*9:(page+1)*9])
def resolution():
 global CURRENT
 CURRENT='09_resolution_sensitivity';cases=['ext__S7_slit_0deg_armchair','ext__S5_H2_D40_W8_ellipseX','ext__S2_slit_45deg','ext__S7_slit_20deg'];fig,axs=plt.subplots(2,3,figsize=(10,6));fig.subplots_adjust(left=.07,right=.98,bottom=.15,top=.90,wspace=.38,hspace=.45)
 for ax,c,l in zip(axs.flat,cases,'abcd'):curvepanel(ax,c,True);label(ax,l)
 for j,key in enumerate(['maximum_recorded_stress_N_m','stress_strain_integral_J_m2']):
  ax=axs[1,j+1];style(ax);label(ax,'ef'[j]);vals=[]
  for c in cases:
   a=RUNS[c]['paired']['C_extended'][key];b=RUNS[c.replace('ext__','res__')]['paired']['C_extended'][key];vals.append(100*(b/a-1))
  ax.bar(range(4),vals,color=['#ff7f0e','#d62728','#ff7f0e','#ff7f0e']);ax.axhline(0,c='.4',lw=.6);ax.set_xticks(range(4));ax.set_xticklabels(['armchair','mesh','45°','20°'],fontsize=7);ax.set_ylabel('Half-step change (%)');ax.set_title('Peak stress' if j==0 else 'Endpoint integral')
  ax.set_ylim(min(min(vals)*1.32,-1),max(max(vals)*1.45,1))
  for i,v in enumerate(vals):ax.annotate(f'{v:+.1f}',(i,v),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=7)
 axs[0,0].legend(frameon=False,fontsize=5.8);axs[1,0].legend(frameon=False,fontsize=5.8)
 title(fig,'Resolution checks: sensitivity, not corrected ground truth');footer(fig,'Teal: half post-damage increment (elastic/minimum increments unchanged)   Purple: separate strict Q4 branch\nAll four scheduled resolution trajectories retain residual flags; two levels do not establish convergence.')
 save(fig,CURRENT,'Four preregistered numerical checks halve only the post-damage increment from 0.005 to 0.0025. Each is a fresh run from the same initial geometry. These are separate from primary C coordinates, and floating-point rerun variability is not isolated from step sensitivity. The strict 20-degree Q4 run is a separate solver branch. No branch is selected as corrected truth.',None,cases)
 CURRENT='10_strict_control_checks';fig,axs=plt.subplots(1,3,figsize=(9,3.6));fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.85,wspace=.36)
 ids=['quality__S1_pristine_zz_verified_Q4','quality__S1_pristine_zz_halfstep_Q7','quality__S1_pristine_zz_quarterstep_Q8'];cols=['#7b2e83','#007f86','#a65f00']
 for c,col,l in zip(ids,cols,['Q4','Q7 (half)','Q8 (quarter)']):
  x,y,b=getcurve(c,'C');axs[0].plot(x,y,c=col,lw=.95,label=l)
 for ax in axs:style(ax)
 axs[0].set(xlabel='Engineering strain',ylabel='2D stress (N/m)',xlim=(.18,.275));axs[0].legend(frameon=False,fontsize=6)
 for ax,k,lab in zip(axs[1:],['maximum_recorded_stress_N_m','stress_strain_integral_J_m2'],['Peak stress (N/m)','Endpoint integral (J/m²)']):
  yy=[RUNS[c]['paired']['C_extended'][k] for c in ids];ax.plot([1,.5,.25],yy,'o-',c='#7b2e83');ax.set(xlabel='Increment factor relative to Q4',ylabel=lab,xticks=[.25,.5,1]);ax.invert_xaxis()
 for ax,l in zip(axs,'abc'):label(ax,l)
 title(fig,'Strict pristine controls: residual checks pass, increment convergence remains unresolved');footer(fig,'Q7 and Q8 halve all increments successively; this differs from the four scheduled post-damage-only checks.')
 save(fig,CURRENT,'Q4/Q7/Q8 pristine controls pass recorded and selected independent static residual checks, but their peak and integral sequences are nonmonotonic. Residual agreement does not establish strain-increment convergence. No extrapolated continuum-step result is claimed.',None,ids)
def secondary():
 global CURRENT
 CURRENT='11_rule_of_mixtures';fig,axs=plt.subplots(1,3,figsize=(9,3.8));fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.86,wspace=.38)
 s0=value(BY['S1_pristine_zz'],'A','strength');y0=BY['S1_pristine_zz']['modulus'];sac=value(BY['S1_pristine_ac'],'A','strength')
 for i,(ax,key) in enumerate(zip(axs[:2],['strength','modulus'])):
  style(ax);paired(ax,DISC,'rho',key);ax.set(xlabel='Relative areal density',ylabel='Maximum stress (N/m)' if key=='strength' else '2D modulus (N/m)');xx=np.array([.45,1.02]);ax.plot(xx,xx*(s0 if key=='strength' else y0),c='.3',lw=.7);ax.axvspan(.77,.83,color='.8',alpha=.15)
  if key=='strength':ax.plot(xx,xx*sac,'--',c='.5',lw=.7)
 style(axs[2]);axs[2].set_label('efficiency')
 for r in DISC:
  xx=r['modulus']/r['rho']/y0;ya=value(r,'A','strength')/r['rho']/s0;yc=value(r,'C','strength')/r['rho']/s0
  axs[2].plot([xx,xx],[ya,yc],c=FAMCOL[r['family']],lw=.65);obs(axs[2],r,'A',xx,ya,size=12)
  if r['C']:obs(axs[2],r,'C',xx,yc,size=12)
 axs[2].set(xlabel='Original modulus efficiency',ylabel='Strength efficiency');axs[2].plot([0,1.2],[0,1.2],':',c='.5',lw=.7)
 for ax,l in zip(axs,'abc'):label(ax,l)
 title(fig,'Strength-based audit: density and efficiency');footer(fig,'Same 96 designs; modulus retained; gray A / colored C; paired lines only where coordinates change.')
 save(fig,CURRENT,'Candidate A-to-C overlay for strength-density and strength efficiency; modulus remains archived because this study targets loading horizons, with early-modulus reproduction differences tabulated separately. Fixed original pristine references and density definitions.','fig1_rule_of_mixtures.pdf',[r['name'] for r in DISC])
 CURRENT='12_load_paths';fig,axs=plt.subplots(1,3,figsize=(9.6,4));fig.subplots_adjust(left=.07,right=.98,bottom=.26,top=.85,wspace=.52)
 porous=[r for r in DISC if r['porosity']>.03]
 for ax,xkey,lab in zip(axs[:2],['rho','min_section'],['Relative density','Minimum load-bearing section']):
  style(ax);paired(ax,porous,xkey,'strength');ax.set(xlabel=lab,ylabel='Maximum stress (N/m)')
 ax=axs[2];style(ax);fams=[f for f in FAMORDER if f!='pristine'];fams.sort(key=lambda f:SUM['family_best']['A'][f]['efficiency'])
 for i,f in enumerate(fams):
  for state,off,col in [('A',-.16,A_COLOR),('C',.16,FAMCOL[f])]:ax.barh(i+off,SUM['family_best'][state][f]['efficiency'],height=.28,color=col)
 ax.set_yticks(range(len(fams)));ax.set_yticklabels([FAMNAME[f] for f in fams],fontsize=6);ax.axvline(.8,c='#c0392b',ls='--',lw=.7);ax.set_xlabel('Best specific-strength efficiency')
 for ax,l in zip(axs,'abc'):label(ax,l)
 title(fig,'Load paths and family-best specific strengths');footer(fig,'Gray: A   Colored: C   The same discovery population and original pristine normalization are retained.')
 save(fig,CURRENT,'Strength-density, strength-minimum-section and family-best specific-strength comparisons. The original predictor/calibration is not retrained. Descriptive correlations and class medians are exported separately for A/B/C. Changes in primary C reflect both reproduction and loading effects.','fig2_load_paths.pdf',[r['name'] for r in porous])
 CURRENT='13_mechanism_measurements';fig,axs=plt.subplots(1,3,figsize=(9,3.8));fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.85,wspace=.4)
 classes=['straight','coarse_round','fine_round','random'];rs=[r for r in ROWS if r['campaign'] in ['discovery','paper'] and r['porosity']>.03 and r['ligament_class'] in classes and not(r['family']=='slit_array' and r['params'].get('angle_deg',0)!=0)]
 assert len(rs)==77
 for i,c in enumerate(classes):
  for r in [r for r in rs if r['ligament_class']==c]:
   ya=value(r,'A','strength')/r['min_section'];yc=value(r,'C','strength')/r['min_section'];axs[0].plot([i-.1,i+.1],[ya,yc],c='.75',lw=.4);obs(axs[0],r,'A',i-.1,ya,size=10);obs(axs[0],r,'C',i+.1,yc,size=10)
  for state,off,col in [('A',-.13,A_COLOR),('C',.13,C_COLOR)]:y=SUM['ligament_medians'][state][c]['median'];axs[0].plot([i+off-.09,i+off+.09],[y,y],c=col,lw=1.5)
 axs[0].set(xticks=range(4),xticklabels=['straight','wide','narrow','random'],ylabel='Net-section strength (N/m)',xlabel='Frozen ligament classes; n=77')
 sd=list(csv.DictReader((DATA/'slit_angle_predictions_and_responses.csv').open()));sweep=[q for q in sd if q['in_angle_sweep']=='True'];best={}
 for q in sorted(sweep,key=lambda q:abs(float(q['porosity'])-.2)):best.setdefault(float(q['theta']),q)
 for th,q in sorted(best.items()):
  r=BY[q['name']];axs[1].plot([th,th],[value(r,'A','strength'),value(r,'C','strength')],c='.7');obs(axs[1],r,'A',th,value(r,'A','strength'));obs(axs[1],r,'C',th,value(r,'C','strength'))
 axs[1].set(xlabel='Slit angle (°); original 12-angle subset',ylabel='Maximum stress (N/m)');paired(axs[2],MESH,'alignment','strength');axs[2].set(xlabel='Original alignment index A',ylabel='Maximum stress (N/m)')
 for ax,l in zip(axs,'abc'):style(ax);label(ax,l)
 title(fig,'Candidate measurement panels for the mechanism figure');footer(fig,'Measured responses are revised; schematics are not mechanistic proof. Gray A / colored C; residual flags retained.')
 save(fig,CURRENT,'Replacement options for the three measurement panels of the mechanism figure, keeping the original 77-design ligament subset, 12-angle nearest-porosity selection and fixed 40 meshes. Schematic and mechanistic claims require separate interpretation; no new mechanistic claim is inferred solely from these coordinates.','fig_mechanisms.pdf',[r['name'] for r in rs])
def predictions():
 global CURRENT
 CURRENT='14_retrospective_predictions';qq=list(csv.DictReader((DATA/'retrospective_prediction_comparison.csv').open()));fig,axs=plt.subplots(2,2,figsize=(9,7.6));fig.subplots_adjust(left=.085,right=.98,bottom=.15,top=.89,hspace=.33,wspace=.38)
 for j,(cohort,method) in enumerate([('Phase-I holdouts','registered'),('Earlier Phase-II sweeps','mechanism')]):
  q=[q for q in qq if q['cohort']==cohort and q['method']==method and q['state']=='A'];
  with (DATA/f'prediction_design_index_{j+1}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['index','name','cohort','parent_run_id']);w.writeheader();w.writerows([dict(index=i+1,name=a['name'],cohort=cohort,parent_run_id=a['parent_run_id']) for i,a in enumerate(q)])
  ax=axs[0,j];style(ax);ax.plot([0,35],[0,35],c='.5',lw=.8);ax.fill_between([0,35],[0,35*.85],[0,35*1.15],color='.8',alpha=.2)
  for a in q:
   r=BY[a['name']];pred=float(a['prediction']);ax.plot([value(r,'A','strength'),value(r,'C','strength')],[pred,pred],c='.65',lw=.7);obs(ax,r,'A',value(r,'A','strength'),pred,size=18);obs(ax,r,'C',value(r,'C','strength'),pred,size=18)
  ax.set(xlabel='Observed maximum stress (N/m)',ylabel='Frozen prediction (N/m)',title=cohort)
  ax=axs[1,j];style(ax)
  for i,a in enumerate(q):
   r=BY[a['name']];pred=float(a['prediction']);ya=100*(pred/value(r,'A','strength')-1);yc=100*(pred/value(r,'C','strength')-1);ax.plot([ya,yc],[i,i],c='.6',lw=.7);ax.scatter(ya,i,c=A_COLOR,s=15);ax.scatter(yc,i,c=C_COLOR,s=15)
  ax.axvline(0,c='.4',lw=.6);ax.axvspan(-15,15,color='.8',alpha=.2);ax.set_yticks(range(len(q)));ax.set_yticklabels([str(i+1) for i in range(len(q))],fontsize=6);ax.set_ylabel('Design index (names in exported table)');ax.set_xlabel('100 × (prediction / observed - 1) (%)')
 for ax,l in zip(axs.flat,'abcd'):label(ax,l)
 title(fig,'Retrospective evaluation only: original registration and scores remain frozen');footer(fig,'Gray: A   Family colors: C in scatter panels; orange: C errors   Predictions/calibration held fixed\nTables include B and both rules for the 18 earlier Phase-II sweeps; no new preregistered success is claimed.')
 save(fig,CURRENT,'A retrospective diagnostic, not a replacement for the historical preregistration figure or its scores. Fixed original predictions are compared with primary C observations; no predictions, file hashes, calibration or original evaluation files are modified. Full tables also include B, net-section predictions and denominators.','fig6_preregistration.pdf',[q['name'] for q in qq])
def readframe(case,branch,index):
 run=RUNS[case]
 if branch=='C':
  p=ROOT/'runs'/case/'frames'/f'{index:05d}.npz';f=dict(np.load(p));base=dict(np.load(p.parent/'00000.npz'));cell=f['cell'];coord=f['coordination'];energy=f['peratom_energy']-base['peratom_energy'];pos=f['positions'];co0=base['coordination']
 else:
  p=ROOT/'inputs/baseline/trajectories'/f"{run['parent_run_id']}.npz";f=np.load(p);cell=f['cells'][index];coord=f['coordination'][index];energy=f['peratom_energy'][index]-f['peratom_energy'][0];pos=f['positions'][index];co0=f['coordination'][0]
 return pos,cell,energy,coord<co0,p

def framepanel(ax,case,branch,index,event):
 pos,cell,de,lost,p=readframe(case,branch,index);L=np.diag(cell);x=pos[:,0]%L[0];y=pos[:,1]%L[1];ax.scatter(x,y,c='.35',s=.75,linewidths=0);ax.scatter(x,y,c=de,cmap='coolwarm',vmin=-.5,vmax=.5,s=.35,linewidths=0)
 ax.scatter(x[lost],y[lost],s=1.5,facecolors='none',edgecolors='k',linewidths=.18)
 ax.set(xlim=(0,L[0]),ylim=(0,L[1]));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 for s in ax.spines.values():s.set(color=A_COLOR if branch=='A' else C_COLOR,linewidth=.8,linestyle='--' if branch=='A' else '-')
 xcurve,ycurve,bad=getcurve(case,branch);flag=bool(bad[index]) if branch=='C' else None;ax.set_title(f"{branch}: {event}\nε={xcurve[index]:.3f}"+('  +' if flag else ''),fontsize=6,pad=2)
 out=OUT/'data/frames';out.mkdir(exist_ok=True);name=f'{case}__{branch}__{index:05d}.npz';np.savez_compressed(out/name,positions=pos,cell=cell,delta_energy_eV=de,coordination_below_initial=lost)
 FRAMEINFO.append(dict(figure=CURRENT,case_id=case,branch=branch,index=index,event=event,strain=float(xcurve[index]),stress_N_m=float(ycurve[index]),source_path=str(p.relative_to(ROOT)),source_sha256=sha(p),plotted_frame=f'data/frames/{name}',residual_flag=flag))
def progression():
 global CURRENT
 cases=['ext__S7_slit_20deg','ext__S5_H2_D40_W8_ellipseX','ext__S2_slit_45deg','ext__S7_slit_0deg_armchair','ext__S5_H2_D40_W8_jitter2']
 for page,case in enumerate(cases):
  CURRENT=f'15_progression_{page+1}';fig=plt.figure(figsize=(10,5.7));gs=fig.add_gridspec(2,6,left=.03,right=.98,bottom=.18,top=.84,height_ratios=[1,1.4],wspace=.2,hspace=.45)
  r=RUNS[case];x,y,bad=getcurve(case,'C');qa=CURVES[(case,'C')];first=next(i for i,q in enumerate(qa) if float(q['cumulative_broken'])>0);B=r['paired']['B_original_protocol_on_rerun']['endpoint_index'];end=len(x)-1;peak=int(np.argmax(y));ai=r['paired']['A_archived']['endpoint_index']
  selected=[('A',ai,'archived end')]+sorted([('C',0,'relaxed'),('C',first,'first bond loss'),('C',peak,'recorded peak'),('C',B,'original stop B'),('C',end,'extended end C')],key=lambda z:z[1])
  for k,(branch,index,event) in enumerate(selected):framepanel(fig.add_subplot(gs[0,k]),case,branch,index,event)
  ax=fig.add_axes([.08,.24,.37,.32]);curvepanel(ax,case,True);ax.legend(frameon=False,fontsize=6,loc='upper right')
  ax=fig.add_axes([.60,.24,.36,.32]);style(ax);ax.plot(x,[float(q['cumulative_broken']) for q in qa],c=C_COLOR);ax.set(xlabel='Engineering strain (rerun only)',ylabel='Cumulative broken-bond events')
  for i in [first,peak,B,end]:ax.axvline(x[i],c='.65',ls=':',lw=.5)
  cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-.5,.5),cmap='coolwarm'),cax=fig.add_axes([.40,.105,.20,.018]),orientation='horizontal');cb.set_label('Per-atom energy change (eV)',fontsize=6);cb.ax.tick_params(labelsize=6)
  title(fig,'Progression candidate: '+r['name'].split('_',1)[1]);footer(fig,'Colors: per-atom energy change from the same trajectory’s relaxed state, −0.5 to +0.5 eV (blue to red)\nCircles: coordination below its relaxed value; +: frame residual flag. Archive and rerun states are separate.')
  save(fig,CURRENT,'Saved states and curves come from their explicitly identified trajectories; A is retained as an archived observation endpoint and is never spliced into C. Colors are per-atom energy changes from each path\'s own relaxed state. Circles mark lower coordination than initially, not an independent chemical bond-healing test. Cumulative bond-loss events can remain positive after reformation. The sheet is a flat, unperturbed periodic monolayer; out-of-plane stability is untested. Full machine-readable plotted atom positions, colors and source hashes are included.','fig7_progression.pdf',[case])
 CURRENT='16_affected_architecture_cells';rs=sorted([r for r in DISC if r['C']],key=lambda r:(FAMORDER.index(r['family']),-value(r,'A','strength')));fig,axs=plt.subplots(5,4,figsize=(9,10.6));fig.subplots_adjust(left=.04,right=.98,top=.91,bottom=.08,wspace=.22,hspace=.58)
 for ax,r in zip(axs.flat,rs):
  thumbnail(ax,r);ax.set_title('\n'.join(textwrap.wrap(r['name'].split('_',1)[1],24)),fontsize=7,pad=2);ax.set_xlabel(f"σ: {value(r,'A','strength'):.2f} → {value(r,'C','strength'):.2f}\nI: {value(r,'A','integral'):.2f} → {value(r,'C','integral'):.2f}\nεend: {value(r,'A','end_strain'):.3f} → {value(r,'C','end_strain'):.3f}"+('  +' if r['residual_flag'] else ''),fontsize=6,labelpad=2)
 title(fig,'Atlas update candidates: the 20 affected discovery cells');footer(fig,'Original relaxed geometries retained; σ in N/m, I in J/m²; A → C. +: new-run residual flag.\nComplete 96-design A/B/C coordinates and family rankings are exported; the other 76 cells retain A.')
 save(fig,CURRENT,'Candidate annotations for the 20 affected cells of the original 96-design architecture atlas. Initial structures do not change. Full discovery tables retain all 96 designs and distinguish peak strain, ending strain and operational stopping. Original mechanism/mode labels should not be transferred automatically to new trajectories.','si/figS_atlas.pdf',[r['name'] for r in rs])
def main():
 for fn in [ashby,angle,alignment,disorder,lambda:disorder(True),curves,resolution,secondary,predictions,progression]:fn()
 (OUT/'figure_manifest.json').write_text(json.dumps(REG,indent=2)+'\n');(DATA/'plotted_points.json').write_text(json.dumps(POINTS,indent=2)+'\n');(DATA/'progression_frames.json').write_text(json.dumps(FRAMEINFO,indent=2)+'\n')
 print('Created',len(REG),'candidate figure pages; saved',len(POINTS),'paired plot points and',len(FRAMEINFO),'frame references.')
if __name__=='__main__':main()
