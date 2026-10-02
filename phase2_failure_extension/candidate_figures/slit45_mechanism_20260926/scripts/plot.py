# %SEPT 26 NEW MECHANISMS: five saved geometry states, including the state nearest strain 0.30.
"""Figure 10: saved Phase-2 states, connected lattice details and loading curves."""
from pathlib import Path
import argparse
import json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];RUN=STUDY/'runs/ext__S2_slit_45deg';EV=16.0217663
(ROOT / 'figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'data').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.linewidth':.6,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
parser=argparse.ArgumentParser();parser.add_argument('--include-phase1',action='store_true');args=parser.parse_args()
z=np.load(RUN/'trajectory.npz');P=z['positions'].astype(float);L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2]
S=[]
for f in sorted((RUN/'frames').glob('*.npz')):
 with np.load(f) as q:S.append({tuple(sorted(map(int,b))) for b in q['bonds']})
with (ROOT/'data/curves.csv').open() as f:curves=list(csv.DictReader(f))
branches=['Phase 2 primary extension','Phase 2 resolution check'];cols=['#287ba5','#58a495'];styles=['-','--']
if args.include_phase1:
 branches.insert(0,'Phase 1 single shot');cols.insert(0,'#313b45');styles.insert(0,'-')
stem='slit45_comparison' if args.include_phase1 else 'slit45_phase2'
# %SEPT 26 NEW MECHANISMS: equally spaced geometry/zoom pairs, without extra labels.
fig=plt.figure(figsize=(9.8,6.8));sel=[];zoom_rows=[];zoom_atoms=[];zoom_edges=[]
frames=[0,17,36,44,116]
# %SEPT 26 NEW MECHANISMS: panel-specific windows identify each local mechanism.
# a/b center the original narrow bridge; c/d center new connections at the peak;
# e retains the late stretched ligament. The caption distinguishes these regions.
tracked_pairs=[(2394,2395),(2394,2395),(2156,2160),(2156,2160),(2540,2614)]
half=np.array([13.,10.]);detail_colour='#397ba0'
# All full-cell views share one physical scale; all blow-ups share a second scale.
full_xlim=(-.5,max(L[k,0] for k in frames)/10+.5)
full_ylim=(-1.4,max(L[k,1] for k in frames)/10+.3)
for j,k in enumerate(frames):
 left=.03+j*.193;wid=.174
 ax=fig.add_axes([left,.74,wid,.205]);segments=[];colors=[]
 for i,l in sorted(S[k]):
  a=P[k,i,:2];v=P[k,l,:2]-a;v-=np.round(v/L[k])*L[k];b=a+v
  for sx in [-1,0,1]:
   for sy in [-1,0,1]:
    v1=a+[sx*L[k,0],sy*L[k,1]];v2=b+[sx*L[k,0],sy*L[k,1]]
    if max(v1[0],v2[0])<0 or min(v1[0],v2[0])>L[k,0] or max(v1[1],v2[1])<0 or min(v1[1],v2[1])>L[k,1]:continue
    segments.append([v1/10,v2/10]);colors.append('#df810f' if (i,l) not in S[0] else '#818d97')
 box=Rectangle((0,0),*(L[k]/10),fill=False,edgecolor='#bac2c9',lw=.6);ax.add_patch(box)
 lc=LineCollection(segments,colors=colors,linewidths=[.6 if c=='#df810f' else .24 for c in colors]);ax.add_collection(lc);lc.set_clip_path(box)
 shift=(L[k,0]-max(L[f,0] for f in frames))/20
 ax.set(xlim=(full_xlim[0]+shift,full_xlim[1]+shift),ylim=full_ylim,aspect='equal');ax.set_anchor('S');ax.axis('off')
 fig.text(left,.924,chr(97+j),fontweight='bold',fontsize=12)
 fig.text(left+.026,.924,rf'$\epsilon={z["eps_x"][k]:.3f}$',fontsize=8.5)
 # Scale labels stay below their bars.
 ax.plot([7.5,9.5],[-.5,-.5],c='#313b45',lw=1.3);ax.text(8.5,-.74,'2 nm',fontsize=7,ha='center',va='top')
 if k==0:
  ax.annotate('',xy=(5.2,-.5),xytext=(3.5,-.5),arrowprops=dict(arrowstyle='->',lw=.8));ax.text(5.5,-.5,'x',va='center',fontsize=8)
 # Atom identities locate each explicitly marked region without modifying coordinates.
 tracked_pair=tracked_pairs[j]
 i,l=tracked_pair;d=P[k,l,:2]-P[k,i,:2];d-=np.round(d/L[k])*L[k]
 centre=(P[k,i,:2]+.5*d)%L[k];lo=centre-half;hi=centre+half
 assert np.all(lo>0) and np.all(hi<L[k]),'Selected detail must lie within this periodic cell.'
 roi=Rectangle(lo/10,*(2*half/10),fill=False,edgecolor=detail_colour,lw=.85,zorder=4);ax.add_patch(roi)
 zoom=fig.add_axes([left,.43,wid,.24]);seg=[];col=[]
 for i,l in sorted(S[k]):
  a=P[k,i,:2]-centre;a-=np.round(a/L[k])*L[k]
  d=P[k,l,:2]-P[k,i,:2];d-=np.round(d/L[k])*L[k];b=a+d
  if np.any(np.maximum(a,b)<-half) or np.any(np.minimum(a,b)>half):continue
  seg.append([a/10,b/10]);newpair=(i,l) not in S[0];col.append('#df810f' if newpair else '#818d97')
  zoom_edges.append(dict(panel=chr(97+j),frame=k,atom_i=i,atom_j=l,new_pair=newpair,x1_nm=float(a[0]/10),y1_nm=float(a[1]/10),x2_nm=float(b[0]/10),y2_nm=float(b[1]/10)))
 zb=Rectangle(-half/10,*(2*half/10),fill=False,edgecolor=detail_colour,lw=.85);zoom.add_patch(zb)
 line=LineCollection(seg,colors=col,linewidths=[1.0 if c=='#df810f' else .58 for c in col]);zoom.add_collection(line);line.set_clip_path(zb)
 pts=P[k,:,:2]-centre;pts-=np.round(pts/L[k])*L[k];inside=np.all(np.abs(pts)<=half,axis=1)
 zoom.scatter(*pts[inside].T/10,s=3.3,c='#303b43',linewidths=0,zorder=3)
 for atom in np.flatnonzero(inside):zoom_atoms.append(dict(panel=chr(97+j),frame=k,atom_id=int(atom),x_nm=float(pts[atom,0]/10),y_nm=float(pts[atom,1]/10)))
 zoom.set(xlim=(-1.34,1.34),ylim=(-1.31,1.03),aspect='equal');zoom.set_anchor('N');zoom.axis('off')
 zoom.plot([-1.20,-.70],[-1.10,-1.10],color='#313b45',lw=1.3);zoom.text(-.95,-1.18,'0.5 nm',fontsize=7,ha='center',va='top')
 for source_x,target_x in [(lo[0]/10,-half[0]/10),(hi[0]/10,half[0]/10)]:
  connector=ConnectionPatch(xyA=(source_x,lo[1]/10),coordsA=ax.transData,xyB=(target_x,half[1]/10),coordsB=zoom.transData,color=detail_colour,lw=.55,alpha=.70,zorder=.5,clip_on=False)
  fig.add_artist(connector)
 zoom_rows.append(dict(panel=chr(97+j),frame=k,tracked_pair=list(tracked_pair),centre_A=centre.tolist(),window_A=(2*half).tolist(),atom_count=int(inside.sum())))
 sel.append(dict(frame=k,strain=float(z['eps_x'][k]),new_pairs_present=len(S[k]-S[0]),transverse_strain=float(z['eps_y'][k]),max_force_eV_A=float(z['max_force_eV_A'][k]),transverse_stress_N_m=float(z['sigma_yy'][k]*EV)))
ax=fig.add_axes([.078,.07,.375,.245]);ax.text(-.02,1.07,'f',transform=ax.transAxes,fontweight='bold',fontsize=12)
for label,c,ls in zip(branches,cols,styles):
 rr=[r for r in curves if r['branch']==label];x=[float(r['strain']) for r in rr];y=[float(r['stress_N_m']) for r in rr]
 ax.plot(x,y,c=c,ls=ls,lw=1.2,label={'Phase 2 primary extension':'Phase 2 longer loading','Phase 2 resolution check':'Phase 2, half strain step'}.get(label,label))
 if label=='Phase 1 single shot':ax.scatter(x[-1],y[-1],s=22,facecolors='white',edgecolors=c,zorder=4)
# Phase-1 endpoint interpretation is in the comparison caption, not the plot.
ax.set(xlim=(0,.70),ylim=(0,20 if args.include_phase1 else 16),xlabel='engineering strain',ylabel='2D stress (N/m)');ax.set_xticks([0,.2,.4,.6]);ax.set_yticks(np.arange(0,21 if args.include_phase1 else 17,2));ax.legend(frameon=False,fontsize=7.5,loc='upper left',handlelength=1.8)
ax=fig.add_axes([.59,.07,.375,.245]);ax.text(-.02,1.07,'g',transform=ax.transAxes,fontweight='bold',fontsize=12)
for label,c,ls in zip(branches,cols,styles):
 rr=[r for r in curves if r['branch']==label];x=[float(r['strain']) for r in rr];y=[int(r['distinct_new_pairs']) for r in rr]
 ax.step(x,y,where='post',c=c,ls=ls,lw=1.2)
 ax.scatter(x[-1],y[-1],s=18,facecolors='white' if label=='Phase 1 single shot' else c,edgecolors=c,zorder=4)
 ax.annotate(str(y[-1]),(x[-1],y[-1]),xytext=(3,5),textcoords='offset points',fontsize=8,color=c)
ax.set(xlim=(0,.70),ylim=(0,210 if args.include_phase1 else 160),xlabel='engineering strain',ylabel='cumulative distinct new pairs');ax.set_xticks([0,.2,.4,.6]);ax.set_yticks(np.arange(0,201 if args.include_phase1 else 161,20))
for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'figures/{stem}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
(ROOT/'data/selected_frames.json').write_text(json.dumps(sel,indent=2)+'\n')
(ROOT/'data/connected_zoom_selection.json').write_text(json.dumps(zoom_rows,indent=2)+'\n')
for name,rows in [('connected_zoom_atoms.csv',zoom_atoms),('connected_zoom_connections.csv',zoom_edges)]:
 with (ROOT/'data'/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Built '+stem+' PDF/SVG/PNG with five connected lattice details')
