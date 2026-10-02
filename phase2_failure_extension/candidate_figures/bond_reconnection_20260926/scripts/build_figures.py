# %SEPT 26 NEW MECHANISMS: definitions and qualifications moved to captions.
"""Build clean scientific views from saved atom positions/bond graphs; no new simulations."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];RUN=STUDY/'runs/ext__S7_slit_20deg';F=ROOT/'figures';D=ROOT/'data';Q=ROOT/'qa'
(ROOT / 'figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'qa').mkdir(parents=True, exist_ok=True)
(ROOT / 'data').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'axes.linewidth':.6,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
z=np.load(RUN/'trajectory.npz');P=z['positions'].astype(float);L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2];eps=z['eps_x'];stress=z['sigma_xx']*16.0217663
S=[]
for fp in sorted((RUN/'frames').glob('*.npz')):
 with np.load(fp) as zz:S.append({tuple(map(int,sorted(v))) for v in zz['bonds']})
local=json.loads((D/'local_selection.json').read_text());roi=np.array(local['roi_ids']);up=np.array(local['upper_patch_ids']);down=np.array(local['lower_patch_ids'])
p,q,r=local['target'],local['old_partner'],local['new_partner']
colors={'base':'#8d969e','new':'#df810f','p':'#252e38','q':'#2377b3','r':'#8856a7','blue':'#3e7499','teal':'#278d88'}
t0=np.array([np.cos(np.deg2rad(20)),np.sin(np.deg2rad(20))])
def mic(v,k):return v-np.round(v/L[k])*L[k]
def center(k):return P[k,roi,:2].mean(0)
def line_data(k,ctr,window):
 seg=[];cols=[];width=[]
 for i,j in S[k]:
  a=P[k,i,:2];b=a+mic(P[k,j,:2]-a,k)
  shifts=[(0,0)]
  if ctr is None:shifts=[(x,y) for x in [-L[k,0],0,L[k,0]] for y in [-L[k,1],0,L[k,1]]]
  for shift in shifts:
   aa=a+shift-(ctr if ctr is not None else 0);bb=b+shift-(ctr if ctr is not None else 0)
   if max(aa[0],bb[0])<window[0] or min(aa[0],bb[0])>window[1] or max(aa[1],bb[1])<window[2] or min(aa[1],bb[1])>window[3]:continue
   fresh=(i,j) not in S[0]
   seg.append([aa,bb]);cols.append(colors['new'] if fresh else colors['base']);width.append((1.7 if fresh else .75) if ctr is not None else (.9 if fresh else .25))
 return seg,cols,width
def lab(ax,letter,subtitle=None):
 ax.text(-.02,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')
 if subtitle:ax.text(.055,1.10,subtitle,transform=ax.transAxes,fontsize=9,va='bottom')
def frame(ax,k,arrows=False,guide=True,identities=True):
 ctr=center(k);window=(-12.5,12.5,-8.3,8.3)
 seg,col,lw=line_data(k,ctr,window);ax.add_collection(LineCollection(seg,colors=col,linewidths=lw,zorder=2))
 # An interface guide, defined by the midpoints of three tracked opposing-bank pairs.
 mids=[]
 for i,j in [(1788,1927),(1855,1928),(2006,2086)]:
  mids.append(P[k,i,:2]+.5*mic(P[k,j,:2]-P[k,i,:2],k)-ctr)
 mids=np.array(mids);v=mids[-1]-mids[0];v/=np.linalg.norm(v);g=mids.mean(0)
 if guide:
  vv=np.array([g-v*9,g+v*9]);ax.plot(vv[:,0],vv[:,1],ls=(0,(3,3)),color='#59616a',lw=.65,alpha=.65,zorder=1)
 for atom,key,off in [(p,'p',(1,-13)),(q,'q',(-12,6)),(r,'r',(7,5))]:
  xy=P[k,atom,:2]-ctr;ax.scatter(*xy,s=26,color=colors[key],edgecolor='white',linewidth=.5,zorder=5)
  if identities:ax.annotate(key,xy,xytext=off,textcoords='offset points',color=colors[key],fontweight='bold',fontsize=9,zorder=6,bbox=dict(facecolor='white',edgecolor='none',pad=.35,alpha=.85))
 if arrows:
  tk=t0*(L[k]/L[0]);tk/=np.linalg.norm(tk)
  for y,sgn in [(6.4,-1),(-5.7,1)]:
   a=np.array([3.0,y]);b=a+sgn*tk*2.7
   ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.1,color='#2c343b',mutation_scale=10),zorder=7)
  slip=abs(local['relative_tangent_nonaffine_A'][k]-local['relative_tangent_nonaffine_A'][35])
  ax.text(.98,.03,f'relative sliding ≈ {slip:.1f} Å',transform=ax.transAxes,ha='right',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=1.5,alpha=.9))
 ax.set(xlim=window[:2],ylim=window[2:],aspect='equal');ax.axis('off')
 ax.plot([-11,-6],[-7.2,-7.2],color='#303841',lw=1.5);ax.text(-8.5,-6.9,'5 Å',ha='center',va='bottom',fontsize=7)
 return ctr
fig=plt.figure(figsize=(9.6,6.65))
ax=fig.add_axes([.025,.545,.315,.38]);k=51
seg,col,lw=line_data(k,None,(0,L[k,0],0,L[k,1]));ax.add_collection(LineCollection(seg,colors=col,linewidths=lw));ax.add_patch(Rectangle((0,0),*L[k],fill=False,lw=.65,edgecolor='#bec4c9'))
ctr=center(k);ax.add_patch(Rectangle(ctr+[-12.5,-8.3],25,16.6,fill=False,lw=1.4,edgecolor='#3e7499'))
ax.annotate('tracked interface',xy=ctr+[0,-8.3],xytext=(L[k,0]*.58,-12),ha='center',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.7,color='#3e7499'),color='#3e7499')
ax.annotate('',xy=(40,L[k,1]+8),xytext=(15,L[k,1]+8),arrowprops=dict(arrowstyle='->',lw=1));ax.text(43,L[k,1]+8,'loading x',va='center',fontsize=8)
ax.plot([4,24],[-6,-6],color='black',lw=1.3);ax.text(14,-5,'2 nm',ha='center',fontsize=7)
ax.set(xlim=(-3,L[k,0]+3),ylim=(-16,L[k,1]+16),aspect='equal');ax.axis('off');lab(ax,'a',r'$\epsilon=0.275$')
ax=fig.add_axes([.405,.61,.235,.28]);lab(ax,'b')
ax.plot(eps,stress,color='#26333f',lw=1.1)
ax.axvline(eps[35],color='#929aa1',ls=(0,(3,3)),lw=.8)
for k,letter,off in [(35,'d',(-15,-8)),(51,'e',(-15,9)),(54,'f',(13,0))]:
 ax.scatter(eps[k],stress[k],s=24,facecolor=colors['new'],edgecolor='white',lw=.5,zorder=4);ax.annotate(letter,(eps[k],stress[k]),xytext=off,textcoords='offset points',fontsize=9,fontweight='bold',arrowprops=dict(arrowstyle='-',lw=.6,color='#555'))
ax.text(.28,.04,'original stop',transform=ax.transAxes,color='#757d84',fontsize=7,rotation=90,va='bottom')
ax.set(xlabel='engineering strain',ylabel='2D stress (N/m)',xlim=(0,.84),ylim=(0,17));ax.set_xticks([0,.2,.4,.6,.8]);ax.set_yticks([0,5,10,15]);ax.tick_params(length=3)
ax=fig.add_axes([.735,.61,.235,.28]);lab(ax,'c')
a=np.array(local['pair_old_distance_A']);b=np.array(local['pair_new_distance_A'])
ax.axhspan(0,2,color='#e7ecef',alpha=.65,zorder=0);ax.axvspan(eps[53],eps[54],color=colors['new'],alpha=.13,lw=0)
ax.plot(eps,a,color=colors['q'],lw=1.5,label='p–q');ax.plot(eps,b,color=colors['r'],lw=1.5,label='p–r')
ax.axhline(2,color='#636c75',lw=.65,ls=(0,(3,3)))
ax.text(.322,2.07,'2 Å criterion',ha='right',fontsize=7,color='#636c75')
ax.set(xlim=(.17,.325),ylim=(1,5),xlabel='engineering strain',ylabel='pair separation (Å)');ax.set_xticks([.18,.22,.26,.30]);ax.legend(frameon=False,loc='upper right',fontsize=8,handlelength=1.5);ax.tick_params(length=3)
for x,k,letter,title in [(.025,35,'d','Before this local reconnection'),(.357,51,'e','Connected across the interface'),(.689,54,'f','Reconnected to a new neighbor')]:
 ax=fig.add_axes([x,.135,.286,.365]);frame(ax,k,arrows=k==54)
 lab(ax,letter,rf'$\epsilon={eps[k]:.3f}$')
handles=[Line2D([0],[0],color=colors['base'],lw=1.2,label='initial connections'),Line2D([0],[0],color=colors['new'],lw=1.8,label='new connections'),Line2D([0],[0],color='#59616a',lw=.7,ls=(0,(3,3)),label='interface guide')]

for ext in ['png','svg','pdf']:fig.savefig(F/f'bond_reconnection_20deg.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
# Angle comparison: first-appearance persistence is a retrospective geometric diagnostic.
R=json.loads((D/'angle_comparison.json').read_text());angles=np.array([v['angle'] for v in R])
with (D/'angle_bond_curves.csv').open() as f:curves=list(csv.DictReader(f))
fig2=plt.figure(figsize=(9.6,4.25))
ax=fig2.add_axes([.065,.20,.24,.61]);lab(ax,'a')
y=np.array([v['first_persistent_new_strain'] for v in R])
ax.plot(angles,y,lw=.8,color='#adb7bf',zorder=1)
for v in R:
 marker='s' if v['angle']==0 else 'o';c=colors['new'] if v['angle']==20 else colors['blue']
 ax.scatter(v['angle'],v['first_persistent_new_strain'],s=38 if v['angle']==20 else 25,marker=marker,facecolor='white' if v['porosity']<.19 else c,edgecolor=c,linewidth=1,zorder=3)

ax.set(xlim=(-4,94),ylim=(.06,.22),xlabel='slit angle (degrees)',ylabel='onset strain');ax.set_xticks([0,20,40,60,90]);ax.tick_params(length=3)
ax=fig2.add_axes([.405,.20,.24,.61]);lab(ax,'b')
y=np.array([v['persistent_new_per1000_at_0.200'] for v in R]);ax.plot(angles,y,lw=.8,color='#adb7bf',zorder=1)
for v in R:
 c=colors['new'] if v['angle']==20 else colors['blue'];marker='s' if v['angle']==0 else 'o'
 ax.scatter(v['angle'],v['persistent_new_per1000_at_0.200'],s=38 if v['angle']==20 else 25,marker=marker,facecolor='white' if v['porosity']<.19 else c,edgecolor=c,linewidth=1,zorder=3)
ax.set(xlim=(-4,94),ylim=(-1,40),xlabel='slit angle (degrees)',ylabel='persistent new pairs / 1,000 C');ax.set_xticks([0,20,40,60,90]);ax.tick_params(length=3)
ax=fig2.add_axes([.748,.20,.173,.61]);lab(ax,'c')
grid=np.linspace(0,.20,161);Z=[]
for v in R:
 rr=[c for c in curves if c['name']==v['name']];es=np.array([float(c['strain']) for c in rr]);yy=np.array([float(c['persistent_unique_new_pairs']) for c in rr])*1000/v['n_atoms']
 ix=np.maximum(np.searchsorted(es,grid+1e-10,side='right')-1,0);Z.append(yy[ix])
im=ax.imshow(Z,origin='lower',aspect='auto',extent=(0,.2,-.5,11.5),cmap='YlGnBu',vmin=0,vmax=40,interpolation='nearest')
ax.set_yticks(np.arange(12));ax.set_yticklabels([str(v) for v in angles]);ax.set(xlabel='engineering strain',ylabel='slit angle (degrees)');ax.set_xticks([0,.1,.2]);ax.tick_params(length=0)
cax=fig2.add_axes([.934,.20,.011,.61]);cb=fig2.colorbar(im,cax=cax);cb.set_label('persistent pairs / 1,000 C',fontsize=8,labelpad=3);cb.ax.tick_params(labelsize=7,length=2)

for ext in ['png','svg','pdf']:fig2.savefig(F/f'angle_dependence.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
with PdfPages(ROOT/'bond_reconnection_review.pdf',metadata={'Title':'Bond reconnection and angle dependence: standalone figure candidates','Author':'Markus J. Buehler','Subject':'Read-only analysis of saved Phase 2 loading-extension trajectories'}) as pdf:
 pdf.savefig(fig,bbox_inches='tight',pad_inches=.06);pdf.savefig(fig2,bbox_inches='tight',pad_inches=.06)
plt.close(fig);plt.close(fig2)
# Animation is made only from saved quasistatic configurations, with no interpolation.
images=[];fa=plt.figure(figsize=(5.4,4.0));ax=fa.add_axes([.04,.04,.92,.86])
frames=list(range(28,64))
for k in frames:
 ax.clear();frame(ax,k,arrows=k>=54)
 ax.set_title(rf'$\epsilon={eps[k]:.3f}$',fontfamily='Arial',fontsize=11,pad=9)
 fa.texts.clear()
 fa.canvas.draw();arr=np.asarray(fa.canvas.buffer_rgba())[:,:,:3].copy()
 images.append(Image.fromarray(arr))
durations=[160]*len(images);durations[0]=850;durations[-1]=1100
images[0].save(F/'sliding_reconnection.gif',save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False,disposal=2);plt.close(fa)
# Reproducibility: all plotted series and selected node coordinates.
with (D/'local_pair_and_sliding_curves.csv').open('w') as f:
 keys=['frame','strain','stress_N_m','p_q_distance_A','p_r_distance_A','relative_tangent_nonaffine_A','force_residual_eV_A','transverse_stress_N_m'];w=csv.DictWriter(f,keys);w.writeheader()
 for k in range(len(eps)):
  w.writerow(dict(zip(keys,[k,eps[k],stress[k],a[k],b[k],local['relative_tangent_nonaffine_A'][k],z['max_force_eV_A'][k],z['sigma_yy'][k]*16.0217663])))
selected=[]
for k in [35,51,54]:
 selected.append(dict(frame=k,strain=float(eps[k]),center=center(k).tolist(),p=P[k,p,:2].tolist(),q=P[k,q,:2].tolist(),r=P[k,r,:2].tolist(),p_q_bond=tuple(sorted([p,q])) in S[k],p_r_bond=tuple(sorted([p,r])) in S[k],new_pairs_present=len(S[k]-S[0])))
(D/'selected_frames.json').write_text(json.dumps(selected,indent=2)+'\n')
print('Created two figure pages, editable SVGs, PNG previews and the saved-state GIF.')
