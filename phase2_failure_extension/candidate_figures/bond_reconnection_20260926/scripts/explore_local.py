from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];RUN=STUDY/'runs/ext__S7_slit_20deg'
z=np.load(RUN/'trajectory.npz');P=z['positions'].astype(float);L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2];eps=z['eps_x'];stress=z['sigma_xx']*16.0217663
states=[]
for p in sorted((RUN/'frames').glob('*.npz')):
 with np.load(p) as f:states.append({tuple(map(int,sorted(b))) for b in f['bonds']})
t0=np.array([np.cos(np.deg2rad(20)),np.sin(np.deg2rad(20))]);n0=np.array([-t0[1],t0[0]])
target=1855;old=1928;new=2008
center=(P[0,target,:2]+P[0,old,:2])/2
ref=P[0,:,:2]-center
t=(ref*t0).sum(1);n=(ref*n0).sum(1)
upper=np.where((abs(t)<7)&(n>2.5)&(n<7.5))[0]
lower=np.where((abs(t)<7)&(n<-2.5)&(n>-7.5))[0]
roi=np.where((abs(t)<15)&(abs(n)<11))[0]
def mic(d,k):return d-np.round(d/L[k])*L[k]
def dist(i,j,k):return np.linalg.norm(mic(P[k,j,:2]-P[k,i,:2],k))
# Fixed material patches; subtract homogeneous deformation, then project onto
# the affinely transported original slit direction. Report relative motion,
# not a crystallographic slip system or an atomically sharp slip discontinuity.
du=[];dn=[];patchdiff=[]
for k in range(len(eps)):
 F=L[k]/L[0]
 u=P[k,:,:2]-P[0,:,:2]*F
 u=mic(u,k)
 dd=u[upper].mean(0)-u[lower].mean(0);tk=t0*F;tk/=np.linalg.norm(tk);nk=np.array([-tk[1],tk[0]])
 du.append(dd@tk);dn.append(dd@nk);patchdiff.append(dd)
print('patches',len(upper),len(lower),'IDs',upper.tolist(),lower.tolist(),'center',center)
for k in [0,34,35,36,45,51,53,54,55,60,65]:
 print(k,round(eps[k],4),'stress',round(stress[k],3),'old/new d',round(dist(target,old,k),3),round(dist(target,new,k),3),'slip',round(du[k],3),'close',round(dn[k],3),'coord',[sum(i in p for p in states[k]) for i in [target,old,new]])
fig,axs=plt.subplots(3,3,figsize=(12,10))
plt.rcParams['font.family']='Arial'
for ax,k in zip(axs.flat,[34,35,36,45,51,53,54,55,60]):
 ctr=P[k,roi,:2].mean(0)
 seg=[];colors=[];width=[]
 for i,j in states[k]:
  d=mic(P[k,j,:2]-P[k,i,:2],k)
  if np.max(abs(P[k,j,:2]-P[k,i,:2]))>10:continue
  seg.append([P[k,i,:2]-ctr,P[k,i,:2]+d-ctr]);colors.append('#e87513' if (i,j) not in states[0] else '#8b939b');width.append(2 if (i,j) not in states[0] else .65)
 ax.add_collection(LineCollection(seg,colors=colors,linewidths=width))
 for i,col in [(target,'#174c84'),(old,'#995fba'),(new,'#00a58f')]:
  xy=P[k,i,:2]-ctr;ax.scatter(*xy,s=28,c=col,zorder=5);ax.annotate(str(i),xy,xytext=(3,4),textcoords='offset points',fontsize=9)
 ax.scatter(*(P[k,upper,:2]-ctr).T,s=5,c='#174c84',alpha=.5);ax.scatter(*(P[k,lower,:2]-ctr).T,s=5,c='#00a58f',alpha=.5)
 ax.set(xlim=(-17,17),ylim=(-11,11),aspect='equal',title=f'{k}: strain {eps[k]:.3f}');ax.axis('off')
fig.tight_layout();fig.savefig(ROOT/'qa/local_exploration.png',dpi=180);plt.close(fig)
data=dict(target=target,old_partner=old,new_partner=new,center_reference=center.tolist(),upper_patch_ids=upper.tolist(),lower_patch_ids=lower.tolist(),roi_ids=roi.tolist(),strain=eps.tolist(),relative_tangent_nonaffine_A=du,relative_normal_nonaffine_A=dn,pair_old_distance_A=[dist(target,old,k) for k in range(len(eps))],pair_new_distance_A=[dist(target,new,k) for k in range(len(eps))])
(ROOT/'data/local_selection.json').write_text(json.dumps(data,indent=2)+'\n')
