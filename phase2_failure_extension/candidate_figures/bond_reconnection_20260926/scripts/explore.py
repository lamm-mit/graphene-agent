"""Inspect saved 20-degree slit bonds and candidate reconnection sites; no simulations."""
from pathlib import Path
import json,csv
from collections import deque
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];RUN=STUDY/'runs/ext__S7_slit_20deg'
T=np.load(RUN/'trajectory.npz');P=T['positions'].astype(float);C=T['cells'];L=np.diagonal(C,axis1=1,axis2=2)[:,:2];eps=T['eps_x'];N=P.shape[1]
states=[]
for f in sorted((RUN/'frames').glob('*.npz')):
    with np.load(f) as z:states.append({tuple(map(int,sorted(b))) for b in z['bonds']})
def mic(d,k):return d-np.round(d/L[k])*L[k]
def distance(pair,k):return float(np.linalg.norm(mic(P[k,pair[1],:2]-P[k,pair[0],:2],k)))
def midpoint(pair,k):return (P[k,pair[0],:2]+.5*mic(P[k,pair[1],:2]-P[k,pair[0],:2],k))%L[k]
def adjacency(k):
    g=[set() for i in range(N)]
    for a,b in states[k]:g[a].add(b);g[b].add(a)
    return g
G=[adjacency(k) for k in range(len(states))]
def ring(pair,k,limit=12):
    a,b=pair;q=deque([(a,0)]);seen={a}
    while q:
        u,d=q.popleft()
        if d>=limit-1:continue
        for v in G[k][u]:
            if (u==a and v==b) or (u==b and v==a):continue
            if v==b:return d+2
            if v not in seen:seen.add(v);q.append((v,d+1))
    return None
seen=set(states[0]);events=[]
for k in range(1,len(states)):
    for pair in sorted(states[k]-states[k-1]):
        first=pair not in seen
        stop=k
        while stop+1<len(states) and pair in states[stop+1]:stop+=1
        row=dict(frame=k,strain=float(eps[k]),i=pair[0],j=pair[1],first=first,initial_distance=distance(pair,0),current_distance=distance(pair,k),midpoint=midpoint(pair,k).tolist(),midpoint_initial=midpoint(pair,0).tolist(),degrees=[len(G[k][u]) for u in pair],continuous_until=stop,until_strain=float(eps[stop]),smallest_cycle=ring(pair,k))
        events.append(row)
    seen|=states[k]
(ROOT/'data/formation_events.json').write_text(json.dumps(events,indent=2)+'\n')
focus=[r for r in events if r['first'] and 0.08<r['strain']<.3 and r['initial_distance']>4 and r['continuous_until']>=r['frame']+5 and 30<r['midpoint_initial'][0]<75 and 25<r['midpoint_initial'][1]<75]
print(json.dumps(focus,indent=2))
plt.rcParams.update({'font.family':'Arial','font.size':9})
fig,axs=plt.subplots(2,3,figsize=(12,8))
for ax,k in zip(axs.flat,[0,16,24,35,51,75]):
    seg=[];col=[];lw=[]
    for i,j in states[k]:
        d=mic(P[k,j,:2]-P[k,i,:2],k)
        if np.max(np.abs(P[k,j,:2]-P[k,i,:2]))>10:continue
        seg.append([P[k,i,:2],P[k,i,:2]+d]);col.append('#ef6c00' if (i,j) not in states[0] else '#8c9399');lw.append(1.8 if (i,j) not in states[0] else .35)
    ax.add_collection(LineCollection(seg,colors=col,linewidths=lw))
    ax.set(xlim=(0,L[k,0]),ylim=(0,L[k,1]),aspect='equal',title=f'frame {k}, strain {eps[k]:.3f}')
    for r in focus:
        if r['frame']==k:
            xy=midpoint((r['i'],r['j']),k);ax.text(*xy,str(r['i']),fontsize=6)
fig.tight_layout();fig.savefig(ROOT/'qa/network_exploration.png',dpi=170);plt.close(fig)
summary=[]
for k in [0,16,24,35,51,75,160]:
    new=states[k]-states[0];degree=np.array([len(x) for x in G[k]])
    summary.append(dict(frame=k,strain=float(eps[k]),new_bonds=len(new),new_bond_smallest_cycles={str(s):sum(ring(pair,k)==s for pair in new) for s in [3,4,5,6,7,8,9,10,11,12,None]},coordination={str(i):int((degree==i).sum()) for i in range(6)}))
(ROOT/'data/topology_overview.json').write_text(json.dumps(summary,indent=2)+'\n')

