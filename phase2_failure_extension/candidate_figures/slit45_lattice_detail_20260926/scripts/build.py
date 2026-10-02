"""Magnified saved lattice states for Figure 10; no new simulation or manuscript edit."""
from pathlib import Path
from collections import Counter,deque
import csv,json,hashlib
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1]
(ROOT / 'figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'data').mkdir(parents=True, exist_ok=True)
RUN=STUDY/'runs/ext__S2_slit_45deg'
z=np.load(RUN/'trajectory.npz');P=z['positions'].astype(float)[:,:,:2];L=np.diagonal(z['cells'],axis1=1,axis2=2)[:,:2]
frames=[0,36,44];S={};G={};N=P.shape[1]
for k in frames:
    with np.load(RUN/'frames'/f'{k:05d}.npz') as f:S[k]={tuple(sorted(map(int,b))) for b in f['bonds']}
    assert S[k]=={tuple(map(int,b)) for b in cKDTree(P[k]%L[k],boxsize=L[k]).query_pairs(2,output_type='ndarray')}
    G[k]=[set() for _ in range(N)]
    for a,b in S[k]:G[k][a].add(b);G[k][b].add(a)
def mic(v,k):return v-np.round(v/L[k])*L[k]
def midpoint(pair,k):
    a,b=pair;return (P[k,a]+.5*mic(P[k,b]-P[k,a],k))%L[k]
def cycle(pair,k,limit=24):
    a,b=pair;queue=deque([a]);prev={a:None};depth={a:0}
    while queue:
        u=queue.popleft()
        if depth[u]>=limit:continue
        for v in sorted(G[k][u]):
            if tuple(sorted((u,v)))==pair:continue
            if v in prev:continue
            prev[v]=u;depth[v]=depth[u]+1
            if v==b:
                path=[b]
                while path[-1]!=a:path.append(prev[path[-1]])
                return path[::-1]
            queue.append(v)
    return None
# Deterministic field of view: new pair nearest the periodic cell centre at the peak.
peak=int(np.argmax(z['sigma_xx']));assert peak==36
pair=min(sorted(S[peak]-S[0]),key=lambda e:np.linalg.norm((midpoint(e,peak)-L[peak]/2)/L[peak]))
centres={k:midpoint(pair,k) for k in frames}
# Same physical-size window at all strains, translated with the same atom identities.
half=np.array([13.,10.])
summary=[];cycle_rows=[]
for k in frames:
    c=Counter();deg=Counter(map(len,G[k]))
    for edge in sorted(S[k]-S[0]):
        path=cycle(edge,k);size=len(path) if path else None;c[str(size) if size else 'no_loop_within_search']=c.get(str(size) if size else 'no_loop_within_search',0)+1
        cycle_rows.append({'frame':k,'atom_i':edge[0],'atom_j':edge[1],'shortest_cycle_size':size,'cycle_atom_ids':path})
    summary.append({'frame':k,'strain':float(z['eps_x'][k]),'stress_N_m':float(z['sigma_xx'][k]*16.0217663),'new_pairs_present':len(S[k]-S[0]),'coordination_counts':dict(deg),'new_pair_shortest_cycle_counts':dict(c),'max_force_eV_A':float(z['max_force_eV_A'][k]),'transverse_stress_N_m':float(z['sigma_yy'][k]*16.0217663),'spans_x':bool(z['spanning'][k])})
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
GRAY='#88939c';ORANGE='#df810f';ATOM='#303b43';BOX='#397ba0'
fig=plt.figure(figsize=(10.5,7.7))
axs=[fig.add_axes(r) for r in [[.045,.565,.415,.335],[.545,.565,.415,.335],[.045,.085,.415,.335],[.545,.085,.415,.335]]]
exports=[];annotations=[]
def draw(ax,k,centre=None):
    iszoom=centre is not None
    panel=({0:'B',36:'C',44:'D'}[k] if iszoom else 'A')
    lo=-half if iszoom else np.array([0.,0.]);hi=half if iszoom else L[k]
    origin=centre if iszoom else np.array([0.,0.])
    seg=[];cols=[];edges_out=[]
    for a,b in sorted(S[k]):
        if iszoom:
            v1=mic(P[k,a]-origin,k);v2=v1+mic(P[k,b]-P[k,a],k)
            images=[(v1,v2)]
        else:
            v1=P[k,a]%L[k];v2=v1+mic(P[k,b]-P[k,a],k)
            images=[(v1+np.array([sx,sy])*L[k],v2+np.array([sx,sy])*L[k]) for sx in [-1,0,1] for sy in [-1,0,1]]
        for v1,v2 in images:
            if np.any(np.maximum(v1,v2)<lo) or np.any(np.minimum(v1,v2)>hi):continue
            seg.append([v1/10,v2/10]);cols.append(ORANGE if (a,b) not in S[0] else GRAY)
            edges_out.append({'panel':panel,'frame':k,'atom_i':a,'atom_j':b,'new_pair':(a,b) not in S[0],'x1_nm':v1[0]/10,'y1_nm':v1[1]/10,'x2_nm':v2[0]/10,'y2_nm':v2[1]/10})
    box=Rectangle(lo/10,*((hi-lo)/10),fill=False,edgecolor='#bdc6cc',lw=.7);ax.add_patch(box)
    lines=LineCollection(seg,colors=cols,linewidths=[(1.5 if c==ORANGE else .85) if iszoom else (.65 if c==ORANGE else .25) for c in cols]);ax.add_collection(lines);lines.set_clip_path(box)
    if iszoom:
        pts=mic(P[k]-origin,k);inside=np.all((pts>=lo)&(pts<=hi),axis=1)
        ax.scatter(*pts[inside].T/10,s=9,c=ATOM,linewidths=0,zorder=3)
        for atom in np.where(inside)[0]:exports.append({'panel':panel,'frame':k,'atom_id':int(atom),'x_nm':pts[atom,0]/10,'y_nm':pts[atom,1]/10,'degree':len(G[k][atom])})
        bar=.5;bx=lo[0]/10+.07;by=lo[1]/10-.10
    else:
        bar=2.;bx=.35;by=-.65
    ax.plot([bx,bx+bar],[by,by],color=ATOM,lw=1.6,clip_on=False)
    ax.text(bx+bar/2,by-.08,f'{bar:g} nm',ha='center',va='top',fontsize=9)
    ax.set(xlim=(lo[0]/10-.04,hi[0]/10+.04),ylim=(lo[1]/10-.35,hi[1]/10+.03),aspect='equal');ax.axis('off')
    return edges_out
edges_export=draw(axs[0],peak)
ctr=centres[peak];axs[0].add_patch(Rectangle((ctr-half)/10,*(2*half/10),fill=False,edgecolor=BOX,lw=1.4))
for ax,k in zip(axs[1:],frames):edges_export.extend(draw(ax,k,centres[k]))
for ax,label,title in zip(axs,['A','B','C','D'],['Peak configuration',r'$\epsilon=0$',r'$\epsilon=0.26125$ (peak)',r'$\epsilon=0.30125$']):
    ax.text(-.015,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=13,ha='left')
    ax.text(.08,1.06,title,transform=ax.transAxes,fontsize=11,ha='left')
# Label a short non-hexagonal loop and a six-atom loop, without implying a full ring census.
for ax,k,wanted in [(axs[2],36,[4,6]),(axs[3],44,[5,7])]:
    used=[]
    for size in wanted:
        candidates=[];seen=set()
        for row in cycle_rows:
            if row['frame']!=k or row['shortest_cycle_size']!=size:continue
            ids=row['cycle_atom_ids'];key=tuple(sorted(ids))
            if key in seen:continue
            seen.add(key)
            pts=[mic(P[k,ids[0]]-centres[k],k)]
            for a,b in zip(ids,ids[1:]):pts.append(pts[-1]+mic(P[k,b]-P[k,a],k))
            pts=np.array(pts)
            closing=pts[-1]+mic(P[k,ids[0]]-P[k,ids[-1]],k)
            if np.linalg.norm(closing-pts[0])>1e-4:continue
            if not np.all((pts>-half+1)&(pts<half-1)):continue
            centre=pts.mean(axis=0)
            if any(np.linalg.norm(centre-u)<3 for u in used):continue
            candidates.append((float(np.linalg.norm(centre)),ids,pts,centre))
        if candidates:
            _,ids,pts,centre=min(candidates,key=lambda x:x[0]);used.append(centre)
            ax.text(*(centre/10),str(size),ha='center',va='center',color='#244a67',fontsize=8.5,fontweight='bold',zorder=5,bbox=dict(facecolor='white',edgecolor='none',pad=.25,alpha=.88))
            annotations.append({'frame':k,'loop_size':size,'atom_ids':ids,'positions_relative_A':pts.tolist(),'label_position_A':centre.tolist()})
handles=[Line2D([],[],marker='o',ls='',color=ATOM,markersize=4,label='carbon atoms'),Line2D([],[],color=GRAY,lw=1.2,label='initial connections still present'),Line2D([],[],color=ORANGE,lw=1.7,label='new connections')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,fontsize=10)
for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'figures/slit45_lattice_detail.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
def write_csv(name,rows):
    with (ROOT/'data'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv('zoom_atoms.csv',exports);write_csv('displayed_connections.csv',edges_export)
(ROOT/'data/shortest_cycles.json').write_text(json.dumps(cycle_rows,indent=2)+'\n')
report={'source_run':str(RUN),'source_trajectory_sha256':hashlib.sha256((RUN/'trajectory.npz').read_bytes()).hexdigest(),'peak_frame':peak,'figure10_peak_panel':'c','figure10_later_panel':'d','tracked_pair':pair,'selection':'New pair nearest the normalized periodic cell centre at the primary stress maximum; same atom pair recentres every zoom.','field_width_A':(2*half).tolist(),'centres_A':{str(k):v.tolist() for k,v in centres.items()},'states':summary,'annotated_loops':annotations,'definition':'Geometric graph of atom pairs at minimum-image separation below 2 A; line colour records membership relative to frame 0, not atom type, defect severity or chemical bond order.','cycle_diagnostic':'Remove one new edge, find its shortest alternative path, then close the loop with that edge. This is a per-edge shortest-cycle diagnostic, not a census of distinct physical rings. Search bound: alternative path up to 24 edges. Annotated local loops were checked to close without periodic winding.','interpretation':'Locally reconnected six-membered loops coexist with non-hexagonal loops and remaining slit edges. These snapshots do not demonstrate a perfect, defect-free honeycomb sheet.','numerical_scope':'Selected peak and near-0.30 states meet recorded residual targets; the trajectory has flagged other frames, and this geometric analysis does not establish numerical convergence.'}
(ROOT/'data/summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'tracked_pair':pair,'states':summary,'annotated_loops':annotations},indent=2))
