"""Paper figure from saved atomic structures and the unchanged released UMAP."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle,ConnectionPatch
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
(ROOT / 'figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'output').mkdir(parents=True, exist_ok=True)
DATA=ROOT/'data'
font=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
if font.exists():font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':'Arial' if font.exists() else 'DejaVu Sans',
 'font.size':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.45})
FG='#172534';MUTED='#53616D';EDGE='#D4DDE2'
COLORS=['#247FA5','#B27A32','#33876A','#AD527F','#7860B2','#89843D']

def projection(xyz):
    yaw,pitch=np.deg2rad([37,20])
    M=np.array([[np.cos(yaw),0,-np.sin(yaw)],[-np.sin(yaw)*np.sin(pitch),np.cos(pitch),-np.cos(yaw)*np.sin(pitch)]])
    assert np.allclose(M@M.T,np.eye(2))
    return np.stack([np.sum(xyz*M[0],axis=1),np.sum(xyz*M[1],axis=1)],axis=1),M

def geometry(fig,r,box,color,lw=.19):
    ax=fig.add_axes(box)
    geo=np.load(DATA/f"GDU-{r['id']:06d}.npz")
    p=geo['positions'][:,:2];cell=np.diag(geo['cell'])[:2];edges=geo['edges']
    assert len(p)==r['n_atoms']
    ax.add_collection(LineCollection(p[edges],colors=color,linewidths=lw))
    # Atoms remain rasterized at print resolution; bonds and cell are vectors.
    ax.scatter(p[:,0],p[:,1],s=.023,c=color,linewidths=0,rasterized=True)
    ax.add_patch(Rectangle((0,0),*cell,fill=False,edgecolor=EDGE,lw=.4))
    span=max(cell)*1.06;c=cell/2
    ax.set(xlim=(c[0]-span/2,c[0]+span/2),ylim=(c[1]-span/2,c[1]+span/2))
    ax.set_aspect('equal');ax.axis('off')
    return ax

def main():
    cfg=json.loads((DATA/'manifest.json').read_text());z=np.load(DATA/'embedding.npz')
    P,M=projection(z['display_xyz']);groups=np.empty(24,int)
    for g,row in enumerate(cfg['languages']):groups[row['themes']]=g
    family=groups[z['theme_index']];color=np.array(COLORS)[family]
    order=np.random.default_rng(419).permutation(len(P))
    fig=plt.figure(figsize=(7.45,6.05),facecolor='white')
    fig.text(.025,.975,'a',fontsize=11,fontweight='bold',color=FG)
    fig.text(.535,.975,'b',fontsize=11,fontweight='bold',color=FG)
    # Main panel is a declared central magnification; overview includes every point.
    ax=fig.add_axes([.013,.391,.491,.567])
    ax.scatter(P[order,0],P[order,1],s=.12,c=color[order],alpha=.40,linewidths=0,rasterized=True)
    ax.set(xlim=(-20,20),ylim=(-20,20));ax.set_aspect('equal');ax.axis('off')
    fig.add_artist(Rectangle((.014,.808),.184,.162,transform=fig.transFigure,facecolor='white',edgecolor='none',zorder=4))
    inset=fig.add_axes([.031,.828,.147,.14],facecolor='white',zorder=5)
    inset.patch.set_visible(True)
    inset.scatter(P[order,0],P[order,1],s=.022,c=color[order],alpha=.52,linewidths=0,rasterized=True)
    extent=np.ptp(P,axis=0);ctr=(P.max(0)+P.min(0))/2
    inset.set(xlim=(ctr[0]-extent[0]*.54,ctr[0]+extent[0]*.54),ylim=(ctr[1]-extent[1]*.54,ctr[1]+extent[1]*.54))
    inset.set_aspect('equal');inset.axis('off')
    inset.add_patch(Rectangle((-20,-20),40,40,fill=False,edgecolor=FG,lw=.5))
    # Six numbered exemplars, matching the library; do not mark centroids.
    pins=[70507,98520,183686,223042,110528,254261]
    offsets=[(-9,9),(-12,1),(10,-8),(8,9),(-10,6),(10,10)]
    anchors=[]
    for g,(idx,offs) in enumerate(zip(pins,offsets)):
        pt=P[idx];target=ax if np.all(np.abs(pt)<20) else inset
        target.scatter(*pt,s=13 if target is ax else 6,facecolors='white',edgecolors=COLORS[g],linewidths=.8,zorder=8)
        target.annotate(str(g+1),pt,xytext=offs,textcoords='offset points',color=FG,fontsize=6.3,fontweight='bold',ha='center',va='center',bbox=dict(boxstyle='circle,pad=.22',facecolor='white',edgecolor=COLORS[g],linewidth=.65),arrowprops=dict(arrowstyle='-',lw=.5,color=COLORS[g]),zorder=10)
        anchors.append({'number':g+1,'id':idx,'x':float(pt[0]),'y':float(pt[1]),'in_overview':target is inset})
    short={0:'Round',1:'Square',2:'Hexagonal',3:'Oriented',4:'Angled slits',5:'Struts',6:'Polygons',7:'Gradients',8:'Flaws / halos',9:'Defects / seams',10:'Parallel veins',11:'Crossed veins',12:'Nested',13:'Fibrous veins',14:'Composite grids',15:'Subveins',16:'Morphing',17:'Pore-slit',18:'Curved veins',19:'Branching',20:'Graded angles',21:'Multiscale',22:'Correlated',23:'Hybrid domains'}
    for g,lang in enumerate(cfg['languages']):
        top=.936-g*.152
        fig.text(.547,top,lang['name'],fontsize=7.9,color=COLORS[g],fontweight='bold')
        for j,ti in enumerate(lang['themes']):
            r=next(v for v in cfg['examples'] if v['theme_index']==ti)
            left=.536+j*.113
            ga=geometry(fig,r,[left,top-.113,.105,.103],COLORS[g])
            label=short.get(ti,r['theme'])
            fig.text(left+.0525,top-.129,label,ha='center',fontsize=7,color=FG)
            if r['id']==pins[g]:
                ga.text(.04,.98,str(g+1),transform=ga.transAxes,fontsize=6.2,fontweight='bold',ha='center',va='center',color=FG,bbox=dict(boxstyle='circle,pad=.18',facecolor='white',edgecolor=COLORS[g],linewidth=.6))
    # Local retrieval uses the unchanged 3D embedding and exact nearest neighbors.
    fig.text(.025,.355,'c',fontsize=11,fontweight='bold',color=FG)
    for rank,r in enumerate(cfg['neighborhood']):
        left=.022+rank*.123
        g=groups[r['theme_index']]
        ga=geometry(fig,r,[left,.145,.114,.17],COLORS[g],lw=.24)
        if rank==0:
            ga.text(.02,.95,'6',transform=ga.transAxes,fontsize=6.2,fontweight='bold',ha='center',va='center',color=FG,bbox=dict(boxstyle='circle,pad=.18',facecolor='white',edgecolor=COLORS[g],linewidth=.6))
        else:
            ga.text(.02,.95,str(rank),transform=ga.transAxes,fontsize=6.2,fontweight='bold',color=MUTED)
        fig.text(left+.057,.119,rf"$\phi={r['porosity']:.2f}$",ha='center',fontsize=6.9,color=FG)
        fig.text(left+.057,.092,rf"$L_x={r['cell_A'][0]/10:.1f}$ nm",ha='center',fontsize=6.9,color=MUTED)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outliers=[]
    for t in fig.texts:
        bb=t.get_window_extent(renderer)
        if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:outliers.append(t.get_text())
    assert not outliers,outliers
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/'figures'/f'fig_design_atlas.{ext}',dpi=600 if ext!='png' else 350,facecolor='white')
    plt.close(fig)
    with (DATA/'plotted_embedding.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['design_id','theme_index','language_index','projection_x','projection_y','in_zoom'])
        for i in range(len(P)):w.writerow([f'GDU-{i:06d}',int(z['theme_index'][i]),int(family[i]),*map(float,P[i]),bool(np.all(np.abs(P[i])<=20))])
    (DATA/'map_pins.json').write_text(json.dumps(anchors,indent=2))
    qa={'population':len(P),'generator_groups':len(np.unique(z['theme_index'])),'library_examples':len(cfg['examples']),
        'in_main_zoom':int(np.all(np.abs(P)<=20,axis=1).sum()),'overview_population':len(P),'projection_matrix':M.tolist(),
        'embedding_refitted':False,'main_zoom_bounds':[-20,20,-20,20],'atom_counts_checked':True,
        'neighborhood_ids':[r['id'] for r in cfg['neighborhood']], 'figure_text_bounds_checked':True,
        'vector_layers':'Typography, connection lines, atomic bonds, cell outlines. Dense embedding and atoms rasterized at 600 dpi.',
        'visual_review':'pending'}
    (ROOT/'output/validation.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
