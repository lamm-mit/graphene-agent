"""Generate 4096 unique atomistic designs using the copied research generators.
All coordinates are losslessly stored as site-occupancy masks on exact lattices.
No force field, relaxation, stress evaluation or dynamics is invoked.
"""
from pathlib import Path
import json,time,hashlib,copy,sys
import numpy as np
from scipy.stats import qmc
import cv2
from PIL import Image
from source.structures import design_space as D
import fast_geometry as F

ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output'
NAMES=['Round pores','Square pores','Hexagonal pores','Oriented pores','Angled slits','Strut networks','Polygon networks','Pore gradients','Flaws and halos','Defects and seams','Parallel veins','Crossed veins','Nested hierarchy','Fibrous veins','Composite grids','Subvein hierarchy']
COLORS=[(77,181,225),(97,205,216),(86,181,239),(110,161,237),(241,184,105),(214,158,109),(164,173,235),(128,202,163),(224,162,151),(171,184,200),(101,211,173),(88,189,183),(161,184,235),(210,166,234),(188,151,231),(222,170,207)]
SIZES=[50,75,100,150,200,250,300,500]

def spec(f,k,q):
    L=SIZES[k//32];L=([1000,1500,2000,2500,3000,3500][f-10] if k==255 and f>=10 else L)
    # Rectangular cells as well as squares, always commensurate with graphene.
    Ly=L*([1,.8,1.2][k%3]);orient='zigzag' if k%2==0 else 'armchair'
    p=dict(Lx=L,Ly=Ly,orientation=orient,seed=10000+256*f+k)
    period=max(12.,L/(3+int(7*q[0])));pd=period*(.28+.3*q[1]);shape=['circle','square','hexagon'][min(f,2)]
    if f<4:
        family='nanomesh_single';p.update(period=period,pore_d=pd,shape=shape if f<3 else 'circle',stagger=k%3!=1,offset=[2.46*q[10],2.46*q[11]])
        if f==3:p.update(aspect=1.3+1.1*q[2],pore_d=pd*.7,angle_deg=90*q[3])
        if k%4==0:p.update(jitter=.08*period*q[4],size_disorder=.15*q[5])
    elif f==4:
        family='slit_array';p.update(period_x=period*1.6,period_y=period,slit_len=period*(.9+.55*q[1]),slit_w=max(2.5,period*(.08+.13*q[2])),angle_deg=90*q[3],stagger=k%2==0)
    elif f==5:
        family='strut_lattice';p.update(spacing=period*1.6,strut_w=max(4.,period*(.3+.4*q[2])),angles_deg=([0,90] if k%3==0 else [0,60,120] if k%3==1 else [30,90,150]))
    elif f==6:
        family='voronoi_network';p.update(n_cells=10+int(25*q[1]),regularity=q[2],ligament_w=max(4.,L/(10+7*q[3])))
    elif f==7:
        family='graded_pores';p.update(period=period,pore_d_min=max(2.5,period*.16),pore_d_max=period*(.5+.22*q[1]),mode='x' if k%2 else 'radial',shape='circle' if k%3 else 'square')
    elif f==8:
        if k%4==0:
            family='precrack';p.update(crack_len=L*(.2+.45*q[1]),crack_w=max(2.5,L*.025),angle_deg=90*q[2])
        else:
            family='ring_around_hole';p.update(hole_d=L*(.08+.13*q[1]),n_rings=1+k%3,ring_pore_d=max(2.8,L*(.02+.025*q[2])),ring_gap=L*(.08+.025*q[3]))
    elif f==9:
        if k%3==0:
            family='seam_pores';p.update(pore_d=max(2.8,period*(.26+.16*q[2])),pitch=period*.8,seam_spacing=period*1.8,seam_dir='x' if k%2 else 'y',seam_anchor=[q[10],q[11]])
        else:
            family='vacancies';p.update(fraction=.004+.025*q[1],organisation='random' if k%3==1 else 'clustered',cluster_size=4+int(15*q[2]))
    elif f<=12:
        family='nanomesh_hier';dom=L/(2+int(q[0]*3));fine=max(11.,dom/(3+q[1]*4))
        p.update(period=fine,pore_d=fine*(.35+.23*q[2]),domain=dom,vein_w=max(4.,dom*(.13+.13*q[3])),levels=3 if f==12 else 2,vein_dirs='x' if f==10 else 'xy',shape='circle' if k%3 else 'square',offset=[2.46*q[10],2.46*q[11]],vein_offset=2.46*q[9])
        if f==12:p.update(level3_domain=L/(1+int(q[4]*2)),level3_vein=dom*(.36+.15*q[5]))
        if k%3==2:p.update(aspect=1.8,angle_deg=90,pore_d=p['pore_d']*.7)
    else:
        family='composite';dom=L/(2+int(q[0]*3));fine=max(12.,dom/(4+q[1]*4));pd=fine*(.28+.13*q[2])
        p.update(coarse='square' if f==14 else 'veins_x',domain=dom,vein_w=dom*(.24+.13*q[3]),bar_w=dom*(.42+.14*q[3]),fill_solid='slits_x',fill_domain='subveins_ellipse' if f==15 else 'round' if k%2 else 'ellipse_x',
          solid_params=dict(slit_len=fine*1.7,slit_w=max(2.,fine*.16),period_y=fine*.7),domain_params=dict(period=fine,pore_d=pd,sub_domain=dom/3,sub_w=max(3.,dom*.07)))
        if f==14:p.update(fill_solid_y='round',solid_y_params=dict(period=fine,pore_d=pd),fill_domain='empty' if k%3==0 else 'round')
    return family,p

def digest(P,L,keep):
    h=hashlib.sha256();h.update(L.astype('<f8').tobytes());h.update(P[:4].astype('<f8').tobytes());h.update(np.packbits(keep).tobytes());return h.hexdigest()

def thumbnail(P,L,ng,keep,size=192):
    pad=7;sc=min((size-2*pad)/L[0],(size-2*pad)/L[1]);xy=np.rint((P[:,:2]-L[:2]/2)*[sc,-sc]+size/2).astype(np.int32)
    im=np.zeros((size,size),np.uint8)
    if len(P)<22000:
        ids=np.flatnonzero(keep);i=np.repeat(ids,3);j=ng[ids].ravel();good=(j>i)&keep[j]&np.all(np.abs(P[j,:2]-P[i,:2])<L[:2]/2,axis=1);b=np.stack([xy[i[good]],xy[j[good]]],axis=1)
        cv2.polylines(im,b,False,230,1,cv2.LINE_AA)
    else:
        # Area occupancy, rendered from every retained atom, not a drawn pore mask.
        r=xy[keep];np.maximum.at(im,(r[:,1],r[:,0]),235)
        if sc>1:im=cv2.dilate(im,np.ones((2,2),np.uint8))
    return im

def equivalence():
    samples=[]
    for f in range(16):
        for k in [3,6]:
            family,p=spec(f,k,np.random.default_rng(f*100+k).uniform(.1,.9,12));p['Lx']=75;p['Ly']=80
            at=D.generate(family,**copy.deepcopy(p));samples.append((family,p,at.positions.copy(),len(at),at.info.get('n_pruned',0)))
    F.install();checks=[]
    for family,p,P,n,pruned in samples:
        at=D.generate(family,**copy.deepcopy(p));same=at.positions.shape==P.shape and np.array_equal(at.positions,P)
        checks.append(dict(family=family,atom_count=n,exact_coordinates=same,pruning_match=pruned==at.info.get('n_pruned',0)))
    (OUT/'generator_equivalence.json').write_text(json.dumps(checks,indent=2));assert all(v['exact_coordinates'] and v['pruning_match'] for v in checks),checks
    print('32 exact source-generator equivalence checks passed',flush=True)

def main():
    equivalence();records=[];hashes=set();attempts=[];t=time.time()
    for f in range(16):
        sequence=qmc.Sobol(12,scramble=True,seed=317+f).random_base2(11)
        for k in range(256):
            id=f*256+k;file=A/'designs'/f'{id:04d}.npz';meta=file.with_suffix('.json')
            if meta.exists():
                r=json.loads(meta.read_text());records.append(r);hashes.add(r['geometry_digest']);continue
            for attempt in range(40):
                q=sequence[k] if attempt==0 else np.random.default_rng(100000*id+attempt).random(12)
                family,p=spec(f,k,q);at=D.generate(family,**copy.deepcopy(p))
                P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True;h=digest(P,L,keep)
                if len(at)>60 and h not in hashes:break
                attempts.append(dict(id=id,attempt=attempt,family=family,params=p,n_atoms=len(at),reason='duplicate' if h in hashes else 'too_few_atoms'))
            else:raise RuntimeError(('Could not generate unique structure',id))
            hashes.add(h)
            r=dict(id=id,theme=NAMES[f],theme_index=f,family=family,params=p,n_atoms=len(at),n_pristine=len(P),cell_A=L.tolist(),porosity=1-len(at)/len(P),n_pruned=at.info.get('n_pruned',0),geometry_digest=h,source='new procedural geometry',mechanical_evaluation=False)
            np.savez_compressed(file,keep=np.packbits(keep),target=np.array([p['Lx'],p['Ly']]),orientation=p['orientation'])
            meta.write_text(json.dumps(r,indent=2));Image.fromarray(thumbnail(P,L,ng,keep)).save(A/'designs'/f'{id:04d}.png')
            records.append(r)
            if k%32==31:print(f'{len(records):4d}/4096 {NAMES[f]:20s} {time.time()-t:.1f}s',flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(completed=len(records),elapsed_s=time.time()-t,last_theme=NAMES[f]),indent=2))
    assert len(records)==4096 and len(hashes)==4096
    (A/'manifest.json').write_text(json.dumps(records,indent=2));(OUT/'generation_attempts.json').write_text(json.dumps(attempts,indent=2))
    summary=dict(designs=4096,unique_geometries=len(hashes),generator_families=sorted(set(r['family'] for r in records)),themes=NAMES,total_retained_atoms=sum(r['n_atoms'] for r in records),min_atoms=min(r['n_atoms'] for r in records),max_atoms=max(r['n_atoms'] for r in records),min_width_nm=min(r['cell_A'][0]/10 for r in records),max_width_nm=max(r['cell_A'][0]/10 for r in records),porosity_range=[min(r['porosity'] for r in records),max(r['porosity'] for r in records)],simulations_run=0)
    (OUT/'generation_summary.json').write_text(json.dumps(summary,indent=2))
    # Each of the 4096 structures occurs exactly once in the spatial atlas.
    order=[]
    for k in range(256):
        for f in range(16):order.append(f*256+k)
    mosaic=np.zeros((45*96,92*96,3),np.uint8)
    for n,id in enumerate(order):
        g=cv2.resize(np.array(Image.open(A/'designs'/f'{id:04d}.png')),(90,90),interpolation=cv2.INTER_AREA)
        rgb=np.rint(g[:,:,None]*np.array(COLORS[id//256])[None,None,:]/255).astype(np.uint8);y,x=divmod(n,92);mosaic[y*96+3:y*96+93,x*96+3:x*96+93]=rgb
    Image.fromarray(mosaic).save(A/'atlas.png');(A/'atlas_order.json').write_text(json.dumps(order))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
