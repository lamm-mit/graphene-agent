"""Deterministic novelty/bridge selection on measured, unrelaxed atomic geometries."""
from pathlib import Path
import json,copy,shutil,time,hashlib,os,concurrent.futures as cf
import numpy as np
from PIL import Image
from scipy.stats import qmc
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import fast_geometry as F
from generate_atlas import spec,digest,thumbnail
from source.structures import design_space as D
from hybrid_geometry import NAMES,COLORS,recipe,generate
from embed_designs import describe
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'graphene_design_dataset_64k';A=ROOT/'assets';OUT=ROOT/'output'
LENGTHS={'Lx','Ly','period','pore_d','jitter','slit_len','slit_w','period_x','period_y','spacing','strut_w','ligament_w','pore_d_min','pore_d_max','crack_len','crack_w','hole_d','ring_pore_d','ring_gap','domain','vein_w','bar_w','level3_domain','level3_vein','sub_domain','sub_w','pitch','seam_spacing','vein_offset'}
def scale_params(p,scale):
 return {k:scale_params(v,scale) if isinstance(v,dict) else (float(v)*scale if k in LENGTHS and v is not None else [float(x)*scale for x in v] if k=='offset' else v) for k,v in p.items()}
def prepare():
 t=time.time();old=json.loads((OLD/'assets/manifest.json').read_text());assert len(old)==64000
 # Copy immutable source artifacts; no links that could mutate the older release.
 for i,r in enumerate(old):
  for ext in ['npz','json','png']:
   src=OLD/'assets/designs'/f'{i:04d}.{ext}';dst=A/'designs'/src.name
   if not dst.exists():shutil.copy2(src,dst)
 print('Preserved 64000 source designs',round(time.time()-t,1),flush=True)
 z=np.load(OLD/'assets/geometric_descriptors.npz');latent=np.load(OLD/'assets/latent_space.npz')
 geom=z['geometry'];sc=StandardScaler().fit(geom)
 shape=z['shape'].copy();shape[:,:1024]*=.70;shape[:,1024:1280]*=.55;shape[:,1280:1296]*=.35;shape[:,1296:]*=.45
 X=np.c_[shape,.55*np.clip(sc.transform(geom),-4,4)/np.sqrt(10)].astype(np.float32)
 base=np.einsum('nj,ij->ni',X-latent['pca_mean'],latent['pca_components'],optimize=False).astype(np.float32);del X,shape
 idx=np.random.default_rng(256000).choice(64000,8192,replace=False);tree=cKDTree(base[:,:16]);nn=tree.query(base[idx,:16],k=96,workers=4)[1]
 pairs=[]
 for i,near in zip(idx,nn):
  options=[j for j in near if old[j]['theme_index']!=old[i]['theme_index']]
  if options:pairs.append([int(i),int(options[0])])
 pairs=np.array(pairs);target=(base[pairs[:,0]]+base[pairs[:,1]])/2
 heldout=np.random.default_rng(771921).choice(np.setdiff1d(np.arange(64000),np.unique(pairs)),4096,replace=False);hn=tree.query(base[heldout,:16],k=96,workers=4)[1];hp=[]
 for i,near in zip(heldout,hn):
  options=[j for j in near if old[j]['theme_index']!=old[i]['theme_index']]
  if options:hp.append([int(i),int(options[0])])
 hp=np.array(hp);ht=(base[hp[:,0]]+base[hp[:,1]])/2
 np.savez_compressed(A/'heldout_targets.npz',targets=ht,pairs=hp,anchor_ids=heldout)
 np.savez_compressed(A/'selection_model.npz',mean=sc.mean_,scale=sc.scale_,components=latent['pca_components'],pca_mean=latent['pca_mean'],baseline=base,targets=target,pairs=pairs)
 (OUT/'selection_model.json').write_text(json.dumps(dict(baseline_count=64000,target_count=len(target),construction='Midpoints of close baseline pairs from different geometry groups; targets are descriptor queries, not interpolated atomic coordinates',seed=256000,selection_space='Frozen 64K weighted-descriptor PCA, first 16 components for neighbour candidate search with 64D reranking'),indent=2))
 return old

def setup_worker():
 global MODEL,TREE,TARGET_TREE
 F.install();MODEL=dict(np.load(A/'selection_model.npz'));TREE=cKDTree(MODEL['baseline'][:,:16]);TARGET_TREE=cKDTree(MODEL['targets'][:,:16])

def project(shape,geometry):
 scaled=np.clip((geometry-MODEL['mean'])/MODEL['scale'],-4,4)/np.sqrt(10)
 ss=shape.copy();ss[:1024]*=.70;ss[1024:1280]*=.55;ss[1280:1296]*=.35;ss[1296:]*=.45
 X=np.r_[ss,.55*scaled].astype(np.float32)
 result=np.einsum('j,ij->i',X-MODEL['pca_mean'],MODEL['components'],optimize=False)
 assert np.isfinite(result).all()
 return result

def candidate(theme,serial,q):
 seed=256000000+theme*1000000+serial
 if theme<16:
  fam,p=spec(theme,serial%255,q[:12]);target=10**(np.log10(48)+q[16]*(np.log10(720)-np.log10(48)))
  if seed%1601==0:target=1500+3500*q[17]
  p=scale_params(p,target/p['Lx']);p['Ly']=target*np.exp((q[17]-.5)*1.6);p['seed']=seed;p['orientation']='zigzag' if q[18]<.5 else 'armchair'
  if 'angle_deg' in p:p['angle_deg']=float(180*q[19])
  if 'size_disorder' in p:p['size_disorder']=float(.35*q[20])
  if 'jitter' in p:p['jitter']=float(.18*p['period']*q[21])
  at=D.generate(fam,**copy.deepcopy(p));P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True;pruned=at.info.get('n_pruned',0)
 else:
  p=recipe(theme,q,seed);fam='periodic_hybrid';P,L,ng,keep,pruned=generate(p)
 n=int(keep.sum());porosity=1-n/len(P)
 if n<100 or porosity>.94 or pruned/len(P)>.35:return None,'geometry_screen'
 h=digest(P,L,keep);s,g=describe(P,L,ng,keep);feat=project(s,g)
 near=TREE.query(feat[:16],k=12)[1];dist=np.linalg.norm(MODEL['baseline'][near]-feat,axis=1);j=int(np.argmin(dist));oldid=int(near[j]);novel=float(dist[j])
 targets=TARGET_TREE.query(feat[:16],k=8)[1];td=np.linalg.norm(MODEL['targets'][targets]-feat,axis=1);k=int(np.argmin(td));targetid=int(targets[k]);bridge=float(td[k])
 r=dict(theme=NAMES[theme],theme_index=theme,family=fam,params=p,n_atoms=n,n_pristine=len(P),cell_A=L.tolist(),porosity=porosity,n_pruned=int(pruned),geometry_digest=h,source='256K geometry expansion',mechanical_evaluation=False,
  selection_candidate=serial,baseline_neighbor_id=oldid,baseline_descriptor_distance=novel,bridge_target_id=targetid,bridge_descriptor_distance=bridge)
 return dict(record=r,mask=np.packbits(keep),shape=s,geometry=g,features=feat),None

def chunk(job):
 theme,start,count,offset=job;stamp=f'{theme:02d}_{start:06d}';done=ROOT/'validation'/f'generation_{stamp}.json'
 if done.exists():return json.loads(done.read_text())
 # Each deterministic job samples 1.5x its final population before selection.
 candidates=[];rejected=[];seen=set();attempt=0
 seq=qmc.Sobol(32,scramble=True,seed=912000+theme*10000+start).random_base2(11)
 need=(count*3+1)//2
 while len(candidates)<need:
  q=seq[attempt] if attempt<len(seq) else np.random.default_rng(theme*10000000+start*10000+attempt).random(32)
  serial=start*2000+attempt;attempt+=1;c,reason=candidate(theme,serial,q)
  if c is None:rejected.append(dict(candidate=serial,reason=reason));continue
  if c['record']['geometry_digest'] in seen:rejected.append(dict(candidate=serial,reason='duplicate_in_pool'));continue
  seen.add(c['record']['geometry_digest']);candidates.append(c)
  if attempt>max(3000,count*20):raise RuntimeError((job,'insufficient candidates'))
 novelty=np.array([c['record']['baseline_descriptor_distance'] for c in candidates]);bridge=np.array([c['record']['bridge_descriptor_distance'] for c in candidates])
 # Keep substantial baseline distance while approaching between-group queries.
 bridge_score=bridge-.35*novelty;chosen=[];kinds={}
 for order,n,kind in [(np.argsort(bridge_score),count//3,'descriptor_bridge'),(np.argsort(-novelty),count//3,'descriptor_novelty'),(np.arange(len(candidates)),count,'low_discrepancy_coverage')]:
  added=0
  for k in order:
   if int(k) in kinds:continue
   chosen.append(int(k));kinds[int(k)]=kind;added+=1
   if len(chosen)==count or added==n:break
  if len(chosen)==count:break
 rows=[];shape=[];geom=[];feats=[]
 for j,k in enumerate(chosen):
  c=candidates[k];r=c['record'];i=offset+start+j;r['id']=i;r['selection_strategy']=kinds[k]
  path=A/'designs'/f'{i:04d}';assert not path.with_suffix('.json').exists(),('refuse overwrite',i)
  p=r['params'];np.savez_compressed(path.with_suffix('.npz'),keep=c['mask'],target=[p['Lx'],p['Ly']],orientation=p['orientation'])
  P,L,ng,keep=F.load_design(path.with_suffix('.npz'));assert digest(P,L,keep)==r['geometry_digest']
  Image.fromarray(thumbnail(P,L,ng,keep)).save(path.with_suffix('.png'))
  tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(r,separators=(',',':')));tmp.rename(path.with_suffix('.json'))
  rows.append(r);shape.append(c['shape']);geom.append(c['geometry']);feats.append(c['features'])
 np.savez_compressed(ROOT/'validation'/f'descriptors_{stamp}.npz',ids=[r['id'] for r in rows],shape=shape,geometry=geom,features=feats)
 report=dict(theme=theme,start=start,count=count,offset=offset,candidates=len(candidates),attempts=attempt,rejections=rejected,selected_median_baseline_distance=float(np.median(novelty[chosen])),pool_median_baseline_distance=float(np.median(novelty)),selected_median_bridge_distance=float(np.median(bridge[chosen])),pool_median_bridge_distance=float(np.median(bridge)))
 done.write_text(json.dumps(report,indent=2));return report

def main():
 old=prepare();jobs=[]
 for f in range(24):
  count=4000 if f<16 else 16000;offset=64000+f*4000 if f<16 else 128000+(f-16)*16000
  jobs.extend([(f,start,min(256,count-start),offset) for start in range(0,count,256)])
 t=time.time();reports=[]
 with cf.ProcessPoolExecutor(max_workers=4,initializer=setup_worker) as pool:
  for r in pool.map(chunk,jobs,chunksize=1):
   reports.append(r);progress=dict(completed=64000+sum(x['count'] for x in reports),total=256000,elapsed_s=time.time()-t,last_theme=r['theme'])
   (OUT/'generation_progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
 records=old+[json.loads((A/'designs'/f'{i:04d}.json').read_text()) for i in range(64000,256000)]
 # Coordinate duplicates across independently generated pools must be resolved,
 # never silently counted as unique. Keep every attempt if an audit finds one.
 hashes=[r['geometry_digest'] for r in records]
 if len(set(hashes))!=256000:
  from collections import Counter
  duplicates={h:n for h,n in Counter(hashes).items() if n>1};(OUT/'duplicates_to_resolve.json').write_text(json.dumps(duplicates));raise RuntimeError('cross-pool duplicate coordinates require recorded replacement')
 (A/'manifest.json').write_text(json.dumps(records,separators=(',',':')))
 oldz=np.load(OLD/'assets/geometric_descriptors.npz');shape=np.empty((256000,oldz['shape'].shape[1]),np.float32);geom=np.empty((256000,10),np.float32);features=np.empty((256000,64),np.float32)
 shape[:64000]=oldz['shape'];geom[:64000]=oldz['geometry'];features[:64000]=np.load(A/'selection_model.npz')['baseline']
 for p in sorted((ROOT/'validation').glob('descriptors_*.npz')):
  z=np.load(p);shape[z['ids']]=z['shape'];geom[z['ids']]=z['geometry'];features[z['ids']]=z['features']
 np.savez_compressed(A/'geometric_descriptors.npz',shape=shape,geometry=geom);np.save(A/'selection_features.npy',features)
 summary=dict(designs=256000,unique_geometries=len(set(hashes)),preserved_designs=64000,new_designs=192000,candidate_pool=sum(r['candidates'] for r in reports),theme_count=24,generator_families=sorted(set(r['family'] for r in records)),total_atoms=sum(r['n_atoms'] for r in records),min_atoms=min(r['n_atoms'] for r in records),max_atoms=max(r['n_atoms'] for r in records),width_range_nm=[min(r['cell_A'][0]/10 for r in records),max(r['cell_A'][0]/10 for r in records)],simulations_run=0)
 (OUT/'generation_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'selection_chunks.json').write_text(json.dumps(reports));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
