"""Fit the enlarged geometry map and quantify descriptor-space coverage."""
from pathlib import Path
import json,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from scipy.spatial import cKDTree
import umap,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output'
def nearest_full(queries,data):
 # Full exhaustive 64D search in float64, blocked to limit temporary memory.
 # Candidate generation uses approximate retrieval; this held-out audit does not.
 data=np.asarray(data,dtype=np.float64);queries=np.asarray(queries,dtype=np.float64)
 norm=np.einsum('ij,ij->i',data,data);out=[];who=[]
 for start in range(0,len(queries),128):
  qq=queries[start:start+128]
  with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
   distances=np.einsum('ij,ij->i',qq,qq)[:,None]+norm[None,:]-2*(qq@data.T)
  assert np.isfinite(distances).all()
  ids=np.argmin(distances,axis=1)
  # Reevaluate winning distances directly to avoid cancellation near zero.
  out.extend(np.linalg.norm(data[ids]-qq,axis=1));who.extend(ids)
 return np.array(out),np.array(who)
def main():
 records=json.loads((A/'manifest.json').read_text());assert len(records)==256000;t=time.time()
 z=np.load(A/'geometric_descriptors.npz');shape=z['shape'];geometry=z['geometry'];assert np.isfinite(shape).all() and np.isfinite(geometry).all()
 if not (A/'latent_space.npz').exists():
  ss=shape.copy();ss[:,:1024]*=.7;ss[:,1024:1280]*=.55;ss[:,1280:1296]*=.35;ss[:,1296:]*=.45
  sc=StandardScaler().fit(geometry);scaled=np.clip(sc.transform(geometry),-4,4)/np.sqrt(10);X=np.c_[ss,.55*scaled].astype(np.float32);del ss,scaled
  pca=PCA(n_components=64,random_state=41,svd_solver='randomized');compressed=pca.fit_transform(X).astype(np.float32);del X
  print('Expanded PCA variance',pca.explained_variance_ratio_.sum(),flush=True)
  model=umap.UMAP(n_components=3,n_neighbors=60,min_dist=.18,spread=1.5,random_state=41,n_epochs=400,n_jobs=1,verbose=True)
  emb=model.fit_transform(compressed);rot=PCA(n_components=3).fit(emb);xyz=rot.transform(emb);xyz=xyz/np.max(np.ptp(xyz,axis=0))*80
  sample=np.random.default_rng(41).choice(256000,2400,replace=False);tw=trustworthiness(compressed[sample],emb[sample],n_neighbors=15)
  np.savez_compressed(A/'latent_space.npz',embedding=emb,display_xyz=xyz,features=compressed,pca_components=pca.components_,pca_mean=pca.mean_,pca_variance=pca.explained_variance_ratio_,geometry_mean=sc.mean_,geometry_scale=sc.scale_,display_rotation=rot.components_,display_mean=rot.mean_)
  (OUT/'embedding_qa.json').write_text(json.dumps(dict(n_designs=256000,pca_dimensions=64,pca_retained_variance=float(pca.explained_variance_ratio_.sum()),embedding='UMAP',components=3,n_neighbors=60,min_dist=.18,spread=1.5,epochs=400,seed=41,trustworthiness_sample_size=2400,trustworthiness_k=15,trustworthiness=float(tw),excludes=['family labels','mechanical properties'],coordinate_interpretation='Geometry similarity only; visual gaps are not automatically feasible missing designs',simulations_run=0),indent=2))
 features=np.load(A/'selection_features.npy');model=np.load(A/'heldout_targets.npz');targets=model['targets'];old_d,old_i=nearest_full(targets,features[:64000]);new_d,new_i=nearest_full(targets,features)
 # The original population is a strict subset; union distances cannot increase.
 assert np.all(new_d<=old_d+1e-10)
 improve=new_d<old_d-1e-10;new_i=np.where(improve,new_i,old_i);new_d=np.minimum(new_d,old_d)
 coverage=dict(query_targets=len(targets),held_out_from_candidate_selection=True,baseline_median_distance=float(np.median(old_d)),expanded_median_distance=float(np.median(new_d)),median_relative_reduction=float(np.median((old_d-new_d)/np.maximum(old_d,1e-12))),fraction_queries_improved=float(np.mean(improve)),fraction_queries_improved_over_10_percent=float(np.mean(new_d<.9*old_d)),nearest_method='Exhaustive Euclidean nearest-neighbour search in all 64 frozen PCA coordinates, float64, against each full population',interpretation='Closer coverage of saved cross-group midpoint queries, not evidence every empty region of UMAP is populated or physically feasible')
 np.savez_compressed(A/'bridge_coverage.npz',targets=targets,parent_pairs=model['pairs'],old_distance=old_d,new_distance=new_d,old_neighbor=old_i,new_neighbor=new_i)
 (OUT/'coverage_qa.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage,indent=2),flush=True)
 dest=A/'sprites.npy'
 if not dest.exists():
  sprites=np.lib.format.open_memmap(dest,mode='w+',dtype=np.uint8,shape=(256000,72,72))
  for i in range(256000):
   g=np.array(Image.open(A/'designs'/f'{i:04d}.png'));sprites[i]=cv2.resize(g,(72,72),interpolation=cv2.INTER_AREA)
   if i%16000==0:print('Sprites',i,flush=True)
  sprites.flush()
 print('Embedding ready',time.time()-t,flush=True)
if __name__=='__main__':main()
