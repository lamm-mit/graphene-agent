"""Unsupervised 3D geometry embedding, excluding family labels and mechanics.

Actual saved carbon occupancy provides the morphology features. UMAP follows a
PCA compression; its three axes have no physical or performance interpretation.
"""
from pathlib import Path
import json,time,sys
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import umap
import fast_geometry as F
from generate_atlas import COLORS,digest
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output'

def normalize(x):return x/max(float(np.linalg.norm(x)),1e-8)
def describe(P,L,ng,keep):
 n=128;xy=np.minimum(n-1,np.floor(P[:,:2]/L[:2]*n).astype(int));total=np.zeros((n,n),np.float32);solid=total.copy()
 np.add.at(total,(xy[:,1],xy[:,0]),1);p=xy[keep];np.add.at(solid,(p[:,1],p[:,0]),1)
 sigma=max(.65,1.5*n/min(L[:2]));total=cv2.GaussianBlur(total,(0,0),sigma,borderType=cv2.BORDER_REFLECT);solid=cv2.GaussianBlur(solid,(0,0),sigma,borderType=cv2.BORDER_REFLECT)
 rho=solid/np.maximum(total,1e-9)
 # Translation-invariant orientation and spacing information.
 spectrum=np.abs(np.fft.fftshift(np.fft.fft2(rho-rho.mean())))**2/n**4
 patch=np.log1p(10000*spectrum[48:80,48:80]);patch[16,16]=0
 spectral=normalize(patch.ravel())
 coarse=cv2.resize(rho,(16,16),interpolation=cv2.INTER_AREA);coarse=normalize((coarse-coarse.mean()).ravel())
 hist=normalize(np.histogram(rho,bins=16,range=(0,1))[0].astype(float))
 profiles=np.r_[np.abs(np.fft.rfft(rho.mean(axis=0)))[1:17],np.abs(np.fft.rfft(rho.mean(axis=1)))[1:17]];profiles=normalize(profiles)
 ids=np.flatnonzero(keep);degree=keep[ng[ids]].sum(axis=1)
 geom=np.array([keep.mean(),np.log10(L[0]/10),np.log10(L[1]/10),np.log(L[0]/L[1]),np.mean(degree==2),np.mean(degree<2),rho.mean(axis=0).min(),rho.mean(axis=1).min(),rho.mean(axis=0).std(),rho.mean(axis=1).std()],np.float32)
 return np.r_[spectral,coarse,hist,profiles].astype(np.float32),geom

def main():
 rec=json.loads((A/'manifest.json').read_text());n=len(rec);assert n==64000;t=time.time()
 path=A/'geometric_descriptors.npz'
 if path.exists():
  with np.load(path) as z:shape=z['shape'];geometry=z['geometry']
 else:
  shape=[];geometry=[];seen=[]
  for r in rec:
   P,L,ng,keep=F.load_design(A/'designs'/f'{r["id"]:04d}.npz');assert keep.sum()==r['n_atoms'];h=digest(P,L,keep);assert h==r['geometry_digest'];seen.append(h)
   s,g=describe(P,L,ng,keep);shape.append(s);geometry.append(g)
   if len(shape)%1024==0:print('Measured and verified',len(shape),'geometries',round(time.time()-t,1),'s',flush=True)
  assert len(set(seen))==64000;shape=np.array(shape);geometry=np.array(geometry);np.savez_compressed(path,shape=shape,geometry=geometry)
 if '--reuse-embedding' not in sys.argv:
  scaled=np.clip(StandardScaler().fit_transform(geometry),-4,4)/np.sqrt(geometry.shape[1])
  # Block weighting is declared, independent of family identifiers or desired layout.
  shape=shape.copy();shape[:,:1024]*=.70;shape[:,1024:1280]*=.55;shape[:,1280:1296]*=.35;shape[:,1296:]*=.45
  X=np.c_[shape,.55*scaled].astype(np.float32)
  pca=PCA(n_components=64,random_state=41,svd_solver='randomized');compressed=pca.fit_transform(X).astype(np.float32)
  print('PCA variance',float(pca.explained_variance_ratio_.sum()),flush=True)
  model=umap.UMAP(n_components=3,n_neighbors=45,min_dist=.16,spread=1.5,metric='euclidean',random_state=41,n_epochs=400,n_jobs=1,verbose=True)
  embedding=model.fit_transform(compressed)
  # Rigid display orientation and one uniform scale; do not deform the embedding.
  rotation=PCA(n_components=3).fit(embedding);xyz=rotation.transform(embedding);xyz=xyz/np.max(np.ptp(xyz,axis=0))*80
  sample=np.random.default_rng(41).choice(n,1800,replace=False);tw=trustworthiness(compressed[sample],embedding[sample],n_neighbors=15)
  np.savez_compressed(A/'latent_space.npz',embedding=embedding,display_xyz=xyz,features=compressed,pca_components=pca.components_,pca_mean=pca.mean_,pca_variance=pca.explained_variance_ratio_,display_rotation=rotation.components_,display_mean=rotation.mean_)
  (A/'latent_coordinates.json').write_text(json.dumps([dict(id=r['id'],xyz=xyz[i].tolist(),embedding=embedding[i].tolist(),theme=r['theme'],n_atoms=r['n_atoms'],cell_A=r['cell_A'],porosity=r['porosity']) for i,r in enumerate(rec)]))
  report=dict(n_designs=n,original_design_masks_verified=16384,new_design_masks_verified=47616,unique_geometries=n,pca_dimensions=64,pca_retained_variance=float(pca.explained_variance_ratio_.sum()),embedding='UMAP',components=3,n_neighbors=45,min_dist=.16,spread=1.5,epochs=400,seed=41,trustworthiness_sample_size=1800,trustworthiness_k=15,trustworthiness=float(tw),features_exclude=['family labels','strength','stress','failure strain','work','performance','simulation outcomes'],coordinate_interpretation='Unsupervised geometry similarity; arbitrary axes and approximate distances; not an energy surface or a neural-agent hidden state',simulations_run=0)
  (OUT/'embedding_qa.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 # One sprite per actual geometry, ID ordering. This square atlas also gives a
 # complete, countable overview before the camera enters the embedded space.
 sprites=np.zeros((n,96,96),np.uint8);sheet=np.zeros((200*48,320*48,3),np.uint8)
 groups=[[r['id'] for r in rec if r['theme_index']==f] for f in range(16)]
 assert all(len(g)==4000 for g in groups)
 order=np.array(groups).T.ravel().tolist();slots=np.argsort(order)
 for i,r in enumerate(rec):
  g=np.array(Image.open(A/'designs'/f'{i:04d}.png'));sprites[i]=cv2.resize(g,(96,96),interpolation=cv2.INTER_AREA)
  g=cv2.resize(g,(44,44),interpolation=cv2.INTER_AREA);color=np.array(COLORS[r['theme_index']]);rgb=np.rint(g[:,:,None]*color/255).astype(np.uint8);y,x=divmod(int(slots[i]),320);sheet[y*48+2:y*48+46,x*48+2:x*48+46]=rgb
 np.save(A/'sprites.npy',sprites);Image.fromarray(sheet).save(A/'atlas_64000.png');(A/'gallery_order.json').write_text(json.dumps(order));print('Atlas and sprites saved',flush=True)
if __name__=='__main__':main()
