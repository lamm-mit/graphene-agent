"""Select representative neighbourhoods after embedding, without moving points."""
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
import fast_geometry as F
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output'
def main():
 rec=json.loads((A/'manifest.json').read_text());z=np.load(A/'latent_space.npz');X=z['features'];xyz=z['display_xyz'];tour=[]
 for f in [4,12,6,15]:
  ids=np.array([r['id'] for r in rec if r['theme_index']==f and .13<r['porosity']<.55 and r['n_pruned']/r['n_pristine']<.045 and 1500<r['n_atoms']<40000])
  assert len(ids)>5
  center=np.median(xyz[ids],axis=0);id=int(ids[np.argmin(np.linalg.norm(xyz[ids]-center,axis=1))]);d=np.linalg.norm(X-X[id],axis=1);near=np.argsort(d)[1:25]
  tour.append(dict(id=id,theme=rec[id]['theme'],xyz=xyz[id].tolist(),nearest_feature_neighbors=near.tolist(),feature_distances=d[near].tolist()))
 # Neighbour links always use input-feature nearest neighbours, not invented edges.
 (A/'tour.json').write_text(json.dumps(tour,indent=2))
 heroes=sorted(set([r['id'] for r in tour]+[r['nearest_feature_neighbors'][j] for r in tour for j in [0,2,4,7,12,20]]))
 for id in heroes:
  P,L,ng,keep=F.load_design(A/'designs'/f'{id:04d}.npz');size=1152;sc=(size-48)/max(L[:2]);xy=np.rint((P[:,:2]-L[:2]/2)*[sc,-sc]+size/2).astype(np.int32);im=np.zeros((size,size),np.uint8)
  ids=np.flatnonzero(keep);ii=np.repeat(ids,3);jj=ng[ids].ravel();ok=(jj>ii)&keep[jj]&np.all(np.abs(P[jj,:2]-P[ii,:2])<L[:2]/2,axis=1)
  cv2.polylines(im,np.stack([xy[ii[ok]],xy[jj[ok]]],axis=1),False,245,max(1,round(sc*.11)),cv2.LINE_AA)
  if sc>5:
   for p in xy[ids]:cv2.circle(im,tuple(p),max(1,round(sc*.13)),255,-1,cv2.LINE_AA)
  Image.fromarray(im).save(A/f'hero_{id:04d}.png')
 print(json.dumps(tour,indent=2),flush=True)
if __name__=='__main__':main()
