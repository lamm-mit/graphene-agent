"""Exact periodic graph winding on retained pristine carbon neighbors.
These geometric tests neither evaluate force balance nor predict failure.
"""
import numpy as np
from numba import njit
@njit(cache=True)
def winding(P,L,ng,keep):
 n=len(keep);seen=np.zeros(n,np.uint8);ix=np.zeros(n,np.int32);iy=np.zeros(n,np.int32);queue=np.empty(n,np.int32)
 components=0;rank=0;bx=0;by=0;wx=False;wy=False
 for root in range(n):
  if not keep[root] or seen[root]:continue
  components+=1;seen[root]=1;queue[0]=root;front=0;end=1
  while front<end:
   i=queue[front];front+=1
   for k in range(3):
    j=ng[i,k]
    if not keep[j]:continue
    xx=ix[i]-int(np.rint((P[j,0]-P[i,0])/L[0]));yy=iy[i]-int(np.rint((P[j,1]-P[i,1])/L[1]))
    if not seen[j]:seen[j]=1;ix[j]=xx;iy[j]=yy;queue[end]=j;end+=1
    else:
     dx=xx-ix[j];dy=yy-iy[j]
     if dx!=0 or dy!=0:
      wx=wx or dx!=0;wy=wy or dy!=0
      if rank==0:rank=1;bx=dx;by=dy
      elif rank==1 and bx*dy-by*dx!=0:rank=2
 return components,rank,wx,wy

def quality(P,L,ng,keep):
 degree=keep[ng[np.flatnonzero(keep)]].sum(axis=1);comp,rank,wx,wy=winding(P,L,ng,keep)
 return dict(fraction_coordination_2=float(np.mean(degree==2)),fraction_coordination_lt2=float(np.mean(degree<2)),geometric_cleanup_flag=bool(np.any(degree<2)),periodic_components=int(comp),periodic_winding_rank=int(rank),periodic_winding_x=bool(wx),periodic_winding_y=bool(wy))
