"""New geometric motifs on a periodic graphene lattice; no physical evaluation.
All scalar fields use integer reciprocal-cell waves and are periodic in x/y.
The geometric neighbor graph is used only for deletion cleanup, never dynamics.
"""
import numpy as np
import fast_geometry as F
from source.structures import design_space as D
from generate_atlas import NAMES as BASE_NAMES,COLORS as BASE_COLORS
NAMES=BASE_NAMES+['Morphing pores','Pore–slit mixtures','Curved vein hierarchies','Branching networks','Graded orientations','Multiscale pore fields','Correlated networks','Hybrid domains']
COLORS=BASE_COLORS+[(238,183,117),(241,140,150),(84,211,200),(183,204,134),(219,169,230),(132,181,246),(227,182,102),(187,162,222)]
TAU=2*np.pi

def recipe(theme,q,seed):
 # Continuous aspect and cell-size distributions replace the former small size grid.
 # Large systems are a small planned tail, recorded explicitly in the parameters.
 scale=10**(np.log10(48)+q[0]*(np.log10(720)-np.log10(48)))
 if seed%1601==0:scale=1500+3500*q[1]
 aspect=np.exp((q[1]-.5)*1.6)
 return dict(Lx=float(scale),Ly=float(scale*aspect),orientation='zigzag' if q[2]<.5 else 'armchair',seed=int(seed),
   motif=int(theme-16),qx=[float(x) for x in q[3:]],field_version=1)

def pattern(xy,L,p):
 q=np.array(p['qx']);x=xy[:,0]/L[0];y=xy[:,1]/L[1];u=x+q[0];v=y+q[1]
 nx=2+int(q[2]*8);ny=2+int(q[3]*8);phi=TAU*q[4]
 # Distances to lattice centers in fractional motif coordinates.
 dx=(u*nx+.5)%1-.5;dy=(v*ny+.5)%1-.5
 theta=np.pi*q[5];a=np.cos(theta)*dx+np.sin(theta)*dy;b=-np.sin(theta)*dx+np.cos(theta)*dy
 exponent=.9+6.1*q[6];rx=.13+.29*q[7];ry=.10+.33*q[8]
 pore=(np.abs(a/rx)**exponent+np.abs(b/ry)**exponent)<1
 slit=(np.abs(a)<.25+.22*q[9])&(np.abs(b)<.025+.12*q[10])
 m=p['motif']
 if m==0:return ~pore
 if m==1:
  cellx=np.floor(u*nx)%nx;celly=np.floor(v*ny)%ny;mix=(np.sin(cellx*(2+q[11])+celly*(1+q[12])+phi)>.2-.4*q[13])
  return ~(np.where(mix,pore,slit))
 if m==2:
  wave=(v*(1+int(q[11]*3))+.05*np.sin(TAU*u*(1+int(q[12]*3))+phi)+.5)%1-.5
  wave2=(u*(1+int(q[13]*3))+.07*np.sin(TAU*v+phi)+.5)%1-.5
  veins=(np.abs(wave)<.035+.08*q[14])|((np.abs(wave2)<.03+.06*q[15])&(q[16]>.3))
  return (~pore)|veins
 if m==3:
  # Three families of smooth periodic branches; widths and curvatures vary.
  w=.08+.19*q[7];f1=np.sin(TAU*(nx*u)+.6*q[8]*np.sin(TAU*v));f2=np.sin(TAU*(ny*v)+.8*q[9]*np.sin(TAU*u)+phi)
  f3=np.sin(TAU*((1+int(q[10]*4))*u+(1+int(q[11]*4))*v)+phi)
  return (np.abs(f1)<w)|(np.abs(f2)<w)|(np.abs(f3)<w*(.3+.7*q[12]))
 if m==4:
  theta=np.pi*(q[5]+q[11]*np.sin(TAU*u)+q[12]*.5*np.sin(TAU*v+phi))
  aa=np.cos(theta)*dx+np.sin(theta)*dy;bb=-np.sin(theta)*dx+np.cos(theta)*dy
  size=1+(.1+.55*q[13])*np.sin(TAU*u+phi)*np.cos(TAU*v)
  return ~((np.abs(aa/(rx*size))**exponent+np.abs(bb/(ry*size))**exponent)<1)
 if m==5:
  n2=2+int(q[11]*3);xx=(u*n2+.5)%1-.5;yy=(v*n2+.5)%1-.5
  big=xx*xx+yy*yy<(.10+.21*q[12])**2
  sx=np.abs((u*(1+int(q[13]*3))+.5)%1-.5);sy=np.abs((v*(1+int(q[14]*3))+.5)%1-.5)
  support=(sx<.025+.055*q[15])|(sy<.025+.055*q[16])
  return (~(pore|big))|support
 if m==6:
  # A thresholded periodic Fourier field, with disorder controlled continuously.
  f=np.zeros_like(x)
  for j in range(6):
   kx=1+int(q[(11+j)%len(q)]*6);ky=(-1 if j%2 else 1)*(1+int(q[(17+j)%len(q)]*6))
   f+=(.4+q[(7+j)%len(q)])*np.cos(TAU*(kx*u+ky*v)+TAU*q[(21+j)%len(q)])/(1+.35*j)
  return f>-.95+.75*q[13]
 if m==7:
  domain=np.sin(TAU*((1+int(q[11]*3))*u+(1+int(q[12]*2))*v)+phi)
  cross=(np.abs((u*(1+int(q[13]*3))+.5)%1-.5)<.04+.08*q[14])
  return (~np.where(domain>0,pore,slit))|cross
 raise ValueError(m)

def generate(p):
 at=F.sheet(p['Lx'],p['Ly'],orientation=p['orientation']);P,L,ng=F.lattice(*at.info['_lattice_key'])
 keep=pattern(P[:,:2],L,p);initial=int(keep.sum());at=at[keep];at=F.prune(at)
 active=np.zeros(len(P),bool);active[at.arrays['site_id']]=True
 return P,L,ng,active,int(initial-active.sum())
