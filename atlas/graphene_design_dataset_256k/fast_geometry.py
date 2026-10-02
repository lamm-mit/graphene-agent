"""Geometry-only acceleration of the preserved generators, without changing masks.

The pristine lattice gives an exact three-neighbour graph. Deletion/pruning uses
that graph instead of rebuilding an ASE neighbour list after each deletion.
Spatial queries restrict the original shape equations to a bounding circle.
"""
from functools import lru_cache
import math
import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from source.structures import graphene as G
from source.structures import design_space as D
from source.structures import composite as C

@lru_cache(maxsize=3)
def lattice(Lx,Ly,a=G.A0_REBO2,orientation='zigzag',vacuum=10.):
    u,c=G.rectangular_unit_cell(a,orientation)
    nx=max(1,round(Lx/c[0,0]));ny=max(1,round(Ly/c[1,1]))
    ix,iy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
    shifts=np.c_[ix.ravel()*c[0,0],iy.ravel()*c[1,1],np.zeros(nx*ny)]
    P=(shifts[:,None,:]+u[None,:,:]).reshape(-1,3);P[:,2]=vacuum
    L=np.array([nx*c[0,0],ny*c[1,1],2*vacuum])
    neighbours=np.empty((len(P),3),np.int32)
    for k in range(4):
        candidates=[]
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for j in range(4):
                    dist=np.linalg.norm(u[j]+[dx*c[0,0],dy*c[1,1],0]-u[k])
                    if 1e-8<dist<1.9:candidates.append((dx,dy,j))
        assert len(candidates)==3
        for j,(dx,dy,atom) in enumerate(candidates):
            neighbours[k::4,j]=(((ix.ravel()+dx)%nx)*ny+(iy.ravel()+dy)%ny)*4+atom
    return P,L,neighbours

def sheet(Lx_target,Ly_target,a=G.A0_REBO2,orientation='zigzag',vacuum=10.):
    P,L,ng=lattice(Lx_target,Ly_target,a,orientation,vacuum)
    at=Atoms(numbers=np.full(len(P),6),positions=P,cell=np.diag(L),pbc=[True,True,False])
    at.new_array('site_id',np.arange(len(P),dtype=np.int32))
    at.info.update(orientation_x=orientation,lattice_constant=a,_lattice_key=(Lx_target,Ly_target,a,orientation,vacuum))
    return at

def prune(atoms,r_bond=1.9,min_coord=2,iterations=10):
    assert r_bond==1.9
    P,L,ng=lattice(*atoms.info['_lattice_key'])
    active=np.zeros(len(P),bool);active[atoms.arrays['site_id']]=True
    initial=active.sum()
    for _ in range(iterations):
        bad=active & (active[ng].sum(axis=1)<min_coord)
        if not bad.any():break
        active[bad]=False
    ids=np.flatnonzero(active)
    if len(ids):
        inv=np.full(len(P),-1,np.int32);inv[ids]=np.arange(len(ids),dtype=np.int32)
        neighbours=inv[ng[ids]];ii=np.repeat(np.arange(len(ids),dtype=np.int32),3);jj=neighbours.ravel();ok=jj>=0
        graph=coo_matrix((np.ones(ok.sum(),np.uint8),(ii[ok],jj[ok])),shape=(len(ids),len(ids))).tocsr()
        n,labels=connected_components(graph,directed=False)
        if n>1:active[ids[labels!=np.argmax(np.bincount(labels))]]=False
    out=atoms[active[atoms.arrays['site_id']]];out.info=dict(atoms.info);out.info['n_pruned']=int(initial-active.sum())
    return out

_tree=None;_tree_positions=None
def in_shape(P,centers,shape,size,aspect=1.,angle=0.,L=None):
    global _tree,_tree_positions
    if _tree_positions is not P:
        _tree=cKDTree(P[:,:2]%L,boxsize=L);_tree_positions=P
    sizes=np.broadcast_to(size,(len(centers),));mask=np.zeros(len(P),bool);ca,sa=math.cos(angle),math.sin(angle)
    for c,s in zip(centers,sizes):
        if shape=='circle':r=s/2*max(1,aspect)
        elif shape=='square':r=s/2*math.sqrt(1+aspect*aspect)
        elif shape=='hexagon':r=s/math.sqrt(3)
        elif shape=='slit':r=math.hypot(s/2,aspect/2)
        else:raise ValueError(shape)
        ids=np.asarray(_tree.query_ball_point(np.asarray(c)%L,r+1e-9),dtype=np.int32)
        if not len(ids):continue
        dx=D._wrap_delta(P[ids,0]-c[0],L[0]);dy=D._wrap_delta(P[ids,1]-c[1],L[1]);u=ca*dx+sa*dy;v=-sa*dx+ca*dy
        if shape=='circle':m=(u*u+(v/aspect)**2)<(s/2)**2
        elif shape=='square':m=(np.abs(u)<s/2)&(np.abs(v)<s*aspect/2)
        elif shape=='slit':m=(np.abs(u)<s/2)&(np.abs(v)<aspect/2)
        else:
            r=s/2;m=(np.abs(v)<r)&(np.abs(.5*np.abs(v)+math.sqrt(3)/2*np.abs(u))<r)&(np.abs(u)<r*2/math.sqrt(3))
        mask[ids[m]]=True
    return mask

def install():
    D.graphene_sheet_for_size=sheet;C.graphene_sheet_for_size=sheet
    D._prune_dangling=prune;D._in_shape=in_shape

def load_design(path):
    with np.load(path) as z:
        P,L,ng=lattice(float(z['target'][0]),float(z['target'][1]),orientation=str(z['orientation']))
        keep=np.unpackbits(z['keep'],count=len(P)).astype(bool)
    return P,L,ng,keep
