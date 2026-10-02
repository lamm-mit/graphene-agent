"""Reproduce a saved design locally using its exact lattice occupancy mask.

Usage: python reproduce.py 1234 output.extxyz
Downloads public code snapshots, mask and manifest; no dynamics or relaxation.
"""
import argparse,gzip,json,sys,tarfile
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
import numpy as np
from ase import Atoms
from ase.io import write
REPO='lamm-mit/graphene-design-universe-256k'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('index',type=int);parser.add_argument('output',type=Path);parser.add_argument('--revision',default='v1.0.0');args=parser.parse_args()
 if not 0<=args.index<256000:raise ValueError('Index outside dataset')
 if args.output.exists():raise FileExistsError(args.output)
 root=Path(snapshot_download(REPO,repo_type='dataset',revision=args.revision,allow_patterns=['code/fast_geometry.py','code/generate_atlas.py','code/source/*.py','code/source/**/*.py']))
 sys.path.insert(0,str(root/'code'));import fast_geometry as F
 from generate_atlas import digest
 manifest=json.loads(gzip.decompress(Path(hf_hub_download(REPO,'provenance/manifest.json.gz',repo_type='dataset',revision=args.revision)).read_bytes()))
 r=manifest[args.index]
 path=hf_hub_download(REPO,f'source_masks/masks-{args.index//256:04d}.tar',repo_type='dataset',revision=args.revision)
 with tarfile.open(path) as tar:
  with np.load(tar.extractfile(f'{args.index:04d}.npz')) as z:
   P,L,ng=F.lattice(*[float(v) for v in z['target']],orientation=str(z['orientation']));keep=np.unpackbits(z['keep'],count=len(P)).astype(bool)
 assert digest(P,L,keep)==r['geometry_digest']
 atoms=Atoms(numbers=np.full(int(keep.sum()),6),positions=P[keep],cell=np.diag(L),pbc=[True,True,False]);atoms.info.update(design_id=f'GDU-{args.index:06d}',length_unit='angstrom',geometry_status='unrelaxed',property_status='not_evaluated')
 write(args.output,atoms,format='extxyz');print(args.output)
if __name__=='__main__':main()
