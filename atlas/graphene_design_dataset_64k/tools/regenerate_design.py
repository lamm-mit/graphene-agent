"""Regenerate a geometry from recorded generator parameters and verify its digest.

Usage: python regenerate_design.py 1234 output.extxyz
For exact stored-site retrieval use load_structure.py or reproduce.py instead.
"""
import argparse,copy,gzip,json,sys
from pathlib import Path
import numpy as np
from ase.io import write
from huggingface_hub import snapshot_download,hf_hub_download
REPO='lamm-mit/graphene-design-universe-64k'
def main():
 p=argparse.ArgumentParser();p.add_argument('index',type=int);p.add_argument('output',type=Path);p.add_argument('--revision',default='v1.0.0');args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 root=Path(snapshot_download(REPO,repo_type='dataset',revision=args.revision,allow_patterns=['code/fast_geometry.py','code/generate_atlas.py','code/source/**']))
 sys.path.insert(0,str(root/'code'));import fast_geometry as F
 from generate_atlas import digest
 from source.structures import design_space as D
 F.install();path=hf_hub_download(REPO,'provenance/manifest.json.gz',repo_type='dataset',revision=args.revision)
 r=json.loads(gzip.decompress(Path(path).read_bytes()))[args.index]
 at=D.generate(r['family'],**copy.deepcopy(r['params']));P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True
 if digest(P,L,keep)!=r['geometry_digest']:raise RuntimeError('Regeneration differs from the stored geometry; use the saved coordinates and investigate runtime differences')
 # Internal generator bookkeeping is excluded from the portable extxyz header.
 at.info=dict(design_id=f'GDU-{args.index:06d}',length_unit='angstrom',geometry_status='unrelaxed',property_status='not_evaluated')
 write(args.output,at,format='extxyz');print('Digest matched:',args.output)
if __name__=='__main__':main()
