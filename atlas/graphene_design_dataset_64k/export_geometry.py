"""Export any saved atlas geometry to XYZ or extxyz with its real cell and PBC.
Example: python export_geometry.py 4095 output/subvein_350nm.extxyz
No regeneration or simulation is needed; occupancy masks preserve exact sites.
"""
from pathlib import Path
import argparse,json
import numpy as np
from ase import Atoms
from ase.io import write
import fast_geometry as F
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('id',type=int);p.add_argument('output',type=Path);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 r=json.loads((ROOT/'assets/designs'/f'{args.id:04d}.json').read_text());P,L,ng,keep=F.load_design(ROOT/'assets/designs'/f'{args.id:04d}.npz')
 at=Atoms(numbers=np.full(keep.sum(),6),positions=P[keep],cell=np.diag(L),pbc=[True,True,False]);at.info['design']=r['params'];at.info['geometry_only']=True
 write(args.output,at);print(args.output)
if __name__=='__main__':main()
