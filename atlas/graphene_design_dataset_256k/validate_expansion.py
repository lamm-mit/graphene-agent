"""Independent release checks beyond generation and coordinate packaging."""
from pathlib import Path
import json,copy,gzip,hashlib,io,tarfile,collections
import numpy as np
import pyarrow.parquet as pq
from ase.io import read
from source.structures import design_space as D
import fast_geometry as F
from hybrid_geometry import generate,NAMES
from generate_atlas import digest
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'graphene_design_dataset_64k';A=ROOT/'assets';OUT=ROOT/'output';REL=ROOT/'release/dataset'
def main():
 F.install();records=json.loads((A/'manifest.json').read_text());prior=json.loads((OLD/'assets/manifest.json').read_text());assert records[:64000]==prior
 assert len(records)==256000 and len({r['geometry_digest'] for r in records})==256000
 tests=[]
 for f in range(24):
  ids=[r['id'] for r in records if r['theme_index']==f and r['id']>=64000];chosen=[ids[0],ids[len(ids)//2],ids[-1]]
  for i in chosen:
   r=records[i];p=r['params']
   if r['family']=='periodic_hybrid':P,L,ng,keep,pruned=generate(p)
   else:
    at=D.generate(r['family'],**copy.deepcopy(p));P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True
   assert digest(P,L,keep)==r['geometry_digest'];tests.append(dict(id=i,theme=f,recipe_reproduces_exact_mask=True))
 (ROOT/'validation/parameter_regeneration.json').write_text(json.dumps(tests,indent=2))
 repair=json.loads((ROOT/'validation/uniqueness_repair/repair_report.json').read_text())
 for i in repair['ids']:
  rec=records[i];at=D.generate(rec['family'],**copy.deepcopy(rec['params']));P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True;assert digest(P,L,keep)==rec['geometry_digest']
 # Independent ASE reading of a sample from each group and global extrema.
 sample={t['id'] for t in tests}|{min(records,key=lambda r:r['n_atoms'])['id'],0,63999,255999};ase=[]
 for i in sorted(sample):
  row=pq.read_table(REL/'data'/f'designs-{i//256:04d}.parquet').to_pylist()[i%256]
  with open(REL/row['coordinate_shard'],'rb') as f:f.seek(row['coordinate_offset']);raw=gzip.decompress(f.read(row['coordinate_bytes']))
  assert hashlib.sha256(raw).hexdigest()==row['coordinates_sha256'];at=read(io.StringIO(raw.decode()),format='extxyz');P,L,ng,keep=F.load_design(A/'designs'/f'{i:04d}.npz')
  assert np.array_equal(at.positions,P[keep]) and np.array_equal(np.asarray(at.cell),np.diag(L)) and np.array_equal(at.pbc,[1,1,0]);ase.append(i)
 stats=dict(total_designs=256000,unique_coordinate_realizations=256000,prior_manifest_exact=True,groups=dict(collections.Counter(r['theme'] for r in records)),parameter_regeneration_checks=len(tests),uniqueness_repair_regeneration_checks=len(repair['ids']),additional_independent_ase_checks=ase,simulations_run=0)
 (OUT/'population_qa.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
