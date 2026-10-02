"""Population, preservation and export-coverage audit of the complete release."""
from pathlib import Path
import collections,gzip,hashlib,io,json
import numpy as np
import pyarrow.parquet as pq
from ase.io import read
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';D=ROOT/'release/dataset';OUT=ROOT/'output'
def main():
 r=json.loads((A/'manifest.json').read_text());assert [x['id'] for x in r]==list(range(64000))
 assert len({x['geometry_digest'] for x in r})==64000
 counts=collections.Counter(x['theme'] for x in r);assert len(counts)==16 and set(counts.values())=={4000}
 old=ROOT.parent/'graphene_latent_universe_4k';prior=json.loads((old/'assets/manifest.json').read_text());assert r[:16384]==prior
 for x in prior:
  name=f'{x["id"]:04d}.npz';assert (A/'designs'/name).read_bytes()==(old/'assets/designs'/name).read_bytes()
 table=pq.read_table(D/'data',columns=['index','design_id','n_atoms','cell_angstrom','pbc','coordinate_shard','coordinate_offset','coordinate_bytes','coordinates_sha256','image_shard','image_offset','image_bytes']);rows=sorted(table.to_pylist(),key=lambda x:x['index']);assert len(rows)==64000
 for i,(source,row) in enumerate(zip(r,rows)):
  assert i==row['index'] and source['n_atoms']==row['n_atoms']
  assert np.array_equal(np.diag(source['cell_A']),row['cell_angstrom']) and row['pbc']==[True,True,False]
 # Independently read global extremes with ASE (in addition to shard samples).
 ids={min(r,key=lambda x:x['n_atoms'])['id'],max(r,key=lambda x:x['n_atoms'])['id'],63999};checks=[]
 for i in sorted(ids):
  row=rows[i]
  with open(D/row['coordinate_shard'],'rb') as f:f.seek(row['coordinate_offset']);raw=gzip.decompress(f.read(row['coordinate_bytes']))
  assert hashlib.sha256(raw).hexdigest()==row['coordinates_sha256']
  at=read(io.StringIO(raw.decode()),format='extxyz');assert len(at)==r[i]['n_atoms'] and np.all(at.numbers==6)
  assert np.array_equal(at.pbc,[True,True,False]) and np.allclose(at.cell,np.diag(r[i]['cell_A']),rtol=0,atol=1e-12)
  assert np.all(at.positions[:,:2]>=-1e-12) and np.all(at.positions[:,:2]<np.asarray(r[i]['cell_A'][:2])+1e-12)
  checks.append(dict(index=i,n_atoms=len(at),ase_read_passed=True))
 geom=np.load(A/'geometric_descriptors.npz')['geometry']
 report=dict(designs=64000,design_group_counts=dict(counts),generator_counts=dict(collections.Counter(x['family'] for x in r)),
  atom_count_range=[min(x['n_atoms'] for x in r),max(x['n_atoms'] for x in r)],total_atoms=sum(x['n_atoms'] for x in r),
  width_range_nm=[min(x['cell_A'][0]/10 for x in r),max(x['cell_A'][0]/10 for x in r)],
  porosity_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile([x['porosity'] for x in r],[0,.25,.5,.75,1]).tolist())),
  designs_with_some_retained_sites_having_fewer_than_two_graph_neighbors=int(np.sum(geom[:,5]>0)),
  previous_16384_manifest_exact=True,previous_16384_site_masks_byte_identical=True,all_catalog_cells_and_pbc_match=True,
  independent_ase_extremes=checks,property_evaluations=0)
 (OUT/'population_qa.json').write_text(json.dumps(report,indent=2));(D/'provenance/population_qa.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
