"""Assemble 256K release data, source code, mechanics interfaces and explorer."""
from pathlib import Path
import gzip,hashlib,json,shutil,tarfile,collections,importlib.metadata
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hybrid_geometry import COLORS,NAMES
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';D=ROOT/'release/dataset';S=ROOT/'release/space';OUT=ROOT/'output';OLD=ROOT.parent/'graphene_design_dataset_64k'
def main():
 records=json.loads((A/'manifest.json').read_text());z=np.load(A/'latent_space.npz');xyz=z['display_xyz'];assert len(records)==len(xyz)==256000
 cols=['index','design_id','coordinate_offset','coordinate_bytes','image_offset','image_bytes','coordinates_sha256','geometry_sha256','n_atoms','cell_angstrom','pbc','periodic_winding_rank','periodic_winding_x','periodic_winding_y','periodic_components','fraction_coordination_lt2','geometric_cleanup_flag','theme','release_cohort','property_status']
 table=pq.read_table(D/'data',columns=cols);meta=sorted(table.to_pylist(),key=lambda r:r['index']);rows=[]
 assert len(meta)==256000 and [x['index'] for x in meta]==list(range(256000))
 for i,(r,m) in enumerate(zip(records,meta)):
  assert i==r['id']==m['index'] and r['geometry_digest']==m['geometry_sha256'] and r['n_atoms']==m['n_atoms']
  rows.append([i,r['theme_index'],r['n_atoms'],r['cell_A'][0]/10,r['cell_A'][1]/10,r['porosity'],*[float(x) for x in xyz[i]],m['coordinate_offset'],m['coordinate_bytes'],m['image_offset'],m['image_bytes'],m['coordinates_sha256'],m['periodic_winding_rank'],m['geometric_cleanup_flag']])
 (S/'designs.json.gz').write_bytes(gzip.compress(json.dumps(dict(groups=NAMES,colors=['rgb'+str(tuple(c)) for c in COLORS],rows=rows),separators=(',',':')).encode(),compresslevel=6,mtime=0))
 columns={'design_id':[f'GDU-{i:06d}' for i in range(256000)],'index':list(range(256000)),'release_cohort':[m['release_cohort'] for m in meta]}
 for k in range(3):columns[f'umap_{k+1}']=z['embedding'][:,k];columns[f'display_{k+1}']=xyz[:,k]
 pq.write_table(pa.table(columns),D/'embedding/coordinates.parquet',compression='zstd')
 for name in ['latent_space.npz','geometric_descriptors.npz','selection_model.npz','heldout_targets.npz','bridge_coverage.npz','selection_features.npy']:shutil.copy2(A/name,D/'embedding'/name)
 (D/'provenance/manifest.json.gz').write_bytes(gzip.compress((A/'manifest.json').read_bytes(),mtime=0))
 masks=D/'source_masks';masks.mkdir(exist_ok=True)
 for si in range(1000):
  path=masks/f'masks-{si:04d}.tar'
  if path.exists():continue
  if si<250:shutil.copy2(OLD/'release/dataset/source_masks'/path.name,path);continue
  with tarfile.open(path,'w',format=tarfile.USTAR_FORMAT) as tar:
   for i in range(si*256,(si+1)*256):
    src=A/'designs'/f'{i:04d}.npz';info=tar.gettarinfo(src,arcname=src.name);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
    with open(src,'rb') as f:tar.addfile(info,f)
  if si%100==0:print('Source shards',si,flush=True)
 reports=[json.loads(p.read_text()) for p in sorted((ROOT/'validation').glob('shard-*.json'))];assert len(reports)==1000
 qa=dict(designs=sum(x['designs'] for x in reports),atoms=sum(x['atoms'] for x in reports),shards=1000,coordinate_bytes=sum(x['coordinate_bytes'] for x in reports),catalog_bytes=sum(x['catalog_bytes'] for x in reports),all_coordinate_text_roundtrips_exact=all(x['all_coordinate_text_roundtrips_exact'] for x in reports),all_geometry_digests_match=all(x['all_geometry_digests_match'] for x in reports),ase_roundtrip_designs=sum(len(x['ase_roundtrip_ids']) for x in reports),ase_max_coordinate_error_A=max(x['ase_max_coordinate_error_A'] for x in reports),preserved_64k_coordinate_shards_byte_identical=all(x.get('prior_coordinates_reused_exact') for x in reports[:250]),simulations_run=0)
 assert qa['designs']==256000 and qa['all_coordinate_text_roundtrips_exact'] and qa['all_geometry_digests_match'] and qa['preserved_64k_coordinate_shards_byte_identical']
 (OUT/'dataset_qa.json').write_text(json.dumps(qa,indent=2));(D/'provenance/coordinate_shards.json').write_text(json.dumps(reports,indent=2))
 quality=dict(periodic_winding_rank_counts=dict(collections.Counter(m['periodic_winding_rank'] for m in meta)),low_coordination_flag_count=sum(m['geometric_cleanup_flag'] for m in meta),periodic_component_counts=dict(collections.Counter(m['periodic_components'] for m in meta)),interpretation='Geometric neighbor graph only; no force-field evaluation or stability claim')
 (OUT/'geometry_quality.json').write_text(json.dumps(quality,indent=2))
 pq.write_table(table.drop(['coordinate_offset','coordinate_bytes','image_offset','image_bytes']),D/'mechanics/design_inventory.parquet',compression='zstd')
 for p in OUT.glob('*.json'):
  if p.name in ['generation_summary.json','embedding_qa.json','coverage_qa.json','selection_model.json','dataset_qa.json','geometry_quality.json','selection_chunks.json','population_qa.json']:shutil.copy2(p,D/'provenance'/p.name)
 for name in ['preflight.json','parameter_regeneration.json']:
  if (ROOT/'validation'/name).exists():shutil.copy2(ROOT/'validation'/name,D/'provenance'/name)
 shutil.copy2(ROOT/'validation/uniqueness_repair/repair_report.json',D/'provenance/uniqueness_repair.json')
 shutil.copy2(ROOT/'planning/expansion_protocol.json',D/'provenance/expansion_protocol.json')
 code=D/'code';code.mkdir(exist_ok=True)
 for name in ['fast_geometry.py','generate_atlas.py','hybrid_geometry.py','expand_diverse.py','embed_designs.py','embed_expanded.py','geometry_quality.py','prepare_tour.py','render_latent.py','finish_delivery.py','export_geometry.py','package_dataset.py','package_stream.py','build_release.py','validate_expansion.py','coverage_figures.py','repair_uniqueness.py','consolidate_generation.py']:
  shutil.copy2(ROOT/name,code/name)
 shutil.copytree(ROOT/'source',code/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 packages=['numpy','scipy','pillow','opencv-python','ase','scikit-learn','umap-learn','numba','pyarrow','datasets','huggingface-hub','jsonschema'];versions={p:importlib.metadata.version(p) for p in packages}
 (code/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
 release=dict(designs=256000,metadata_rows=len(meta),image_members=256000,coordinate_members=256000,source_masks=256000,embedding_finite=bool(np.isfinite(xyz).all()),unique_geometries=len({r['geometry_digest'] for r in records}),library_versions=versions,published_properties=[],license='Apache-2.0',visibility='public')
 (OUT/'release_qa.json').write_text(json.dumps(release,indent=2));(D/'provenance/release_qa.json').write_text(json.dumps(release,indent=2));print(json.dumps(qa,indent=2));print(json.dumps(quality,indent=2))
if __name__=='__main__':main()
