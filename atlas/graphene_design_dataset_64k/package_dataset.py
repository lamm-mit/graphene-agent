"""Release builder: viewer Parquet + indexed tar shards of standard extxyz.gz.

Every exported coordinate is parsed back and compared with the saved geometry.
ASE performs independent format checks on a stratified sample and size extremes.
No property evaluation or coordinate relaxation occurs.
"""
from pathlib import Path
import concurrent.futures as cf
import gzip,hashlib,io,json,math,os,tarfile,time
import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq
from PIL import Image
from datasets import Features,Image as HFImage,Value,Sequence
from ase.io import read
import fast_geometry as F
from generate_atlas import digest,thumbnail

ROOT=Path(__file__).resolve().parent; A=ROOT/'assets'; RELEASE=ROOT/'release/dataset'; OUT=ROOT/'output'
SHARD=256
FIELDS=Features({
 'design_id':Value('string'),'index':Value('int32'),'image':HFImage(),
 'theme':Value('string'),'theme_index':Value('int8'),'generator':Value('string'),
 'n_atoms':Value('int64'),'n_pristine':Value('int64'),'n_pruned':Value('int64'),
 'cell_angstrom':Sequence(Sequence(Value('float64'),length=3),length=3),
 'pbc':Sequence(Value('bool'),length=3),'species':Value('string'),
 'length_unit':Value('string'),'geometry_status':Value('string'),
 'porosity':Value('float64'),'width_nm':Value('float64'),'height_nm':Value('float64'),
 'orientation':Value('string'),'parameters_json':Value('string'),
 'geometry_sha256':Value('string'),'coordinates_sha256':Value('string'),
 'coordinate_shard':Value('string'),'coordinate_member':Value('string'),
 'coordinate_offset':Value('int64'),'coordinate_bytes':Value('int64'),
 'image_shard':Value('string'),'image_offset':Value('int64'),'image_bytes':Value('int64'),
 'property_status':Value('string')})

def extxyz(r,P,L,keep):
 Q=np.ascontiguousarray(P[keep]); n=len(Q)
 lattice=' '.join(format(x,'.17g') for x in np.diag(L).ravel())
 header=(f'{n}\nLattice="{lattice}" Properties=species:S:1:pos:R:3 '
         f'pbc="T T F" design_id="GDU-{r["id"]:06d}" length_unit=angstrom '
         'geometry_status=unrelaxed property_status=not_evaluated\n').encode()
 sink=pa.BufferOutputStream()
 pc.write_csv(pa.table({'species':pa.array(np.full(n,'C')),'x':Q[:,0],'y':Q[:,1],'z':Q[:,2]}),sink,
              write_options=pc.WriteOptions(include_header=False,delimiter=' ',quoting_style='none',batch_size=65536))
 raw=header+sink.getvalue().to_pybytes()
 # Independent text parsing of every exported atom, including the largest designs.
 table=pc.read_csv(pa.BufferReader(raw[len(header):]),
   read_options=pc.ReadOptions(column_names=['species','x','y','z'],use_threads=False),
   parse_options=pc.ParseOptions(delimiter=' '),
   convert_options=pc.ConvertOptions(column_types={'species':pa.string(),'x':pa.float64(),'y':pa.float64(),'z':pa.float64()}))
 restored=np.column_stack([table[k].to_numpy() for k in ['x','y','z']])
 assert len(restored)==n and np.array_equal(restored,Q)
 assert set(table['species'].unique().to_pylist())=={'C'}
 return raw,Q

def package_one(args):
 si,rows,sample_ids=args
 target=RELEASE/'coordinates'/f'structures-{si:04d}.tar'
 catalog=RELEASE/'data'/f'designs-{si:04d}.parquet'; audit=ROOT/'validation'/f'shard-{si:04d}.json'
 if target.exists() and catalog.exists() and audit.exists():return json.loads(audit.read_text())
 assert not target.exists() and not catalog.exists(),f'Incomplete published shard {si}; preserve and inspect before retry'
 part=target.with_suffix('.tar.partial'); out=[]; ase_checks=[]; atoms=0; max_error=0
 image_path=RELEASE/'images'/f'previews-{si:04d}.tar'
 with tarfile.open(part,'w',format=tarfile.USTAR_FORMAT) as tar, tarfile.open(image_path.with_suffix('.tar.partial'),'w',format=tarfile.USTAR_FORMAT) as image_tar:
  for r in rows:
   id=r['id'];P,L,ng,keep=F.load_design(A/'designs'/f'{id:04d}.npz')
   assert int(keep.sum())==r['n_atoms'] and np.array_equal(L,r['cell_A'])
   assert digest(P,L,keep)==r['geometry_digest']
   assert np.isfinite(P).all() and np.all(L>0)
   raw,Q=extxyz(r,P,L,keep); compressed=gzip.compress(raw,compresslevel=1,mtime=0)
   assert gzip.decompress(compressed)==raw
   if id in sample_ids:
    at=read(io.StringIO(raw.decode()),format='extxyz')
    err=float(np.max(np.abs(at.positions-Q)));max_error=max(max_error,err)
    assert err<1e-12 and np.allclose(at.cell,np.diag(L),rtol=0,atol=1e-12)
    assert np.array_equal(at.pbc,[True,True,False]) and np.all(at.numbers==6)
    assert at.info['design_id']==f'GDU-{id:06d}'
    ase_checks.append(id)
   member=f'GDU-{id:06d}.extxyz.gz';info=tarfile.TarInfo(member);info.size=len(compressed);info.mode=0o644;info.mtime=0
   offset=tar.offset+512;tar.addfile(info,io.BytesIO(compressed))
   im=Image.fromarray(thumbnail(P,L,ng,keep,size=384));buffer=io.BytesIO();im.save(buffer,format='PNG')
   image_data=buffer.getvalue();ii=tarfile.TarInfo(f'GDU-{id:06d}.png');ii.size=len(image_data);ii.mode=0o644;ii.mtime=0
   image_offset=image_tar.offset+512;image_tar.addfile(ii,io.BytesIO(image_data))
   out.append(dict(image_shard='images/'+image_path.name,image_offset=image_offset,image_bytes=len(image_data),design_id=f'GDU-{id:06d}',index=id,image={'bytes':buffer.getvalue(),'path':f'GDU-{id:06d}.png'},
     theme=r['theme'],theme_index=r['theme_index'],generator=r['family'],n_atoms=len(Q),n_pristine=r['n_pristine'],n_pruned=r['n_pruned'],
     cell_angstrom=np.diag(L).tolist(),pbc=[True,True,False],species='C',length_unit='angstrom',geometry_status='unrelaxed',
     porosity=r['porosity'],width_nm=L[0]/10,height_nm=L[1]/10,orientation=r['params']['orientation'],parameters_json=json.dumps(r['params'],sort_keys=True),
     geometry_sha256=r['geometry_digest'],coordinates_sha256=hashlib.sha256(raw).hexdigest(),
     coordinate_shard='coordinates/'+target.name,coordinate_member=member,coordinate_offset=offset,coordinate_bytes=len(compressed),property_status='not_evaluated'))
   atoms+=len(Q)
 table=pa.Table.from_pylist(out,schema=FIELDS.arrow_schema)
 pq.write_table(table,catalog.with_suffix('.parquet.partial'),compression='zstd',row_group_size=64,write_page_index=True)
 # Verify tar offsets and gzip members before committing either output.
 with open(part,'rb') as f:
  for row in out:
   f.seek(row['coordinate_offset']);data=f.read(row['coordinate_bytes']);assert data[:2]==b'\x1f\x8b'
 with tarfile.open(part) as tar:
  assert len(tar.getmembers())==len(rows)
  for row,member in zip(out,tar.getmembers()):
   assert member.name==row['coordinate_member'] and member.offset_data==row['coordinate_offset']
 image_path.with_suffix('.tar.partial').rename(image_path)
 part.rename(target);catalog.with_suffix('.parquet.partial').rename(catalog)
 report=dict(shard=si,designs=len(rows),atoms=atoms,all_coordinate_text_roundtrips_exact=True,all_geometry_digests_match=True,
   ase_roundtrip_ids=ase_checks,ase_max_coordinate_error_A=max_error,coordinate_bytes=target.stat().st_size,catalog_bytes=catalog.stat().st_size,
   sha256_tar=hashlib.file_digest(open(target,'rb'),'sha256').hexdigest())
 audit.write_text(json.dumps(report,indent=2));return report

def main():
 for d in ['coordinates','images','data','tools','provenance','embedding']:(RELEASE/d).mkdir(exist_ok=True)
 rows=json.loads((A/'manifest.json').read_text());assert len(rows)==64000 and [r['id'] for r in rows]==list(range(64000))
 # Deterministic coverage: every shard, all design groups, largest and smallest.
 samples=set(range(0,64000,256))|{max(rows,key=lambda x:x['n_atoms'])['id'],min(rows,key=lambda x:x['n_atoms'])['id']}
 jobs=[(i,rows[i*SHARD:(i+1)*SHARD],samples) for i in range(math.ceil(len(rows)/SHARD))]; reports=[];t=time.time()
 with cf.ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(package_one,jobs):
   reports.append(r);print(f'Packaged {len(reports)*SHARD}/64000; {time.time()-t:.1f}s',flush=True)
 summary=dict(designs=sum(x['designs'] for x in reports),atoms=sum(x['atoms'] for x in reports),shards=len(reports),
  coordinate_bytes=sum(x['coordinate_bytes'] for x in reports),catalog_bytes=sum(x['catalog_bytes'] for x in reports),
  all_coordinate_text_roundtrips_exact=True,all_tar_offsets_verified=True,ase_roundtrip_designs=sum(len(x['ase_roundtrip_ids']) for x in reports),
  ase_max_coordinate_error_A=max(x['ase_max_coordinate_error_A'] for x in reports),simulations_run=0)
 (OUT/'dataset_qa.json').write_text(json.dumps(summary,indent=2));(RELEASE/'provenance/dataset_qa.json').write_text(json.dumps(summary,indent=2))
 (RELEASE/'provenance/coordinate_shards.json').write_text(json.dumps(reports,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
