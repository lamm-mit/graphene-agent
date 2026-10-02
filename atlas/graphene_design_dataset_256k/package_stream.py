"""Package one generation cohort at a time; reuse the prior coordinate bytes."""
import concurrent.futures as cf
import json,time,shutil
import pyarrow as pa
import pyarrow.parquet as pq
from package_dataset import *
OLD=ROOT.parent/'graphene_design_dataset_64k'
def preserved(si):
 audit=ROOT/'validation'/f'shard-{si:04d}.json'
 if audit.exists():return json.loads(audit.read_text())
 rows=pq.read_table(OLD/'release/dataset/data'/f'designs-{si:04d}.parquet').to_pylist()
 for row in rows:
  i=row['index'];P,L,ng,keep=F.load_design(A/'designs'/f'{i:04d}.npz');assert int(keep.sum())==row['n_atoms'];assert digest(P,L,keep)==row['geometry_sha256']
  row.update(quality(P,L,ng,keep));row.update(release_cohort='preserved_64k',selection_strategy='preserved',baseline_neighbor_id=None,baseline_descriptor_distance=None,bridge_target_id=None,bridge_descriptor_distance=None)
 for kind,prefix in [('coordinates','structures'),('images','previews')]:
  src=OLD/'release/dataset'/kind/f'{prefix}-{si:04d}.tar';dst=RELEASE/kind/src.name
  if not dst.exists():shutil.copy2(src,dst)
 pq.write_table(pa.Table.from_pylist(rows,schema=FIELDS.arrow_schema),RELEASE/'data'/f'designs-{si:04d}.parquet',compression='zstd',row_group_size=64)
 report=json.loads((OLD/'validation'/f'shard-{si:04d}.json').read_text());report['prior_coordinates_reused_exact']=True;report['new_periodic_graph_checks']=True
 assert hashlib.file_digest(open(RELEASE/'coordinates'/f'structures-{si:04d}.tar','rb'),'sha256').hexdigest()==report['sha256_tar']
 report['catalog_bytes']=(RELEASE/'data'/f'designs-{si:04d}.parquet').stat().st_size;audit.write_text(json.dumps(report,indent=2));return report

def run(si):
 if si<250:return preserved(si)
 rows=[json.loads((A/'designs'/f'{i:04d}.json').read_text()) for i in range(si*256,(si+1)*256)]
 samples={rows[0]['id'],min(rows,key=lambda r:r['n_atoms'])['id'],max(rows,key=lambda r:r['n_atoms'])['id']}
 return package_one((si,rows,samples))
def main():
 for d in ['coordinates','images','data','tools','provenance','embedding']:(RELEASE/d).mkdir(exist_ok=True)
 t=time.time();pending=[];done=0
 with cf.ProcessPoolExecutor(max_workers=3) as pool:
  for si in range(1000):
   while len(pending)>=6:
    finished,unfinished=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
    for f in finished:f.result();done+=1
    pending=list(unfinished)
   while not all((A/'designs'/f'{i:04d}.json').exists() for i in range(si*256,(si+1)*256)):
    for f in pending:
     if f.done():f.result()
    time.sleep(3)
   pending.append(pool.submit(run,si))
   if si%16==0:
    status=dict(submitted_shards=si+1,completed_at_least=done,elapsed_s=time.time()-t);(OUT/'packaging_progress.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
  for f in pending:f.result()
 print('All 1000 shards completed',flush=True)
if __name__=='__main__':main()
