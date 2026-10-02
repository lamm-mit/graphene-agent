"""Replace globally repeated new geometries, preserving every prior artifact.

Run only after generation and packaging have exited. No original 64K design is
modified. Select from 16 eligible replacement candidates using the same stated
strategy as the displaced record, with a deterministic extra Sobol seed.
"""
from pathlib import Path
import json,shutil,copy,time
import numpy as np
from scipy.stats import qmc
from PIL import Image
import expand_diverse as E
import package_dataset as PK
ROOT=E.ROOT;A=E.A;OUT=E.OUT;ARCH=ROOT/'validation/uniqueness_repair'
def main():
 # Acquire the producer lock and verify that every shard writer has committed.
 import fcntl
 lock=open(ROOT/'logs/build.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert len(list((ROOT/'validation').glob('shard-*.json')))==1000
 status=json.loads((OUT/'build_status.json').read_text()) if (OUT/'build_status.json').exists() else {}
 pid=status.get('packaging_pid')
 if pid:
  import subprocess
  check=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
  if str(ROOT) in check.stdout and 'package_stream.py' in check.stdout:raise RuntimeError('Packaging producer is still present; inspect before repair')
 ARCH.mkdir(parents=True,exist_ok=True)
 if not (ARCH/'manifest_before.json').exists():
  initial=json.loads((E.OLD/'assets/manifest.json').read_text())+[json.loads((A/'designs'/f'{i:04d}.json').read_text()) for i in range(64000,256000)]
  from collections import Counter
  counts=Counter(r['geometry_digest'] for r in initial);duplicates={h:[r for r in initial if r['geometry_digest']==h] for h,n in counts.items() if n>1}
  (ARCH/'manifest_before.json').write_text(json.dumps(initial,separators=(',',':')));(ARCH/'duplicates.json').write_text(json.dumps(duplicates,indent=2))
 E.setup_worker();records=json.loads((ARCH/'manifest_before.json').read_text());groups=json.loads((ARCH/'duplicates.json').read_text());replace=sorted(r['id'] for rows in groups.values() for r in sorted(rows,key=lambda x:x['id'])[1:]);assert replace and min(replace)>=64000
 allhash={r['geometry_digest'] for r in records};history=[];eligible=0;reports={};t=time.time()
 for i in replace:
  old=records[i];theme=old['theme_index'];method=old['selection_strategy'];seq=qmc.Sobol(32,scramble=True,seed=741000+i*13).random_base2(8);pool=[];reject=[]
  for attempt,q in enumerate(seq):
   c,reason=E.candidate(theme,90000000+i*32+attempt,q)
   if c is None:reject.append(dict(attempt=attempt,reason=reason));continue
   eligible+=1
   if c['record']['geometry_digest'] in allhash:reject.append(dict(attempt=attempt,reason='global_duplicate'));continue
   pool.append(c)
   if len(pool)==16:break
  assert len(pool)==16
  if method=='descriptor_bridge':chosen=min(pool,key=lambda c:c['record']['bridge_descriptor_distance']-.35*c['record']['baseline_descriptor_distance'])
  elif method=='descriptor_novelty':chosen=max(pool,key=lambda c:c['record']['baseline_descriptor_distance'])
  else:chosen=pool[0]
  new=chosen['record'];new.update(id=i,selection_strategy=method,uniqueness_repair=dict(prior_geometry_digest=old['geometry_digest'],sampler_seed=741000+i*13,eligible_pool=16));allhash.add(new['geometry_digest'])
  for ext in ['npz','json','png']:
   src=A/'designs'/f'{i:04d}.{ext}';dst=ARCH/'designs'/src.name;dst.parent.mkdir(exist_ok=True)
   assert not dst.exists();shutil.move(src,dst)
  p=new['params'];stem=A/'designs'/f'{i:04d}';np.savez_compressed(stem.with_suffix('.npz'),keep=chosen['mask'],target=[p['Lx'],p['Ly']],orientation=p['orientation']);P,L,ng,keep=E.F.load_design(stem.with_suffix('.npz'));assert E.digest(P,L,keep)==new['geometry_digest'];Image.fromarray(E.thumbnail(P,L,ng,keep)).save(stem.with_suffix('.png'));stem.with_suffix('.json').write_text(json.dumps(new,separators=(',',':')))
  offset=64000+theme*4000 if theme<16 else 128000+(theme-16)*16000;start=(i-offset)//256*256;name=f'descriptors_{theme:02d}_{start:06d}.npz';path=ROOT/'validation'/name;backup=ARCH/'descriptors'/name;backup.parent.mkdir(exist_ok=True)
  if not backup.exists():shutil.copy2(path,backup)
  z=dict(np.load(path));j=int(np.flatnonzero(z['ids']==i)[0]);z['shape'][j]=chosen['shape'];z['geometry'][j]=chosen['geometry'];z['features'][j]=chosen['features'];np.savez_compressed(path,**z)
  records[i]=new;history.append(dict(id=i,prior=old,replacement=new,eligible_candidate_pool=len(pool),screened_rejections=reject));print('Replaced repeated geometry',i,theme,flush=True)
 # Rebuild only the coordinate/image/catalog shards affected by replaced IDs.
 for si in sorted({i//256 for i in replace}):
  for kind,prefix,suffix in [('coordinates','structures','tar'),('images','previews','tar'),('data','designs','parquet')]:
   src=ROOT/'release/dataset'/kind/f'{prefix}-{si:04d}.{suffix}';dst=ARCH/'shards'/kind/src.name;dst.parent.mkdir(exist_ok=True,parents=True);assert not dst.exists();shutil.move(src,dst)
  audit=ROOT/'validation'/f'shard-{si:04d}.json';backup=ARCH/'shards'/audit.name;shutil.move(audit,backup)
  rows=records[si*256:(si+1)*256];samples={rows[0]['id'],min(rows,key=lambda r:r['n_atoms'])['id'],max(rows,key=lambda r:r['n_atoms'])['id']};PK.package_one((si,rows,samples))
 assert len({r['geometry_digest'] for r in records})==256000
 (A/'manifest.json').write_text(json.dumps(records,separators=(',',':')))
 result=dict(replacements=len(replace),ids=replace,preserved_64k_unchanged=True,global_unique_geometries=256000,additional_eligible_candidates=eligible,affected_shards=sorted({i//256 for i in replace}),elapsed_s=time.time()-t,history=history)
 (ARCH/'repair_report.json').write_text(json.dumps(result,indent=2));print('Global uniqueness restored and affected coordinate shards rebuilt',flush=True)
if __name__=='__main__':main()
