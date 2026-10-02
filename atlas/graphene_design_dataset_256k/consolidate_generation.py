"""Consolidate completed, checked generation chunks after uniqueness repair."""
from pathlib import Path
import json
import numpy as np
import expand_diverse as E
ROOT=E.ROOT;A=E.A;OUT=E.OUT;OLD=E.OLD
def main():
 records=json.loads((A/'manifest.json').read_text());assert len(records)==256000 and len({r['geometry_digest'] for r in records})==256000
 old=json.loads((OLD/'assets/manifest.json').read_text());assert records[:64000]==old
 z=np.load(OLD/'assets/geometric_descriptors.npz');shape=np.empty((256000,z['shape'].shape[1]),np.float32);geom=np.empty((256000,10),np.float32);features=np.empty((256000,64),np.float32);seen=np.zeros(256000,bool)
 shape[:64000]=z['shape'];geom[:64000]=z['geometry'];features[:64000]=np.load(A/'selection_model.npz')['baseline'];seen[:64000]=True
 for p in sorted((ROOT/'validation').glob('descriptors_*.npz')):
  x=np.load(p);ids=x['ids'];assert not seen[ids].any();seen[ids]=True;shape[ids]=x['shape'];geom[ids]=x['geometry'];features[ids]=x['features']
 assert seen.all() and np.isfinite(shape).all() and np.isfinite(geom).all() and np.isfinite(features).all()
 np.savez_compressed(A/'geometric_descriptors.npz',shape=shape,geometry=geom);np.save(A/'selection_features.npy',features)
 reports=[json.loads(p.read_text()) for p in sorted((ROOT/'validation').glob('generation_*.json'))];repair=json.loads((ROOT/'validation/uniqueness_repair/repair_report.json').read_text())
 summary=dict(designs=256000,unique_geometries=256000,preserved_designs=64000,new_designs=192000,candidate_pool=sum(r['candidates'] for r in reports)+sum(h['eligible_candidate_pool'] for h in repair['history']),global_uniqueness_replacements=repair['replacements'],theme_count=24,generator_families=sorted(set(r['family'] for r in records)),total_atoms=sum(r['n_atoms'] for r in records),min_atoms=min(r['n_atoms'] for r in records),max_atoms=max(r['n_atoms'] for r in records),width_range_nm=[min(r['cell_A'][0]/10 for r in records),max(r['cell_A'][0]/10 for r in records)],simulations_run=0)
 (OUT/'generation_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'selection_chunks.json').write_text(json.dumps(dict(initial_selection_pools=reports,post_selection_uniqueness_repair='provenance/uniqueness_repair.json')));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
