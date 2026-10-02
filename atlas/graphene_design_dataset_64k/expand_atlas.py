"""Expansion of the preserved 16384-geometry atlas to 64000 designs. No simulations."""
from pathlib import Path
import json,time,copy,shutil
import numpy as np
from PIL import Image
from scipy.stats import qmc
from generate_atlas import spec,NAMES,digest,thumbnail
from source.structures import design_space as D
import fast_geometry as F
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output';OLD=ROOT.parent/'graphene_latent_universe_4k'
LENGTHS={'Lx','Ly','period','pore_d','jitter','slit_len','slit_w','period_x','period_y','spacing','strut_w','ligament_w','pore_d_min','pore_d_max','crack_len','crack_w','hole_d','ring_pore_d','ring_gap','domain','vein_w','bar_w','level3_domain','level3_vein','sub_domain','sub_w','pitch','seam_spacing','vein_offset'}
def scaled(p,scale):
 out={}
 for key,value in p.items():
  if isinstance(value,dict):out[key]=scaled(value,scale)
  elif key in LENGTHS and value is not None:out[key]=float(value)*scale
  elif key=='offset':out[key]=[x*scale for x in value]
  else:out[key]=value
 return out
def main():
 F.install();records=json.loads((OLD/'assets/manifest.json').read_text());hashes={r['geometry_digest'] for r in records};rejected=[];t=time.time()
 for r in records:
  for ext in ['json','npz','png']:
   target=A/'designs'/f'{r["id"]:04d}.{ext}'
   if not target.exists():shutil.copy2(OLD/'assets/designs'/target.name,target)
 for f in range(16):
  qvals=qmc.Sobol(12,scramble=True,seed=2026200+f).random_base2(12)
  for k in range(2976):
   id=16384+f*2976+k;file=A/'designs'/f'{id:04d}.npz';meta=file.with_suffix('.json')
   if meta.exists():
    r=json.loads(meta.read_text());records.append(r);hashes.add(r['geometry_digest']);continue
   for attempt in range(80):
    q=qvals[k] if attempt==0 else np.random.default_rng(id*1000+attempt).random(12)
    family,p=spec(f,k%255,q);p=scaled(p,[.75,.95,1.1,1.3,1.5,1.7][k//496]);p['seed']=40000+id
    if attempt:
     # Small, discrete lattices may exhaust a particular parameter slice.
     # Vary actual cell and motif dimensions together, retaining atom spacing.
     p=scaled(p,1+.075*(attempt%13)+.04*q[9])
    # Registry variations are genuine additional geometries, not repeated images.
    if f in [0,1,2,3,10,11,12,13,14,15]:p['offset']=[2.46*q[10],2.46*q[11]]
    at=D.generate(family,**copy.deepcopy(p));P,L,ng=F.lattice(*at.info['_lattice_key']);keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True;h=digest(P,L,keep)
    if h not in hashes and len(at)>=100:break
    rejected.append(dict(id=id,attempt=attempt,family=family,params=p,reason='duplicate' if h in hashes else 'fewer_than_100_atoms'))
   else:
    (OUT/'rejected_generation_attempts_incomplete.json').write_text(json.dumps(rejected,indent=2))
    raise RuntimeError(('Unique geometry unavailable',id))
   hashes.add(h);r=dict(id=id,theme=NAMES[f],theme_index=f,family=family,params=p,n_atoms=len(at),n_pristine=len(P),cell_A=L.tolist(),porosity=1-len(at)/len(P),n_pruned=at.info.get('n_pruned',0),geometry_digest=h,source='new procedural geometry',mechanical_evaluation=False)
   np.savez_compressed(file,keep=np.packbits(keep),target=[p['Lx'],p['Ly']],orientation=p['orientation']);meta.write_text(json.dumps(r,indent=2));Image.fromarray(thumbnail(P,L,ng,keep)).save(file.with_suffix('.png'));records.append(r)
   if k%128==127:print(f'{len(records):5d}/64000  {NAMES[f]:20s}  {time.time()-t:.1f}s',flush=True)
  (OUT/'generation_progress.json').write_text(json.dumps(dict(completed=len(records),elapsed_s=time.time()-t)))
 assert len(records)==len(hashes)==64000
 (A/'manifest.json').write_text(json.dumps(records,indent=2));(OUT/'rejected_generation_attempts.json').write_text(json.dumps(rejected,indent=2))
 summary=dict(designs=len(records),unique_geometries=len(hashes),retained_original_atlas=16384,new_geometries=47616,theme_count=16,generator_families=sorted(set(r['family'] for r in records)),total_atoms=sum(r['n_atoms'] for r in records),width_range_nm=[min(r['cell_A'][0]/10 for r in records),max(r['cell_A'][0]/10 for r in records)],max_atoms=max(r['n_atoms'] for r in records),simulations_run=0)
 (OUT/'generation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
