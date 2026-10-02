"""Regenerate the paper's 24 atlas examples and neighbors from released recipes.

Checks exact geometry digests, cell/PBC and atom coordinates against stored data,
and standard extended-XYZ round trips. Does not simulate mechanical properties.
"""
from pathlib import Path
import copy, json, sys, tempfile
import numpy as np
from ase import Atoms
from ase.io import read, write
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'atlas/graphene_design_dataset_256k'))
import fast_geometry as F
from generate_atlas import digest
from source.structures import design_space as D
from hybrid_geometry import generate

def main():
    F.install()
    data=ROOT/'phase2_failure_extension/candidate_figures/atlas_main_paper_20261002/data'
    cfg=json.loads((data/'manifest.json').read_text())
    examples={r['id']:r for r in cfg['examples']+cfg['neighborhood']}
    assert {r['theme_index'] for r in cfg['examples']}==set(range(24))
    checked=[]
    with tempfile.TemporaryDirectory() as tmp:
        for idx,r in examples.items():
            if r['family']=='periodic_hybrid':
                P,L,_,keep,_=generate(copy.deepcopy(r['params']))
                at=Atoms(numbers=np.full(keep.sum(),6),positions=P[keep],cell=np.diag(L),pbc=[True,True,False])
            else:
                at=D.generate(r['family'],**copy.deepcopy(r['params']))
                P,L,_=F.lattice(*at.info['_lattice_key'])
                keep=np.zeros(len(P),bool);keep[at.arrays['site_id']]=True
            assert digest(P,L,keep)==r['geometry_digest'],idx
            saved=np.load(data/f'GDU-{idx:06d}.npz')
            assert np.array_equal(at.positions,saved['positions']),idx
            assert np.array_equal(at.cell.array,saved['cell']),idx
            assert np.array_equal(at.pbc,saved['pbc']),idx
            at.info=dict(design_id=f'GDU-{idx:06d}',geometry_status='unrelaxed',property_status='not_evaluated',length_unit='angstrom')
            path=Path(tmp)/f'{idx}.extxyz';write(path,at,format='extxyz');restored=read(path)
            assert np.allclose(at.positions,restored.positions,atol=5e-9,rtol=0),idx
            assert np.allclose(at.cell,restored.cell,atol=1e-12,rtol=0),idx
            assert np.array_equal(at.pbc,restored.pbc) and np.all(restored.numbers==6),idx
            assert restored.info['geometry_status']=='unrelaxed',idx
            checked.append(dict(id=idx,atoms=len(at),digest=r['geometry_digest']))
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    (out/'atlas_regeneration.json').write_text(json.dumps(dict(designs=checked,groups=24,all_passed=True),indent=2)+'\n')
    print(f'PASS: {len(checked)} exact recipe regenerations across all 24 groups; cell/PBC and extxyz round trips.')
if __name__=='__main__':main()
