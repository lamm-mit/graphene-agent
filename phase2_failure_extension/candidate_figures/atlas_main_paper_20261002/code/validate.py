"""Independent checks of the bundled map, exact neighbor ranks and geometries."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
cfg=json.loads((DATA/'manifest.json').read_text())
z=np.load(DATA/'embedding.npz')
assert len(z['design_id'])==256000
assert np.array_equal(z['design_id'],np.arange(256000))
assert np.unique(z['theme_index'],return_counts=True)[1].tolist()==[8000]*16+[16000]*8
assert sorted(r['theme_index'] for r in cfg['examples'])==list(range(24))
assert hashlib.sha256((DATA/'embedding.npz').read_bytes()).hexdigest()==cfg['source_embedding_file_hash']
xyz=z['display_xyz'].astype(np.float64)
ids=np.lexsort((z['design_id'],np.sum((xyz-xyz[254261])**2,axis=1)))[:4]
assert ids.tolist()==[r['id'] for r in cfg['neighborhood']]
for r in cfg['examples']+cfg['neighborhood']:
    g=np.load(DATA/f"GDU-{r['id']:06d}.npz")
    assert len(g['positions'])==r['n_atoms']
    assert np.allclose(np.diag(g['cell']),r['cell_A'])
    assert g['pbc'].tolist()==[True,True,False]
    assert np.min(g['edges'])>=0 and np.max(g['edges'])<len(g['positions'])
    assert np.allclose(np.linalg.norm(g['positions'][g['edges'][:,0]]-g['positions'][g['edges'][:,1]],axis=1),2.460177/np.sqrt(3),atol=1e-6)
print('PASS: population, 24 group counts, unchanged embedding, actual atomic coordinates/cells/bonds and exact 3D neighbor ranks.')
