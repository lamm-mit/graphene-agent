import sys, time, numpy as np, torch
sys.path.insert(0, ".")
from ase.build import graphene
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator

def compare(atoms, label, device="cpu", dtype=torch.float64):
    a = atoms.copy(); a.calc = make_reference_calculator()
    E0 = a.get_potential_energy(); F0 = a.get_forces(); s0 = a.get_stress()
    b = atoms.copy(); b.calc = TorchRebo2ScrCalculator(device=device, dtype=dtype, skin=0.0)
    E1 = b.get_potential_energy(); F1 = b.get_forces(); s1 = b.get_stress()
    print(f"{label:40s} N={len(atoms):5d} dE={E1-E0:+.3e} eV  dE/atom={(E1-E0)/len(atoms):+.2e}  max|dF|={np.abs(F1-F0).max():.2e}  max|F|={np.abs(F0).max():.3f}  max|dS|={np.abs(s1-s0).max():.2e} (|S|max {np.abs(s0).max():.3e})")
    return E0, E1, F0, F1, s0, s1

g = graphene(a=2.46, size=(4,4,1), vacuum=10.0); g.pbc=[True,True,False]
compare(g, "pristine graphene 4x4 (cpu f64)")
rng = np.random.default_rng(1)
h = g.copy(); h.positions += rng.normal(0, 0.1, h.positions.shape)
compare(h, "random displaced 0.1A (cpu f64)")
h2 = g.copy(); h2.positions += rng.normal(0, 0.2, h2.positions.shape)
compare(h2, "random displaced 0.2A (cpu f64)")
# strained
s = g.copy(); c = s.cell.copy(); c[0] *= 1.15; s.set_cell(c, scale_atoms=True)
compare(s, "15% strained (cpu f64)")
# vacancy
v = g.copy(); del v[5]
compare(v, "single vacancy (cpu f64)")
v2 = h.copy(); del v2[[3,4,9]]
compare(v2, "3 vacancies + displaced (cpu f64)")
# stretched bond near rupture: pull one atom
w = g.copy(); w.positions[0,0] += 0.9
compare(w, "atom displaced 0.9A (cpu f64)")
# mps float32
compare(h, "random displaced 0.1A (mps f32)", device="mps", dtype=torch.float32)
