"""Automated tests (pytest -q tests/).  Fast subset of the validation suite plus smoke tests of the pipeline."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, pytest
from atomistics.structures.design_space import pristine, nanomesh_single, precrack, generate, FAMILIES
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.pytorch.neighborlist import build_dense_neighbor_list, brute_force_pairs, _encode
from potentials.rebo2scr.reference import atomistica_reference as aref
from simulation.system import BatchedSystem
from simulation.minimization.fire import fire_minimize
from simulation.aqs.aqs import AQSRunner, AQSConfig
from analysis.metrics import compute_metrics
from atomistics.descriptors.descriptors import compute_descriptors
from atomistics.io import roundtrip_check

rng = np.random.default_rng(0)


@pytest.mark.skipif(not aref.available(), reason="Atomistica not installed")
@pytest.mark.parametrize("make", [lambda: pristine(Lx=20, Ly=20), lambda: nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16), lambda: precrack(Lx=30, Ly=30, crack_len=8)])
def test_energy_forces_match_atomistica(make):
    atoms = make(); atoms.positions += rng.normal(0, 0.08, atoms.positions.shape)
    a = atoms.copy(); a.calc = aref.make_reference_calculator(); E0, F0, S0 = a.get_potential_energy(), a.get_forces(), a.get_stress()
    b = atoms.copy(); b.calc = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0); E1, F1, S1 = b.get_potential_energy(), b.get_forces(), b.get_stress()
    assert abs(E1 - E0) / len(atoms) < 1e-9
    assert np.abs(F1 - F0).max() < 1e-8
    assert np.abs(S1 - S0).max() < 1e-10


def test_finite_difference_forces():
    atoms = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    atoms.calc = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0)
    F = atoms.get_forces(); h = 1e-4
    for i in rng.choice(len(atoms), 3, replace=False):
        for k in range(3):
            p0 = atoms.positions[i, k]
            atoms.positions[i, k] = p0 + h; Ep = atoms.get_potential_energy()
            atoms.positions[i, k] = p0 - h; Em = atoms.get_potential_energy(); atoms.positions[i, k] = p0
            assert abs(-(Ep - Em) / (2 * h) - F[i, k]) < 1e-4


def test_neighbor_list_vs_brute_force():
    atoms = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); cell = np.asarray(atoms.cell, float)
    i1, j1, S1 = brute_force_pairs(atoms.positions, cell, atoms.pbc, 4.3, max_image=2)
    nl = build_dense_neighbor_list([atoms.positions], cell[None], list(atoms.pbc), cutoff=4.0, skin=0.3, short_cutoff=2.82)
    m = nl.mask.numpy(); i2 = np.repeat(np.arange(len(atoms))[:, None], nl.M, 1)[m]; j2 = nl.idx.numpy()[m]; S2 = nl.shift.numpy().astype(int)[m]
    assert np.array_equal(np.sort(_encode(i1, j1, S1, len(atoms))), np.sort(_encode(i2, j2, S2, len(atoms))))


def test_invariances():
    atoms = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    calc = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0)
    a = atoms.copy(); a.calc = calc; E0, F0 = a.get_potential_energy(), a.get_forces()
    assert np.abs(F0.sum(0)).max() < 1e-10
    b = atoms.copy(); b.positions += [5.3, -2.1, 0]; b.calc = calc
    assert abs(b.get_potential_energy() - E0) < 1e-9
    c = atoms.repeat((2, 1, 1)); c.calc = calc
    assert abs(c.get_potential_energy() / len(c) - E0 / len(atoms)) < 1e-9


def test_fast_mode_precision():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    atoms = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    a = atoms.copy(); a.calc = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0); F0 = a.get_forces(); E0 = a.get_potential_energy()
    b = atoms.copy(); b.calc = TorchRebo2ScrCalculator(device=dev, dtype=torch.float32, skin=0.0); F1 = b.get_forces(); E1 = b.get_potential_energy()
    assert abs(E1 - E0) / len(atoms) < 1e-4 and np.abs(F1 - F0).max() < 5e-3


def test_generators_produce_spanning_structures():
    from simulation.topology import spanning_x
    from atomistics.descriptors.descriptors import bond_graph
    for fam in FAMILIES:
        if fam == "nanomesh_hier":
            kw = {"Lx": 80, "Ly": 80, "domain": 40, "vein_w": 8}
        elif fam == "repeat":  # exact periodic repetition of a base pattern (size-convergence utility)
            kw = {"base_family": "nanomesh_single", "base_params": {"Lx": 32, "Ly": 32}, "reps": (2, 2)}
        else:
            kw = {"Lx": 60, "Ly": 60}
        a = generate(fam, **kw)
        i, j, S, d = bond_graph(a)
        assert len(a) > 100 and spanning_x(i, j, S, len(a)), fam
        assert 0 <= a.info["porosity"] < 0.7


def test_descriptors_and_io():
    a = nanomesh_single(Lx=48, Ly=48, pore_d=8, period=16)
    d = compute_descriptors(a, a.info["design"])
    assert 0.1 < d["porosity"] < 0.35 and d["pore_size_mean_A"] > 3 and d["ligament_width_mean_A"] > 3
    rt = roundtrip_check(a, os.path.join(ROOT, "validation", "results", "roundtrip_test"))
    assert all(v["ok"] for v in rt.values())


def test_relaxation_and_aqs_smoke():
    eng = TorchRebo2Scr(device="cpu", dtype=torch.float64)
    atoms = pristine(Lx=15, Ly=15)
    sysb = BatchedSystem([atoms], eng); nl = sysb.build_neighbor_list(skin=0.3)
    nl, res = fire_minimize(eng, sysb, nl, fmax=1e-3, max_steps=500)
    assert res.converged.all()
    cfg = AQSConfig(d_strain=0.01, d_strain_min=0.005, max_strain=0.04, fmax=0.02, max_steps=8, transverse_relax=True)
    r = AQSRunner(eng, sysb, cfg, log=lambda *a: None).run()[0]
    assert len(r["eps_x"]) >= 3 and r["sigma_xx"][-1] > r["sigma_xx"][0]
    m = compute_metrics(r)
    assert 150 < m["modulus_2d_Nm"] < 350
