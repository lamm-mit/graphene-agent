"""TESTS 3, 9, 10, 11, 12 -- finite-difference forces, neighbour list vs brute force, translation
invariance, total-force balance, periodic-image invariance."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.pytorch.neighborlist import build_dense_neighbor_list, brute_force_pairs, _encode
from atomistics.structures.design_space import pristine, nanomesh_single, precrack


def fd_forces(atoms, calc, h=1e-4, n_atoms=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(atoms), n_atoms, replace=False)
    atoms = atoms.copy(); atoms.calc = calc
    F = atoms.get_forces()
    errs = []
    for i in idx:
        for k in range(3):
            p0 = atoms.positions[i, k]
            atoms.positions[i, k] = p0 + h; Ep = atoms.get_potential_energy()
            atoms.positions[i, k] = p0 - h; Em = atoms.get_potential_energy()
            atoms.positions[i, k] = p0
            fd = -(Ep - Em) / (2 * h)
            errs.append(abs(fd - F[i, k]))
    return float(np.max(errs)), float(np.max(np.abs(F)))


def fd_stress(atoms, calc, h=1e-5):
    atoms = atoms.copy(); atoms.calc = calc
    s = atoms.get_stress()
    vol = atoms.get_volume()
    errs = []
    for comp, (a, b) in enumerate([(0, 0), (1, 1), (0, 1)]):
        def E_at(e):
            at = atoms.copy()
            eps = np.zeros((3, 3)); eps[a, b] += e / 2; eps[b, a] += e / 2
            at.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + eps).T, scale_atoms=True)
            at.calc = calc
            return at.get_potential_energy()
        fd = (E_at(h) - E_at(-h)) / (2 * h) / vol
        ref = s[comp] if comp < 2 else s[5]
        errs.append(abs(fd - ref))
    return float(np.max(errs)), float(np.max(np.abs(s)))


def run():
    out = {}
    calc64 = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0)
    rng = np.random.default_rng(3)
    # --- TEST 3 finite differences (displaced graphene, pore edge, crack tip)
    fd = {}
    for name, atoms in [("displaced_graphene", pristine(Lx=20, Ly=20)), ("nanomesh", nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16)),
                        ("precrack", precrack(Lx=30, Ly=30, crack_len=8))]:
        atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
        e, fmax = fd_forces(atoms, calc64)
        es, smax = fd_stress(atoms, calc64)
        fd[name] = {"max_abs_force_error": e, "max_abs_force": fmax, "max_abs_stress_error": es, "max_abs_stress": smax}
    out["TEST3_finite_difference"] = fd
    # --- TEST 9 neighbour list vs brute force (small periodic sheet with skin)
    eng = TorchRebo2Scr(device="cpu", dtype=torch.float64)
    nl_res = {}
    for name, atoms in [("graphene", pristine(Lx=12, Ly=12)), ("nanomesh", nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16))]:
        atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
        cell = np.asarray(atoms.cell, float)
        i1, j1, S1 = brute_force_pairs(atoms.positions, cell, atoms.pbc, 4.3, max_image=2)
        nl = build_dense_neighbor_list([atoms.positions], cell[None], list(atoms.pbc), cutoff=4.0, skin=0.3, short_cutoff=2.82)
        m = nl.mask.numpy()
        i2 = np.repeat(np.arange(len(atoms))[:, None], nl.M, 1)[m]; j2 = nl.idx.numpy()[m]; S2 = nl.shift.numpy().astype(int)[m]
        k1 = np.sort(_encode(i1, j1, S1, len(atoms))); k2 = np.sort(_encode(i2, j2, S2, len(atoms)))
        nl_res[name] = {"n_pairs_brute": int(len(k1)), "n_pairs_celllist": int(len(k2)), "identical": bool(len(k1) == len(k2) and np.all(k1 == k2)),
                        "max_neighbors": int(nl.max_neighbors), "M": int(nl.M), "Mb": int(nl.Mb), "avg_neighbors": float(nl.average_neighbors())}
    out["TEST9_neighbor_list"] = nl_res
    # --- TEST 10 translation invariance, TEST 11 force balance, TEST 12 periodic image invariance
    inv = {}
    atoms = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    a = atoms.copy(); a.calc = calc64; E0 = a.get_potential_energy(); F0 = a.get_forces()
    shift = np.array([7.3, -11.1, 0.0])
    b = atoms.copy(); b.positions += shift; b.calc = calc64; E1 = b.get_potential_energy(); F1 = b.get_forces()
    inv["TEST10_translation"] = {"dE": float(abs(E1 - E0)), "max_dF": float(np.abs(F1 - F0).max())}
    inv["TEST11_force_balance"] = {"sum_F": float(np.abs(F0.sum(0)).max()), "max_F": float(np.abs(F0).max())}
    c = atoms.copy(); c.wrap(); c.calc = calc64; E2 = c.get_potential_energy(); F2 = c.get_forces()
    d = atoms.repeat((2, 1, 1)); d.calc = calc64; E3 = d.get_potential_energy(); F3 = d.get_forces()
    inv["TEST12_periodic_images"] = {"dE_wrap": float(abs(E2 - E0)), "max_dF_wrap": float(np.abs(F2 - F0).max()),
                                     "dE_per_atom_2x_supercell": float(abs(E3 / len(d) - E0 / len(atoms))),
                                     "max_dF_2x_supercell": float(np.abs(F3[:len(atoms)] - F0).max())}
    out.update(inv)
    return out


if __name__ == "__main__":
    res = run()
    os.makedirs("validation/results", exist_ok=True)
    json.dump(res, open("validation/results/force_tests.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
