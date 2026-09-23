"""TEST 18 -- complete stress-strain comparison of the Torch-driven AQS with an AQS driven by the
authoritative Atomistica backend (ASE FIRE + Rebo2Scr), same protocol (uniaxial strain steps of 1%,
fixed transverse strain, FIRE fmax = 0.02 eV/A), for a precracked sheet and a nanomesh."""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from ase.optimize import FIRE as ASEFIRE
from ase.neighborlist import primitive_neighbor_list
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner, AQSConfig
from atomistics.structures.design_space import precrack, nanomesh_single
EV = 16.0217663


def n_bonds(atoms, r=2.0):
    i, j = primitive_neighbor_list("ij", pbc=atoms.pbc, cell=np.asarray(atoms.cell, float), positions=atoms.positions, cutoff=r)
    return int(len(i) // 2)


def ase_aqs(atoms0, calc, strains, fmax=0.02, steps=5000):
    """ASE-driven AQS: affine strain step, FIRE relaxation (ASE implementation), stress from calculator."""
    at = atoms0.copy(); at.calc = calc
    ASEFIRE(at, logfile=None, dt=0.1, maxstep=0.2).run(fmax=fmax, steps=steps)
    L0 = np.array(at.cell)[[0, 1], [0, 1]]
    ref = at.copy()
    sxx, nb, E, conv = [], [], [], []
    e_prev = 0.0
    for e in strains:
        c = np.array(at.cell); c[0, 0] = L0[0] * (1 + e)
        p = at.positions.copy(); p[:, 0] *= (1 + e) / (1 + e_prev)
        at.set_cell(c); at.positions = p; e_prev = e
        opt = ASEFIRE(at, logfile=None, dt=0.1, maxstep=0.2); opt.run(fmax=fmax, steps=steps)
        s = at.get_stress(); A = c[0, 0] * c[1, 1]; V = at.get_volume()
        sxx.append(s[0] * V / A * EV); nb.append(n_bonds(at)); E.append(at.get_potential_energy()); conv.append(bool(np.sqrt((at.get_forces() ** 2).sum(1)).max() < fmax))
    return dict(strain=list(map(float, strains)), sxx=sxx, n_bonds=nb, energy=E, converged=conv)


def torch_aqs(atoms0, device, dtype, strains):
    eng = TorchRebo2Scr(device=device, dtype=dtype)
    sysb = BatchedSystem([atoms0], eng)
    cfg = AQSConfig(d_strain=float(strains[1] - strains[0]), refine_first_damage=False, max_strain=float(strains[-1]) + 1e-9,
                    fmax=0.02, transverse_relax=False, zero_stress_relax=False, stop_when_not_spanning=False,
                    stop_stress_fraction=-1.0, max_steps=len(strains) + 2)
    r = AQSRunner(eng, sysb, cfg, log=lambda *a: None)
    res = r.run()[0]
    return dict(strain=res["eps_x"].tolist(), sxx=(res["sigma_xx"] * EV).tolist(), n_bonds=res["n_bonds"].tolist(),
                energy=res["energy"].tolist(), converged=res["fire_converged"].tolist())


if __name__ == "__main__":
    strains = np.round(np.arange(0.01, 0.161, 0.01), 4)
    out = {}
    for name, atoms in [("precrack", precrack(Lx=40, Ly=40, crack_len=10)), ("nanomesh", nanomesh_single(Lx=48, Ly=48, pore_d=8, period=16))]:
        t0 = time.time()
        ref = ase_aqs(atoms, make_reference_calculator(), strains)
        t1 = time.time()
        tor_ase = ase_aqs(atoms, TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.3), strains)
        t2 = time.time()
        tor64 = torch_aqs(atoms, "cpu", torch.float64, strains)
        t3 = time.time()
        tor32 = torch_aqs(atoms, "mps", torch.float32, strains)
        t4 = time.time()
        out[name] = {"atomistica_ase_fire": ref, "torch_cpu64_ase_fire": tor_ase, "torch_cpu64_batched_fire": tor64,
                     "torch_mps32_batched_fire": tor32,
                     "timing_s": {"atomistica": t1 - t0, "torch_ase": t2 - t1, "torch64": t3 - t2, "torch32": t4 - t3}, "n_atoms": len(atoms)}
        for k in ["torch_cpu64_ase_fire", "torch_cpu64_batched_fire", "torch_mps32_batched_fire"]:
            a = np.array(ref["sxx"]); b = np.array(out[name][k]["sxx"][1:1 + len(a)]) if k != "torch_cpu64_ase_fire" else np.array(out[name][k]["sxx"])
            m = min(len(a), len(b))
            out[name][k + "_max_abs_dsigma_Nm"] = float(np.abs(a[:m] - b[:m]).max())
            print(f"{name} {k}: max|dsigma| = {out[name][k + '_max_abs_dsigma_Nm']:.3e} N/m  (peak ref {a.max():.2f})")
        print(name, "timings", out[name]["timing_s"])
    json.dump(out, open("validation/results/reference_aqs.json", "w"), indent=1)
    print("saved")
