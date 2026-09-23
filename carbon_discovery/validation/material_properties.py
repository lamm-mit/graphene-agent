"""TESTS 5-8 -- relaxed lattice constant / bond length, cohesive energy, small-strain elastic
response (2D modulus, Poisson ratio), armchair vs zigzag tensile response (homogeneous, relaxed
internal coordinates), Torch engine (cpu float64 and fast mode) vs Atomistica."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.optimize import minimize_scalar
from ase.optimize import FIRE as ASEFIRE
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator
from atomistics.structures.graphene import graphene_sheet
EV = 16.0217663


def energy_vs_a(calc, a, orientation="zigzag"):
    at = graphene_sheet(2, 2, a=a, orientation=orientation)
    at.calc = calc
    return at.get_potential_energy() / len(at)


def relaxed_lattice_constant(calc):
    res = minimize_scalar(lambda a: energy_vs_a(calc, a), bracket=(2.40, 2.46, 2.52), tol=1e-10)
    a0 = res.x
    return float(a0), float(a0 / np.sqrt(3.0)), float(res.fun)


def elastic_2d(calc, a0, h=1e-3):
    """2D stiffness from stress-strain finite differences with relaxed atoms (uniaxial strain, fixed
    transverse): C11, C12 (N/m); then Y2D = C11 - C12^2/C11, nu = C12/C11."""
    def stress(ex, ey, orientation="zigzag"):
        at = graphene_sheet(3, 3, a=a0, orientation=orientation)
        c = np.asarray(at.cell).copy(); c[0, 0] *= 1 + ex; c[1, 1] *= 1 + ey
        at.set_cell(c, scale_atoms=True)
        at.calc = calc
        opt = ASEFIRE(at, logfile=None); opt.run(fmax=1e-5, steps=2000)
        s = at.get_stress()
        A = c[0, 0] * c[1, 1]
        V = at.get_volume()
        return np.array([s[0], s[1]]) * V / A * EV
    sp = stress(h, 0); sm = stress(-h, 0)
    C11 = (sp[0] - sm[0]) / (2 * h); C21 = (sp[1] - sm[1]) / (2 * h)
    sp = stress(0, h); sm = stress(0, -h)
    C22 = (sp[1] - sm[1]) / (2 * h); C12 = (sp[0] - sm[0]) / (2 * h)
    Y = C11 - C12 * C21 / C22
    nu = C12 / C22
    return dict(C11=float(C11), C22=float(C22), C12=float(C12), C21=float(C21), Y2D=float(Y), nu=float(nu))


def tensile_curve(calc, a0, orientation, strains):
    """Homogeneous uniaxial strain (fixed transverse) with relaxed atoms: sigma_xx, sigma_yy (N/m)."""
    at0 = graphene_sheet(3, 3, a=a0, orientation=orientation)
    sx, sy, E = [], [], []
    at = at0.copy()
    for e in strains:
        c = np.asarray(at0.cell).copy(); c[0, 0] *= 1 + e
        p = at0.positions.copy(); p[:, 0] *= 1 + e
        at = at0.copy(); at.set_cell(c); at.positions = p
        at.calc = calc
        ASEFIRE(at, logfile=None).run(fmax=1e-4, steps=3000)
        s = at.get_stress(); V = at.get_volume(); A = c[0, 0] * c[1, 1]
        sx.append(s[0] * V / A * EV); sy.append(s[1] * V / A * EV); E.append(at.get_potential_energy() / len(at))
    return np.array(sx), np.array(sy), np.array(E)


def run(device_fast="mps"):
    calcs = {"atomistica": make_reference_calculator(),
             "torch_cpu64": TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.3),
             "torch_fast32": TorchRebo2ScrCalculator(device=device_fast, dtype=torch.float32, skin=0.3)}
    out = {}
    for lab, calc in calcs.items():
        a0, bond, e0 = relaxed_lattice_constant(calc)
        out[lab] = {"a0": a0, "bond_length": bond, "cohesive_energy_per_atom": e0}
    a0 = out["atomistica"]["a0"]
    for lab, calc in calcs.items():
        out[lab]["elastic"] = elastic_2d(calc, a0)
    strains = np.linspace(0, 0.30, 31)
    curves = {}
    for lab, calc in calcs.items():
        for orient in ["zigzag", "armchair"]:
            sx, sy, E = tensile_curve(calc, a0, orient, strains)
            curves[f"{lab}_{orient}"] = {"strain": strains.tolist(), "sxx": sx.tolist(), "syy": sy.tolist(), "E": E.tolist(),
                                          "peak_stress": float(sx.max()), "peak_strain": float(strains[int(np.argmax(sx))])}
    out["tensile_curves"] = curves
    return out


if __name__ == "__main__":
    res = run()
    os.makedirs("validation/results", exist_ok=True)
    json.dump(res, open("validation/results/material_properties.json", "w"), indent=2)
    for lab in ["atomistica", "torch_cpu64", "torch_fast32"]:
        r = res[lab]
        print(f"{lab:12s} a0={r['a0']:.6f} A  bond={r['bond_length']:.6f} A  Ecoh={r['cohesive_energy_per_atom']:.6f} eV/atom  "
              f"C11={r['elastic']['C11']:.2f} C12={r['elastic']['C12']:.2f} Y2D={r['elastic']['Y2D']:.2f} N/m nu={r['elastic']['nu']:.4f}")
    for k, v in res["tensile_curves"].items():
        print(f"{k:24s} peak {v['peak_stress']:.2f} N/m at strain {v['peak_strain']:.2f}")
