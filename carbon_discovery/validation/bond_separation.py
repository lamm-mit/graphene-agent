"""TEST 19 -- bond-separation validation: E(r) and F(r) through complete rupture for representative
carbon bonding environments, Torch engine (cpu float64, and mps float32) vs Atomistica Rebo2Scr.

Environments:
  A. C2 dimer (isolated bond)
  B. one bond inside a periodic graphene sheet: atom j displaced along the bond direction, all
     other atoms fixed (bond in a sp2 environment, including the screening of the stretched bond)
  C. homogeneous affine stretch of zigzag- and armchair-oriented periodic sheets (all bonds, up to 80%)
  D. crack-tip bond: bond at the tip of a slit crack, tip atom pulled perpendicular to the crack
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from ase import Atoms
from ase.build import graphene
from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator
from atomistics.structures.design_space import pristine, precrack


def scan(make_atoms, rs, calcs, atom_index, direction):
    """Return dict of arrays: r, E[label], Fdir[label] (force on the displaced atom projected on direction)."""
    out = {"r": np.array(rs)}
    for label, calc in calcs.items():
        E, F = [], []
        for r in rs:
            a = make_atoms(r)
            a.calc = calc
            E.append(a.get_potential_energy())
            F.append(float(np.dot(a.get_forces()[atom_index], direction)))
        out[f"E_{label}"] = np.array(E)
        out[f"F_{label}"] = np.array(F)
    return out


def run(device_fast="mps"):
    calcs = {"atomistica": make_reference_calculator(),
             "torch_cpu64": TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0),
             "torch_fast32": TorchRebo2ScrCalculator(device=device_fast, dtype=torch.float32, skin=0.0)}
    results = {}
    # A. dimer
    rs = np.linspace(0.9, 4.2, 331)
    def dimer(r):
        return Atoms("C2", positions=[[0, 0, 0], [r, 0, 0]], cell=[25, 25, 25], pbc=False)
    results["A_dimer"] = scan(dimer, rs, calcs, 1, np.array([1.0, 0, 0]))
    # B. bond in graphene: sheet 6x6 rectangular zigzag, pull atom j along bond direction
    base = pristine(Lx=20, Ly=20, orientation="zigzag")
    from ase.neighborlist import neighbor_list
    i, j, D = neighbor_list("ijD", base, 1.6)
    k = 0
    ai, aj, dvec = int(i[k]), int(j[k]), D[k] / np.linalg.norm(D[k])
    r0 = np.linalg.norm(D[k])
    rs = np.linspace(r0, 3.4, 200)   # beyond ~3.5 A the pulled atom collides with the next lattice atom
    def bond_in_sheet(r):
        a = base.copy()
        a.positions[aj] = a.positions[ai] + dvec * r
        return a
    res = scan(bond_in_sheet, rs, calcs, aj, dvec)
    results["B_bond_in_graphene"] = res
    # C. homogeneous stretch
    for orient in ["zigzag", "armchair"]:
        b0 = pristine(Lx=15, Ly=15, orientation=orient)
        strains = np.linspace(0.0, 0.8, 161)
        out = {"strain": strains}
        for label, calc in calcs.items():
            E, S = [], []
            for e in strains:
                a = b0.copy()
                c = np.array(a.cell); c[0, 0] *= 1 + e
                p = a.positions.copy(); p[:, 0] *= 1 + e
                a.set_cell(c); a.positions = p
                a.calc = calc
                E.append(a.get_potential_energy() / len(a))
                s = a.get_stress()
                A = c[0, 0] * c[1, 1]
                S.append(s[0] * abs(np.linalg.det(c)) / A * 16.0217663)   # 2D stress N/m
            out[f"E_{label}"] = np.array(E)
            out[f"S_{label}"] = np.array(S)
        results[f"C_stretch_{orient}"] = out
    # D. crack-tip bond
    ck = precrack(Lx=40, Ly=40, crack_len=12, crack_w=3.0, angle_deg=90)
    # find crack tip: atom nearest to the crack tip position (centre + half length along y)
    L = np.array([ck.cell[0, 0], ck.cell[1, 1]])
    tip = np.array([L[0] / 2, L[1] / 2 + 6.0])
    d2 = ((ck.positions[:, :2] - tip) ** 2).sum(1)
    t = int(np.argmin(d2))
    i, j, D = neighbor_list("ijD", ck, 1.6)
    nb = j[i == t]
    # pull tip atom in +x (perpendicular to crack) -- stretching the bond(s) bridging the tip
    rs = np.linspace(0.0, 1.8, 181)
    def cracktip(dx):
        a = ck.copy()
        a.positions[t, 0] += dx
        return a
    res = scan(cracktip, rs, calcs, t, np.array([1.0, 0, 0]))
    res["r"] = rs
    results["D_cracktip_pull"] = res
    return results


def analyse(results):
    """Compute agreement metrics and smoothness diagnostics."""
    summary = {}
    for key, res in results.items():
        x = res.get("r", res.get("strain"))
        s = {}
        if "E_atomistica" in res:
            for lab in ["torch_cpu64", "torch_fast32"]:
                s[f"maxabs_dE_{lab}"] = float(np.max(np.abs(res[f"E_{lab}"] - res["E_atomistica"])))
                if f"F_{lab}" in res:
                    s[f"maxabs_dF_{lab}"] = float(np.max(np.abs(res[f"F_{lab}"] - res["F_atomistica"])))
                if f"S_{lab}" in res:
                    s[f"maxabs_dS_{lab}"] = float(np.max(np.abs(res[f"S_{lab}"] - res["S_atomistica"])))
            # force smoothness: largest jump between consecutive samples relative to the sampling step
            Fa = res.get("F_atomistica", res.get("S_atomistica"))
            dF = np.diff(Fa)
            s["max_force_jump_between_samples"] = float(np.max(np.abs(dF)))
            s["sample_step"] = float(x[1] - x[0])
            s["max_force"] = float(np.max(np.abs(Fa)))
            # cutoff re-stiffening: any increase of |F| beyond the main maximum (r > r_peak) in the decaying tail
            ip = int(np.argmax(np.abs(Fa)))
            tail = np.abs(Fa[ip:])
            s["tail_max_restiffening"] = float(np.max(np.diff(tail))) if len(tail) > 2 else 0.0
        summary[key] = s
    return summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = run()
    summ = analyse(res)
    os.makedirs("validation/results", exist_ok=True)
    np.savez_compressed("validation/results/bond_separation.npz", **{f"{k}__{kk}": v for k, d in res.items() for kk, v in d.items()})
    json.dump(summ, open("validation/results/bond_separation_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panels = [("A_dimer", "r", "C2 dimer"), ("B_bond_in_graphene", "r", "bond in graphene (atom pulled)"),
              ("D_cracktip_pull", "r", "crack-tip atom pulled"), ("C_stretch_zigzag", "strain", "homogeneous stretch (zigzag)")]
    for col, (key, xk, title) in enumerate(panels):
        d = res[key]; x = d[xk]
        axE, axF = axes[0, col], axes[1, col]
        axE.plot(x, d["E_atomistica"], "k-", lw=2, label="Atomistica Rebo2Scr")
        axE.plot(x, d["E_torch_cpu64"], "r--", lw=1, label="Torch cpu f64")
        axE.plot(x, d["E_torch_fast32"], "b:", lw=1, label="Torch mps f32")
        axE.set_title(title); axE.set_ylabel("E (eV)" if xk == "r" else "E/atom (eV)"); axE.legend(fontsize=7)
        Fk = "F" if "F_atomistica" in d else "S"
        axF.plot(x, d[f"{Fk}_atomistica"], "k-", lw=2); axF.plot(x, d[f"{Fk}_torch_cpu64"], "r--", lw=1); axF.plot(x, d[f"{Fk}_torch_fast32"], "b:", lw=1)
        axF.set_xlabel("r (A)" if xk == "r" else "strain"); axF.set_ylabel("F along pull (eV/A)" if Fk == "F" else "2D stress (N/m)")
        ax2 = axF.twinx(); ax2.plot(x, np.abs(d[f"{Fk}_torch_cpu64"] - d[f"{Fk}_atomistica"]), "r-", lw=0.5, alpha=0.6); ax2.plot(x, np.abs(d[f"{Fk}_torch_fast32"] - d[f"{Fk}_atomistica"]), "b-", lw=0.5, alpha=0.6); ax2.set_yscale("log"); ax2.set_ylabel("|difference|", fontsize=8)
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"figures/validation/bond_separation.{ext}", dpi=200 if ext == "png" else None)
    print("figure saved")
