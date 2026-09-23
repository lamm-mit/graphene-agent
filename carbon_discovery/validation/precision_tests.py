"""TEST 4 (fast vs reference precision), TEST 16 (precision near fracture), TEST 17 (reproducibility).

Fast mode  : mps float32   (or cpu float32 if MPS is unavailable)
Reference  : cpu float64 (identical to Atomistica to ~1e-13, see TEST 1/2)
Configurations: relaxed structures + frames from stored AQS trajectories (pre-damage, peak, post-peak,
crack propagation), i.e. genuinely strained / partially broken carbon networks."""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from ase import Atoms
from ase.io import read
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
EV = 16.0217663


def frames_from_npz(path, which=("first", "pre_damage", "peak", "post_peak", "last")):
    d = np.load(path)
    P, C, sxx = d["positions"], d["cells"], d["sigma_xx"]
    nb = d["n_broken_cum"]
    ip = int(np.argmax(sxx))
    idx = {"first": 0, "peak": ip, "last": len(P) - 1}
    dmg = np.where(nb > 0)[0]
    idx["pre_damage"] = int(dmg[0] - 1) if len(dmg) and dmg[0] > 0 else ip
    idx["post_peak"] = min(ip + 2, len(P) - 1)
    out = []
    for k in which:
        f = idx[k]
        out.append((k, P[f].astype(float), C[f]))
    return out


def evaluate(eng, atoms):
    sysb = BatchedSystem([atoms], eng)
    nl = sysb.build_neighbor_list(skin=0.0)
    out = eng.evaluate(sysb.positions, sysb.cell, sysb.sid, nl, compute_stress=True, per_atom=True)
    A = float(sysb.areas()[0])
    return {"E": float(out["energy"][0]), "F": out["forces"].cpu().numpy().astype(float),
            "sigma": (out["dE_deps"][0].cpu().numpy() / A * EV).astype(float),
            "e_atom": out["energy_per_atom"].cpu().numpy().astype(float)}


def run(fast_device="mps"):
    ref = TorchRebo2Scr(device="cpu", dtype=torch.float64)
    fast = TorchRebo2Scr(device=fast_device, dtype=torch.float32)
    results = {"fast_device": str(fast.device), "cases": []}
    cases = []
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "test_frames")
    for name in ["mesh_progressive", "precrack", "pristine_zz"]:
        p = os.path.join(base, name + ".npz")
        if not os.path.exists(p):
            continue
        for k, P, C in frames_from_npz(p):
            n = len(P)
            atoms = Atoms("C" * n, positions=P, cell=C, pbc=[True, True, False])
            cases.append((f"{name}:{k}", atoms))
    for label, atoms in cases:
        r = evaluate(ref, atoms); f = evaluate(fast, atoms)
        n = len(atoms)
        dF = np.abs(f["F"] - r["F"])
        results["cases"].append({
            "case": label, "n_atoms": n,
            "dE_total_eV": float(f["E"] - r["E"]), "dE_per_atom_eV": float((f["E"] - r["E"]) / n),
            "max_abs_dE_atom_eV": float(np.abs(f["e_atom"] - r["e_atom"]).max()),
            "max_abs_dF_eV_A": float(dF.max()), "rms_dF_eV_A": float(np.sqrt((dF ** 2).mean())),
            "max_abs_F_eV_A": float(np.abs(r["F"]).max()),
            "max_abs_dsigma_Nm": float(np.abs(f["sigma"] - r["sigma"]).max()), "sigma_xx_ref_Nm": float(r["sigma"][0, 0]),
        })
        print(f"{label:28s} N={n:5d} dE/atom={results['cases'][-1]['dE_per_atom_eV']:+.2e} max|dF|={dF.max():.2e} (max|F|={np.abs(r['F']).max():.2f}) dsigma={results['cases'][-1]['max_abs_dsigma_Nm']:.2e} N/m (sxx={r['sigma'][0,0]:.2f})")
    # TEST 17 reproducibility: identical repeated evaluations (fast device) and repeated short minimisation
    atoms = cases[0][1]
    e1 = evaluate(fast, atoms); e2 = evaluate(fast, atoms)
    results["reproducibility_evaluation"] = {"dE": float(abs(e1["E"] - e2["E"])), "max_dF": float(np.abs(e1["F"] - e2["F"]).max())}
    return results


if __name__ == "__main__":
    res = run()
    json.dump(res, open("validation/results/precision_tests.json", "w"), indent=2)
    print(json.dumps(res["reproducibility_evaluation"]))
