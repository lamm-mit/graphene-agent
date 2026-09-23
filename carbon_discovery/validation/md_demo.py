"""Optional finite-temperature MD demonstration (mechanism / temperature check): constant-strain-rate
tension of the reference nanomesh at 300 K (Langevin), compared with the AQS curve of the same structure."""
from __future__ import annotations
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from atomistics.structures.design_space import nanomesh_single
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.minimization.fire import fire_minimize
from simulation.md.md import run_md
from simulation.aqs.aqs import AQSRunner, AQSConfig

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-3      # strain per ps
    eng = TorchRebo2Scr(device=device, dtype=torch.float32)
    atoms = nanomesh_single(Lx=48, Ly=48, pore_d=8, period=16)
    out = {"T": T, "strain_rate_per_ps": rate, "n_atoms": len(atoms), "device": device}
    # AQS reference for the same structure
    sysa = BatchedSystem([atoms], eng); cfg = AQSConfig(d_strain=0.005, d_strain_elastic=0.01, max_strain=0.30, fmax=0.02, post_peak_strain_limit=0.12, stop_stress_fraction=0.1)
    ra = AQSRunner(eng, sysa, cfg, log=lambda *a: None).run()[0]
    out["aqs"] = {"eps": ra["eps_x"].tolist(), "sxx": (ra["sigma_xx"] * 16.0217663).tolist()}
    # MD: equilibrate 5 ps at T, then strain at constant rate (fixed transverse) with periodic sampling
    sysm = BatchedSystem([atoms], eng); nl = sysm.build_neighbor_list(skin=0.3)
    nl, res = fire_minimize(eng, sysm, nl, fmax=0.02, max_steps=2000)
    fr_eq, nl = run_md(eng, sysm, nl, T=T, steps=10000, dt_fs=0.5, gamma_ps=5.0, strain_rate_per_ps=0.0, record_every=1000, seed=1)
    steps = int(0.25 / (rate * 0.5e-3))
    fr, nl = run_md(eng, sysm, nl, T=T, steps=steps, dt_fs=0.5, gamma_ps=2.0, strain_rate_per_ps=rate, record_every=400, seed=2, log=lambda s: print(s, flush=True))
    out["md"] = {"eps": fr["eps_x"], "sxx": [s[0] for s in fr["sigma_xx"]], "T": fr["T"], "time_fs": fr["time_fs"], "wall_time_s": fr["wall_time_s"], "equilibration_T": fr_eq["T"]}
    json.dump(out, open(os.path.join(ROOT, "validation", "results", f"md_demo_T{int(T)}.json"), "w"), indent=1)
    print("done", fr["wall_time_s"])
