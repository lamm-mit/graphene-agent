"""Optional finite-temperature molecular dynamics on the Torch REBO2+S engine (batched).

Velocity-Verlet with a Langevin thermostat (Bussi-Parrinello style discretisation of the
Ornstein-Uhlenbeck step) in ASE units (eV, A, amu; time unit = 10.1805 fs).  Optional constant
engineering strain rate along x (cell and positions scaled affinely every step).  Intended for
mechanism validation / temperature-effect checks, not for the main AQS campaign."""
from __future__ import annotations
import math, time
import numpy as np
import torch

KB = 8.617333262e-5        # eV/K
ASE_TIME_FS = 10.180505    # 1 ASE time unit in fs
M_C = 12.011               # amu


def run_md(engine, system, nl, T=300.0, steps=1000, dt_fs=0.5, gamma_ps=1.0, strain_rate_per_ps=0.0,
           record_every=50, skin=0.3, seed=0, log=None):
    dev, dt_ = system.device, system.dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = dt_fs / ASE_TIME_FS
    gamma = gamma_ps / 1000.0 * ASE_TIME_FS   # 1/ASE-time
    m = M_C
    N, B = system.N, system.B
    R = system.positions.detach().clone()
    # Maxwell-Boltzmann velocities (2D: in-plane only; z velocities zero to keep the sheet planar unless desired)
    sigma_v = math.sqrt(KB * T / m)
    v = torch.randn((N, 3), generator=g).to(dev, dt_) * sigma_v
    v[:, 2] = 0.0
    c1 = math.exp(-gamma * dt); c2 = math.sqrt((1 - c1 * c1) * KB * T / m)
    out = engine.evaluate(R, system.cell, system.sid, nl, compute_stress=False)
    F = out["forces"]
    frames = {"step": [], "time_fs": [], "T": [], "Epot": [], "Ekin": [], "eps_x": [], "sigma_xx": [], "positions": []}
    eps = 0.0
    de = strain_rate_per_ps / 1000.0 * dt_fs   # strain per step
    t0 = time.perf_counter()
    for it in range(steps + 1):
        if it % record_every == 0:
            out = engine.evaluate(R, system.cell, system.sid, nl, compute_stress=True)
            A = system.areas()
            sxx = (out["dE_deps"][:, 0, 0] / A).cpu().numpy() * 16.0217663
            ek = 0.5 * m * (v * v).sum()
            ndof = 2 * N
            Tinst = float(2 * ek / (ndof * KB))
            frames["step"].append(it); frames["time_fs"].append(it * dt_fs); frames["T"].append(Tinst); frames["Epot"].append(float(out["energy_total"]))
            frames["Ekin"].append(float(ek)); frames["eps_x"].append(eps); frames["sigma_xx"].append(sxx.tolist()); frames["positions"].append(R.cpu().numpy().astype(np.float32))
            if log: log(f"md step {it} t={it*dt_fs:.0f} fs T={Tinst:.0f} K eps={eps:.4f} sxx={sxx}")
        if it == steps:
            break
        # strain increment (affine)
        if de != 0.0:
            R[:, 0] *= (1 + eps + de) / (1 + eps)
            cell = system.cell.clone(); cell[:, 0, 0] *= (1 + eps + de) / (1 + eps); system.cell = cell
            eps += de
        # velocity Verlet + Langevin (BAOAB-like)
        v = v + 0.5 * dt * F / m
        R = R + 0.5 * dt * v
        v = c1 * v + c2 * torch.randn((N, 3), generator=g).to(dev, dt_)
        v[:, 2] = 0.0
        R = R + 0.5 * dt * v
        if engine.needs_rebuild(R, system.cell, system.sid, nl):
            system.positions = R
            nl = system.build_neighbor_list(skin=skin)
        out = engine.evaluate(R, system.cell, system.sid, nl, compute_stress=False)
        F = out["forces"]
        v = v + 0.5 * dt * F / m
    system.positions = R
    frames["wall_time_s"] = time.perf_counter() - t0
    return frames, nl
