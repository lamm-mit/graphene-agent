"""Batched FIRE minimiser (device resident).  Follows the classic FIRE scheme as implemented in
ASE (Bitzek et al., PRL 97, 170201 (2006)) with per-structure adaptive time step and mixing,
per-atom step limiting, and per-structure convergence."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FireResult:
    converged: np.ndarray        # [B] bool
    iterations: int
    fmax: np.ndarray             # [B] final max force (eV/A)
    n_evaluations: int
    n_rebuilds: int
    wall_time: float
    failed: np.ndarray = None    # [B] bool: non-finite forces encountered
    soft: np.ndarray = None      # [B] bool: accepted by the stall criterion (fmax < stall_factor*tol)


def fire_minimize(engine, system, nl, fmax=0.02, max_steps=3000, dt=0.1, dt_max=1.0, dt_min=0.002,
                  maxstep=0.2, n_min=5, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99,
                  check_every=10, active=None, skin=0.3, log=None, stall_iters=500, stall_factor=5.0):
    """Minimise all (active) structures of `system` in place.  Returns (nl, FireResult).

    Stall criterion: a structure whose maximum force is below stall_factor*fmax but has not improved by
    10% over the last `stall_iters` iterations (soft modes, e.g. dangling edge chains after bond loss)
    is accepted as converged and flagged in FireResult.soft."""
    t0 = time.perf_counter()
    B, N = system.B, system.N
    dev, dt_ = system.device, system.dtype
    R = system.positions
    v = torch.zeros_like(R)
    dtb = torch.full((B,), dt, dtype=dt_, device=dev)
    alpha = torch.full((B,), alpha_start, dtype=dt_, device=dev)
    npos = torch.zeros(B, dtype=torch.long, device=dev)
    act = torch.ones(B, dtype=torch.bool, device=dev) if active is None else torch.as_tensor(active, device=dev)
    converged = ~act
    n_eval = 0
    n_rebuild = 0
    fmax_b = torch.zeros(B, dtype=dt_, device=dev)
    it = 0
    failed = torch.zeros(B, dtype=torch.bool, device=dev)
    soft = torch.zeros(B, dtype=torch.bool, device=dev)
    best = torch.full((B,), float("inf"), dtype=dt_, device=dev)
    last_improve = torch.zeros(B, dtype=torch.long, device=dev)
    for it in range(max_steps):
        # displacement-based neighbour-list rebuild check every iteration (a Verlet skin of `skin`
        # with half-skin criterion; atoms may move up to `maxstep` per iteration)
        if it > 0 and engine.needs_rebuild(R, system.cell, system.sid, nl):
            nl = system.build_neighbor_list(skin=skin)
            n_rebuild += 1
        out = engine.evaluate(R, system.cell, system.sid, nl, compute_stress=False)
        n_eval += 1
        F = out["forces"]
        fn2 = (F * F).sum(1)
        if it % check_every == 0:
            fmax_b = torch.sqrt(system.seg_max(fn2))
            bad = ~torch.isfinite(fmax_b)
            if bool(bad.any()):
                failed = failed | bad
                converged = converged | bad          # freeze diverged structures
                F = torch.where(torch.isfinite(F), F, torch.zeros_like(F))
                fn2 = (F * F).sum(1)
            improved = fmax_b < 0.9 * best
            best = torch.minimum(best, fmax_b)
            last_improve = torch.where(improved, torch.full_like(last_improve, it), last_improve)
            stalled = ((it - last_improve) > stall_iters) & (fmax_b < stall_factor * fmax) & ~converged
            soft = soft | stalled
            converged = converged | (fmax_b < fmax) | stalled
            if bool(converged.all()):
                break
        act_atom = (~converged)[system.sid].to(dt_)[:, None]
        P = system.seg_sum((F * v).sum(1))
        pos = P > 0
        npos = torch.where(pos, npos + 1, torch.zeros_like(npos))
        grow = pos & (npos > n_min)
        dtb = torch.where(grow, torch.clamp(dtb * f_inc, max=dt_max), dtb)
        alpha = torch.where(grow, alpha * f_alpha, alpha)
        dtb = torch.where(pos, dtb, torch.clamp(dtb * f_dec, min=dt_min))
        alpha = torch.where(pos, alpha, torch.full_like(alpha, alpha_start))
        v = torch.where(pos[system.sid][:, None], v, torch.zeros_like(v))
        v = v + dtb[system.sid][:, None] * F
        vnorm = torch.sqrt(system.seg_sum((v * v).sum(1)))
        fnorm = torch.sqrt(system.seg_sum(fn2))
        ratio = torch.where(fnorm > 0, vnorm / torch.clamp(fnorm, min=1e-30), torch.zeros_like(fnorm))
        a_at = alpha[system.sid][:, None]
        v = (1.0 - a_at) * v + a_at * ratio[system.sid][:, None] * F
        dr = dtb[system.sid][:, None] * v
        drn = torch.sqrt((dr * dr).sum(1, keepdim=True))
        dr = dr * torch.clamp(maxstep / torch.clamp(drn, min=1e-30), max=1.0)
        R = R + dr * act_atom
        system.positions = R
        if log is not None and it % 200 == 0:
            log(f"  fire it={it} fmax={fmax_b.max().item():.3e}")
    else:
        it = max_steps
    # final force check
    out = engine.evaluate(R, system.cell, system.sid, nl, compute_stress=False)
    n_eval += 1
    fn2 = (out["forces"] ** 2).sum(1)
    fmax_b = torch.sqrt(system.seg_max(fn2))
    converged = (fmax_b < fmax) | (~act) | soft
    failed = failed | ~torch.isfinite(fmax_b)
    system.positions = R.detach()
    return nl, FireResult(converged=converged.cpu().numpy(), iterations=it, fmax=fmax_b.cpu().numpy(),
                          n_evaluations=n_eval, n_rebuilds=n_rebuild, wall_time=time.perf_counter() - t0,
                          failed=failed.cpu().numpy(), soft=soft.cpu().numpy())
