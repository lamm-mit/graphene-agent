"""Athermal quasi-static (AQS) uniaxial tension driver for batches of periodic 2D carbon sheets.

Protocol (per strain step, all structures of the batch advance together, each with its own
strain schedule):
  1. increment the engineering strain eps_x of the loading axis (affine mapping of positions/cell);
  2. predict the transverse strain eps_y from the previous Poisson slope (affine);
  3. FIRE relaxation of all atomic coordinates (fmax criterion);
  4. optional transverse relaxation: secant iterations on eps_y until |sigma_yy| < tol;
  5. compute 2D stress sigma = (1/A) dE/deps (eV/A^2), per-atom energy / virial, bond topology;
  6. detect persistent topology changes (bond count / new broken bonds);
  7. adaptive refinement: the first bond-breaking event is bracketed by bisection of the strain
     increment down to d_strain_min (structure reverted to the last accepted state);
  8. continue until the structure no longer spans the loading direction / stress collapsed / eps_max.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from ..minimization.fire import fire_minimize
from ..topology import bond_pairs_from_neighbor_data, bond_keys, spanning_x, largest_fragment_fraction, coordination_from_bonds

EV_A2_TO_N_M = 16.0217663


@dataclass
class AQSConfig:
    d_strain: float = 0.005
    d_strain_elastic: float = None          # larger increment used before the first bond-breaking event (None = d_strain)
    d_strain_min: float = 0.00125
    refine_first_damage: bool = True
    max_strain: float = 0.40
    fmax: float = 0.02
    max_fire_steps: int = 1200
    trial_fire_steps: int = 400
    transverse_fire_steps: int = 400       # budget of each transverse (secant) sub-relaxation            # first relaxation phase; bond loss detected here triggers bisection without relaxing the avalanche
    transverse_relax: bool = True
    sigma_t_tol: float = 0.005          # eV/A^2  (0.08 N/m)
    max_transverse_iter: int = 3
    max_transverse_step: float = 0.005
    stop_stress_fraction: float = 0.05
    post_peak_strain_limit: float | None = 1.0  # None disables the administrative post-peak cap
    stop_when_not_spanning: bool = True
    skin: float = 0.3
    r_bond: float = 2.0
    strain_axis: int = 0
    zero_stress_relax: bool = True
    zero_stress_tol: float = 0.005
    max_steps: int = 400
    record_positions: bool = True


@dataclass
class AQSFrameData:
    eps_x: float
    eps_y: float
    sigma: np.ndarray          # [3] xx, yy, xy in eV/A^2
    energy: float
    n_bonds: int
    n_broken_new: int
    n_formed_new: int
    n_broken_cum: int
    fire_iterations: int
    fire_converged: bool
    transverse_iterations: int
    spanning: bool
    largest_fragment: float
    n_fragments: int
    wall_time: float
    attempted_step: int = 0
    max_force_eV_A: float = float('nan')


class AQSRunner:
    def __init__(self, engine, system, config: AQSConfig, log=print, on_done=None, on_frame=None):
        self.engine = engine
        self.on_done = on_done
        self.on_frame = on_frame
        self.attempted_step = 0
        self.sys = system
        self.cfg = config
        self.log = log or (lambda *a, **k: None)
        B = system.B
        self.B = B
        self.eps_x = np.zeros(B)
        self.eps_y = np.zeros(B)
        self.d_eps = np.full(B, config.d_strain_elastic if config.d_strain_elastic else config.d_strain)
        self.done = np.zeros(B, bool)
        self.reason = [""] * B
        self.peak_sigma = np.zeros(B)
        self.peak_strain = np.zeros(B)
        self.first_damage_strain = np.full(B, np.nan)
        self.first_damage_found = np.zeros(B, bool)
        self.slope_y = np.full(B, -0.17)
        self.C_t = None
        self.frames = [[] for _ in range(B)]
        self.positions_frames = [[] for _ in range(B)]
        self.cell_frames = [[] for _ in range(B)]
        self.peratom_energy_frames = [[] for _ in range(B)]
        self.peratom_virial_frames = [[] for _ in range(B)]
        self.bond_frames = [[] for _ in range(B)]
        self.coordination_frames = [[] for _ in range(B)]
        self.broken_bond_events = [[] for _ in range(B)]   # (eps_x, i, j, S, midpoint xy)
        self.n_broken_cum = np.zeros(B, int)
        self.stats = {"n_evaluations": 0, "fire_iterations": 0, "n_rebuilds": 0, "n_reverts": 0, "n_soft_converged": 0}
        self.nl = None
        self.L0 = None
        self.initial_bond_keys = [None] * B
        self.last_bond_keys = [None] * B
        self.saved_positions = None
        self.saved_cell = None
        self.saved_eps = None

    # ------------------------------------------------------------------ helpers
    def _affine(self, b_mask, new_eps_x, new_eps_y):
        """Map structures b (mask) from their current (eps_x, eps_y) to new strains."""
        sys = self.sys
        fx = np.ones(self.B)
        fy = np.ones(self.B)
        sel = np.where(b_mask)[0]
        fx[sel] = (1.0 + new_eps_x[sel]) / (1.0 + self.eps_x[sel])
        fy[sel] = (1.0 + new_eps_y[sel]) / (1.0 + self.eps_y[sel])
        fxt = torch.as_tensor(fx, dtype=sys.dtype, device=sys.device)
        fyt = torch.as_tensor(fy, dtype=sys.dtype, device=sys.device)
        R = sys.positions
        scale = torch.stack([fxt[sys.sid], fyt[sys.sid], torch.ones_like(fxt[sys.sid])], 1)
        sys.positions = R * scale
        cell = sys.cell.clone()
        cell[:, 0, :] = cell[:, 0, :] * fxt[:, None]
        cell[:, 1, :] = cell[:, 1, :] * fyt[:, None]
        sys.cell = cell
        self.eps_x[sel] = new_eps_x[sel]
        self.eps_y[sel] = new_eps_y[sel]

    def _minimize(self, active, max_steps=None, stall_iters=500):
        nl, res = fire_minimize(self.engine, self.sys, self.nl, fmax=self.cfg.fmax, max_steps=max_steps or self.cfg.max_fire_steps,
                                active=active, skin=self.cfg.skin, stall_iters=stall_iters)
        self.nl = nl
        if res.soft is not None:
            self.stats["n_soft_converged"] += int(res.soft.sum())
        self.stats["n_evaluations"] += res.n_evaluations
        self.stats["fire_iterations"] += res.iterations
        self.stats["n_rebuilds"] += res.n_rebuilds
        return res

    def _evaluate(self, per_atom=True):
        sys = self.sys
        if self.engine.needs_rebuild(sys.positions, sys.cell, sys.sid, self.nl):
            self.nl = sys.build_neighbor_list(skin=self.cfg.skin)
            self.stats["n_rebuilds"] += 1
        out = self.engine.evaluate(sys.positions, sys.cell, sys.sid, self.nl, compute_stress=True, per_atom=per_atom)
        self.stats["n_evaluations"] += 1
        A = sys.areas()
        dEde = out["dE_deps"]
        sigma = torch.stack([dEde[:, 0, 0], dEde[:, 1, 1], 0.5 * (dEde[:, 0, 1] + dEde[:, 1, 0])], 1) / A[:, None]
        out["sigma2d"] = sigma.cpu().numpy().astype(float)    # eV/A^2
        out["energy_np"] = out["energy"].cpu().numpy().astype(float)
        return out

    def _bonds(self, out):
        """bond arrays per structure from the current neighbour data."""
        sys, nl = self.sys, self.nl
        Mb = nl.Mb
        idx = nl.idx[:, :Mb].cpu().numpy()
        shift = nl.shift[:, :Mb].cpu().numpy().astype(int)
        r = out["r"][:, :Mb].cpu().numpy()
        mask = out["pair_mask"][:, :Mb].cpu().numpy()
        res = []
        for b in range(self.B):
            sl = sys.slice(b)
            off = sys.offsets[b]
            n = sys.natoms[b]
            i, j, S = bond_pairs_from_neighbor_data(idx[sl] - off, shift[sl], r[sl], mask[sl], n, self.cfg.r_bond)
            res.append((i, j, S))
        return res

    def _transverse_relax(self, active, out):
        """Secant iterations on eps_y until |sigma_yy| < tol for active structures."""
        cfg = self.cfg
        n_iter = np.zeros(self.B, int)
        sig = out["sigma2d"]
        for k in range(cfg.max_transverse_iter):
            need = active & (np.abs(sig[:, 1]) > cfg.sigma_t_tol) & ~self.done
            if not need.any():
                break
            d_eps_y = np.zeros(self.B)
            d_eps_y[need] = -sig[need, 1] / self.C_t[need]
            # small trust region: follow the current transverse branch continuously (the homogeneous
            # response of highly strained graphene can be multi-valued; large secant steps would hop branches)
            d_eps_y = np.clip(d_eps_y, -cfg.max_transverse_step, cfg.max_transverse_step)
            sig_old = sig.copy()
            eps_y_old = self.eps_y.copy()
            self._affine(need, self.eps_x, self.eps_y + d_eps_y)
            self._minimize(need, max_steps=cfg.transverse_fire_steps)
            out = self._evaluate(per_atom=False)
            sig = out["sigma2d"]
            # secant update of the transverse modulus estimate
            for b in np.where(need)[0]:
                de = self.eps_y[b] - eps_y_old[b]
                ds = sig[b, 1] - sig_old[b, 1]
                if abs(de) > 1e-9 and ds / de > 0.05 * self.C_t[b] and np.isfinite(ds / de):
                    self.C_t[b] = ds / de
                if self.eps_x[b] > self._eps_x_prev[b]:
                    self.slope_y[b] = float(np.clip((self.eps_y[b] - self._eps_y_prev[b]) / (self.eps_x[b] - self._eps_x_prev[b]), -0.6, 0.6))
                n_iter[b] += 1
        return out, n_iter

    # ------------------------------------------------------------------ zero-stress reference state
    def relax_zero_stress(self):
        cfg = self.cfg
        sys = self.sys
        self.nl = sys.build_neighbor_list(skin=cfg.skin)
        active = np.ones(self.B, bool)
        res = self._minimize(active)
        out = self._evaluate(per_atom=False)
        A = sys.areas().cpu().numpy()
        rho = sys.natoms / A
        self.C_t = 21.0 * (rho / 0.3819)          # eV/A^2 initial transverse modulus guess (graphene ~21 eV/A^2)
        C_x = self.C_t.copy()
        if cfg.zero_stress_relax:
            eps = np.zeros((self.B, 2))
            for k in range(10):
                sig = out["sigma2d"]
                need = (np.abs(sig[:, 0]) > cfg.zero_stress_tol) | (np.abs(sig[:, 1]) > cfg.zero_stress_tol)
                if not need.any():
                    break
                d = np.zeros((self.B, 2))
                d[need, 0] = -sig[need, 0] / C_x[need]
                d[need, 1] = -sig[need, 1] / self.C_t[need]
                d = np.clip(d, -0.02, 0.02)
                sig_old = sig.copy()
                new_x = self.eps_x + d[:, 0]
                new_y = self.eps_y + d[:, 1]
                self._affine(need, new_x, new_y)
                self._minimize(need)
                out = self._evaluate(per_atom=False)
                sig = out["sigma2d"]
                for b in np.where(need)[0]:
                    if abs(d[b, 0]) > 1e-9:
                        est = (sig[b, 0] - sig_old[b, 0]) / d[b, 0]
                        if np.isfinite(est) and est > 0.05 * C_x[b]:
                            C_x[b] = est
                    if abs(d[b, 1]) > 1e-9:
                        est = (sig[b, 1] - sig_old[b, 1]) / d[b, 1]
                        if np.isfinite(est) and est > 0.05 * self.C_t[b]:
                            self.C_t[b] = est
            self.log(f"  zero-stress relaxation: eps0 = {np.array2string(np.stack([self.eps_x, self.eps_y],1), precision=4)}, sigma = {np.array2string(out['sigma2d'][:, :2]*EV_A2_TO_N_M, precision=3)} N/m")
        # reset strain reference
        sys.cell0 = sys.cell.clone()
        self.eps_x[:] = 0.0
        self.eps_y[:] = 0.0
        self.L0 = sys.cells_numpy()[:, [0, 1], [0, 1]].copy()
        self.modulus_x_estimate = C_x
        out = self._evaluate(per_atom=True)
        return out

    # ------------------------------------------------------------------ main loop
    def run(self):
        cfg = self.cfg
        sys = self.sys
        t_start = time.perf_counter()
        out = self.relax_zero_stress()
        bonds = self._bonds(out)
        for b in range(self.B):
            keys = bond_keys(*bonds[b])
            self.initial_bond_keys[b] = keys
            self.last_bond_keys[b] = keys
        self._record(out, bonds, np.zeros(self.B, int), np.zeros(self.B, int), np.zeros(self.B, int), np.ones(self.B, bool),
                     np.zeros(self.B, int), np.ones(self.B, bool), t_start)
        self.saved_positions = sys.positions.detach().clone()
        self.saved_cell = sys.cell.clone()
        self.saved_eps = (self.eps_x.copy(), self.eps_y.copy())
        self.saved_last_keys = list(self.last_bond_keys)
        step = 0
        while not self.done.all() and step < cfg.max_steps:
            step += 1
            self.attempted_step = step
            active = ~self.done
            self._eps_x_prev = self.eps_x.copy()
            self._eps_y_prev = self.eps_y.copy()
            new_x = self.eps_x + self.d_eps * active
            # transverse prediction only when the transverse cell is relaxed afterwards; otherwise fixed transverse strain
            new_y = self.eps_y + (self.slope_y * self.d_eps * active if cfg.transverse_relax else 0.0)
            self._affine(active, new_x, new_y)
            # phase 1: short relaxation; if a bond already breaks here and the first damage is still being
            # bracketed, revert immediately (the damage avalanche need not be relaxed to bisect the strain)
            res = self._minimize(active, max_steps=cfg.trial_fire_steps)
            if cfg.refine_first_damage and (~self.first_damage_found & active).any():
                out_t = self._evaluate(per_atom=False)
                bonds_t = self._bonds(out_t)
                revert = np.zeros(self.B, bool)
                for b in np.where(active & ~self.first_damage_found)[0]:
                    keys = bond_keys(*bonds_t[b])
                    if len(np.setdiff1d(self.last_bond_keys[b], keys)) > 0 and self.d_eps[b] > cfg.d_strain_min * 1.001:
                        revert[b] = True
                if revert.any():
                    self._revert(revert)
                    self.d_eps[revert] *= 0.5
                    self.stats["n_reverts"] += int(revert.sum())
                    self.log(f"  step {step}: early bisection for {np.where(revert)[0].tolist()} -> d_eps={self.d_eps[revert]}")
                    active = active & ~revert
                    if not active.any():
                        continue
            if not bool(res.converged[active].all()):
                res = self._minimize(active)
            if res.failed is not None and res.failed.any():
                bad = res.failed & active
                if bad.any():
                    self.log(f"  step {step}: non-finite forces for {np.where(bad)[0].tolist()} -> reverting, marking numerical failure")
                    self._revert(bad)
                    for b in np.where(bad)[0]:
                        self.done[b] = True
                        self.reason[b] = "numerical failure (non-finite forces during minimisation)"
                    active = active & ~bad
                    if not active.any():
                        continue
            out = self._evaluate(per_atom=False)
            n_t = np.zeros(self.B, int)
            if cfg.transverse_relax:
                out, n_t = self._transverse_relax(active, out)
            out = self._evaluate(per_atom=True)
            bonds = self._bonds(out)
            # states accepted by the stall criterion while bonds were still breaking, or that lost the
            # load-bearing path, are relaxed to full convergence (no stall criterion) before being recorded
            redo = np.zeros(self.B, bool)
            for b in np.where(active)[0]:
                i_, j_, S_ = bonds[b]
                lost = len(np.setdiff1d(self.last_bond_keys[b], bond_keys(i_, j_, S_))) > 0
                soft_b = bool(res.soft[b]) if res.soft is not None else False
                if (soft_b and lost) or (lost and not spanning_x(i_, j_, S_, sys.natoms[b])):
                    redo[b] = True
            if redo.any():
                self._minimize(redo, stall_iters=10 ** 9)
                out = self._evaluate(per_atom=True)
                bonds = self._bonds(out)
            n_new_broken = np.zeros(self.B, int)
            n_new_formed = np.zeros(self.B, int)
            revert = np.zeros(self.B, bool)
            for b in np.where(active)[0]:
                keys = bond_keys(*bonds[b])
                broken = np.setdiff1d(self.last_bond_keys[b], keys)
                formed = np.setdiff1d(keys, self.last_bond_keys[b])
                n_new_broken[b] = len(broken)
                n_new_formed[b] = len(formed)
                if (cfg.refine_first_damage and n_new_broken[b] > 0 and not self.first_damage_found[b]
                        and self.d_eps[b] > cfg.d_strain_min * 1.001):
                    revert[b] = True
            if revert.any():
                self._revert(revert)
                self.d_eps[revert] *= 0.5
                self.stats["n_reverts"] += int(revert.sum())
                self.log(f"  step {step}: bisection for {np.where(revert)[0].tolist()} -> d_eps={self.d_eps[revert]}")
                if revert.all():
                    continue
                # structures not reverted proceed; re-evaluate after revert (positions of reverted changed)
                out = self._evaluate(per_atom=True)
                bonds = self._bonds(out)
                active = active & ~revert
            for b in np.where(active)[0]:
                keys = bond_keys(*bonds[b])
                broken = np.setdiff1d(self.last_bond_keys[b], keys)
                # record damage events (midpoints in current frame)
                if len(broken) > 0:
                    ib, jb, Sb = self.last_bonds[b]
                    lk = self.last_bond_keys[b]
                    pos = sys.positions_numpy(b)
                    cell = sys.cells_numpy()[b]
                    sel = np.isin(lk, broken)
                    for ii, jj, SS in zip(ib[sel], jb[sel], Sb[sel]):
                        mid = 0.5 * (pos[ii] + pos[jj] + SS @ cell)
                        self.broken_bond_events[b].append((float(self.eps_x[b]), int(ii), int(jj), SS.tolist(), mid[:2].tolist()))
                    self.n_broken_cum[b] += len(broken)
                    if not self.first_damage_found[b]:
                        self.first_damage_found[b] = True
                        self.first_damage_strain[b] = self.eps_x[b]
                        self.d_eps[b] = cfg.d_strain
                self.last_bond_keys[b] = keys
            self.last_bonds = bonds
            fire_conv = res.converged
            self._record(out, bonds, n_new_broken, n_new_formed, np.full(self.B, res.iterations), fire_conv, n_t, active, t_start)
            # termination + bookkeeping
            sig = out["sigma2d"]
            for b in np.where(active)[0]:
                s = sig[b, 0]
                if s > self.peak_sigma[b]:
                    self.peak_sigma[b] = s
                    self.peak_strain[b] = self.eps_x[b]
                fr = self.frames[b][-1]
                if cfg.stop_when_not_spanning and not fr.spanning:
                    self.done[b] = True
                    self.reason[b] = "not spanning (fractured)"
                elif self.n_broken_cum[b] > 0 and s < cfg.stop_stress_fraction * self.peak_sigma[b]:
                    self.done[b] = True
                    self.reason[b] = "stress collapsed"
                elif self.eps_x[b] >= cfg.max_strain - 1e-12:
                    self.done[b] = True
                    self.reason[b] = "max strain reached"
                elif (cfg.post_peak_strain_limit is not None and self.n_broken_cum[b] > 0
                      and self.eps_x[b] >= self.peak_strain[b] + cfg.post_peak_strain_limit - 1e-12):
                    self.done[b] = True
                    self.reason[b] = "post-peak strain limit reached (load still carried)"
                elif s < 0 and self.n_broken_cum[b] > 0 and self.eps_x[b] > self.peak_strain[b]:
                    self.done[b] = True
                    self.reason[b] = "compressive after fracture"
            # per-structure completion callback (early persistence)
            if self.on_done is not None:
                for b in np.where(active & self.done)[0]:
                    try:
                        self.on_done(int(b), self.result_for(int(b)))
                    except Exception as e:
                        self.log(f"  on_done callback failed for {b}: {e!r}")
            # save accepted state
            self.saved_positions = sys.positions.detach().clone()
            self.saved_cell = sys.cell.clone()
            self.saved_eps = (self.eps_x.copy(), self.eps_y.copy())
            self.saved_last_keys = list(self.last_bond_keys)
            self.saved_last_bonds = list(self.last_bonds)
            self.log(f"step {step:3d} " + " | ".join(
                f"{sys.names[b][:14]}: e={self.eps_x[b]:.4f} s={sig[b,0]*EV_A2_TO_N_M:6.2f}N/m nb={n_new_broken[b]:3d}{'*' if self.done[b] else ''}"
                for b in range(self.B)) + f"  fire_it={res.iterations} t={time.perf_counter()-t_start:.0f}s")
        for b in range(self.B):
            if not self.done[b]:
                self.reason[b] = "max steps reached"
                self.done[b] = True
                if self.on_done is not None:
                    try:
                        self.on_done(int(b), self.result_for(int(b)))
                    except Exception as e:
                        self.log(f"  on_done callback failed for {b}: {e!r}")
        return self.results()

    def _revert(self, revert):
        sys = self.sys
        sel = torch.as_tensor(revert, device=sys.device)[sys.sid]
        R = sys.positions.clone()
        R[sel] = self.saved_positions[sel]
        sys.positions = R
        cell = sys.cell.clone()
        cell[torch.as_tensor(revert, device=sys.device)] = self.saved_cell[torch.as_tensor(revert, device=sys.device)]
        sys.cell = cell
        self.eps_x[revert] = self.saved_eps[0][revert]
        self.eps_y[revert] = self.saved_eps[1][revert]
        for b in np.where(revert)[0]:
            self.last_bond_keys[b] = self.saved_last_keys[b]
        if hasattr(self, "saved_last_bonds"):
            for b in np.where(revert)[0]:
                self.last_bonds[b] = self.saved_last_bonds[b]

    def _record(self, out, bonds, n_new_broken, n_new_formed, fire_it, fire_conv, n_t, active, t_start):
        sys = self.sys
        if not hasattr(self, "last_bonds"):
            self.last_bonds = bonds
        sig = out["sigma2d"]
        E = out["energy_np"]
        e_at = out["energy_per_atom"].detach().cpu().numpy().astype(np.float32)
        w = out["virial_per_atom"].detach().cpu().numpy()
        P = sys.positions_numpy()
        C = sys.cells_numpy()
        for b in range(self.B):
            if not active[b]:
                continue
            sl = sys.slice(b)
            i, j, S = bonds[b]
            n = sys.natoms[b]
            span = spanning_x(i, j, S, n)
            lf, nfrag = largest_fragment_fraction(i, j, n)
            fr = AQSFrameData(eps_x=float(self.eps_x[b]), eps_y=float(self.eps_y[b]), sigma=sig[b].copy(), energy=float(E[b]),
                              n_bonds=int(len(i)), n_broken_new=int(n_new_broken[b]), n_formed_new=int(n_new_formed[b]),
                              n_broken_cum=int(self.n_broken_cum[b]), fire_iterations=int(fire_it[b]),
                              fire_converged=bool(fire_conv[b]), transverse_iterations=int(n_t[b]), spanning=span,
                              largest_fragment=lf, n_fragments=nfrag, wall_time=time.perf_counter() - t_start,
                              attempted_step=self.attempted_step,
                              max_force_eV_A=float(torch.sqrt((out['forces'][sl] ** 2).sum(1)).max().detach().cpu()))
            self.frames[b].append(fr)
            if self.cfg.record_positions:
                self.positions_frames[b].append(P[b].astype(np.float32))
                self.cell_frames[b].append(C[b].astype(np.float64))
                self.peratom_energy_frames[b].append(e_at[sl])
                wb = w[sl]
                self.peratom_virial_frames[b].append(np.stack([wb[:, 0, 0], wb[:, 1, 1], 0.5 * (wb[:, 0, 1] + wb[:, 1, 0])], 1).astype(np.float32))
                self.bond_frames[b].append(np.stack([i, j], 1).astype(np.int32))
                self.coordination_frames[b].append(coordination_from_bonds(i, j, n).astype(np.int8))
            if self.on_frame is not None:
                self.on_frame(self, b, fr)

    def results(self):
        return [self.result_for(b) for b in range(self.B)]

    def result_for(self, b):
        if True:
            fr = self.frames[b]
            d = {
                "name": self.sys.names[b],
                "n_atoms": int(self.sys.natoms[b]),
                "eps_x": np.array([f.eps_x for f in fr]),
                "eps_y": np.array([f.eps_y for f in fr]),
                "sigma_xx": np.array([f.sigma[0] for f in fr]),
                "sigma_yy": np.array([f.sigma[1] for f in fr]),
                "sigma_xy": np.array([f.sigma[2] for f in fr]),
                "energy": np.array([f.energy for f in fr]),
                "n_bonds": np.array([f.n_bonds for f in fr]),
                "n_broken_new": np.array([f.n_broken_new for f in fr]),
                "n_formed_new": np.array([f.n_formed_new for f in fr]),
                "n_broken_cum": np.array([f.n_broken_cum for f in fr]),
                "fire_iterations": np.array([f.fire_iterations for f in fr]),
                "fire_converged": np.array([f.fire_converged for f in fr]),
                "transverse_iterations": np.array([f.transverse_iterations for f in fr]),
                "spanning": np.array([f.spanning for f in fr]),
                "largest_fragment": np.array([f.largest_fragment for f in fr]),
                "n_fragments": np.array([f.n_fragments for f in fr]),
                "wall_time": np.array([f.wall_time for f in fr]),
                "attempted_step": np.array([f.attempted_step for f in fr]),
                "max_force_eV_A": np.array([f.max_force_eV_A for f in fr]),
                "L0": self.L0[b].copy(),
                "peak_sigma": float(self.peak_sigma[b]),
                "peak_strain": float(self.peak_strain[b]),
                "first_damage_strain": float(self.first_damage_strain[b]),
                "termination": self.reason[b],
                "broken_bond_events": self.broken_bond_events[b],
                "positions": self.positions_frames[b],
                "cells": self.cell_frames[b],
                "peratom_energy": self.peratom_energy_frames[b],
                "peratom_virial": self.peratom_virial_frames[b],
                "bonds": self.bond_frames[b],
                "coordination": self.coordination_frames[b],
                "stats": dict(self.stats),
                "config": asdict(self.cfg),
            }
            return d
