"""
TorchRebo2Scr -- exact PyTorch implementation of the screened second-generation REBO
potential (REBO2+S, "Rebo2Scr") for carbon.

The functional form, cut-off radii, screening function, angular spline, bicubic P table and
tricubic F table follow the authoritative Atomistica implementation
(src/potentials/bop/rebo2/bop_kernel_rebo2.f90 with SCREENING, NUM_NEIGHBORS, ALT_DIHEDRAL
and trig_off_t cutoffs; default parameters, dihedral term disabled by default exactly as in
`atomistica.Rebo2Scr()`).  Forces are obtained by automatic differentiation of the energy,
F = -dE/dR; the virial/stress from the derivative with respect to an affine strain of all
pair vectors.

Energy (carbon only):
    E = sum_{i<j} fc_ar(r_ij) [ V_R(r_ij) + bbar_ij V_A(r_ij) ]
    bbar_ij = 1/2 ( b_ij + b_ji + F_CC(N_i^t, N_j^t, N_ij^conj) )
    b_ij    = ( 1 + sum_{k!=i,j} fc_bo(r_ik) g(cos theta_jik, N_i^t) + P_CC(N_i^H, N_i^C) )^{-1/2}
with the screened cut-off functions
    fc_x(r_ij) = (1 - fCin(r_ij)) S_ij fCx(r_ij) + fCin(r_ij),   x in {ar, bo, nc}
    S_ij = prod_k S(C_ijk),  S(C) = exp[-((Cmax - C)/(C - Cmin))^2]  (Cmin < C < Cmax), 0 (C<=Cmin), 1 (C>=Cmax)
    C_ijk = (2 (x_ik + x_jk) - (x_ik - x_jk)^2 - 1) / (1 - (x_ik - x_jk)^2),  x_ik = r_ik^2 / r_ij^2
(k restricted to atoms whose projection lies between i and j and r_ik < r_ij).

Devices: cpu (float64 reference / float32), mps (float32), cuda (float32/float64).
"""
from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..reference.rebo2scr_parameters import Rebo2ScrParameters, CITATIONS
from .neighborlist import DenseNeighborList, build_dense_neighbor_list


# --------------------------------------------------------------------------------------
# small autograd helpers
# --------------------------------------------------------------------------------------
class _ClampMaxStraightThrough(torch.autograd.Function):
    """value = min(x, hi); gradient passes through unchanged.

    Atomistica caps the neighbour counts (N_i <= 4, N_i^t <= 3, N^conj <= 8) in value but keeps
    using the uncapped derivative in the force expression.  To reproduce the reference forces
    bit-for-bit in over-coordinated environments we mirror that behaviour here.  In graphene-like
    carbon (N <= 3) the caps are never active and forces are exact gradients of the energy."""

    @staticmethod
    def forward(ctx, x, hi):
        return torch.clamp(x, max=hi)

    @staticmethod
    def backward(ctx, g):
        return g, None


def clamp_max_st(x, hi):
    return _ClampMaxStraightThrough.apply(x, hi)


def select_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class EngineInfo:
    potential: str = "screened second-generation REBO (REBO2+S / Rebo2Scr)"
    implementation: str = "TorchRebo2Scr (PyTorch autograd port of Atomistica Rebo2Scr kernel)"
    reference_implementation: str = "Atomistica 1.2.7 Rebo2Scr (commit 14a86f2c, 2025-10-21)"
    parameterization: str = "Brenner 2002 + Pastewka 2008 screening; Atomistica default tables; dihedral=False"
    citations: list = field(default_factory=lambda: list(CITATIONS))
    parameter_checksum: str = ""
    device: str = ""
    dtype: str = ""
    torch_version: str = torch.__version__
    units: str = "eV, Angstrom; 2D stress in eV/A^2 (x16.0218 = N/m)"


class TorchRebo2Scr:
    """Batched, device-resident REBO2+S energy/force/virial engine."""

    def __init__(self, device="auto", dtype=torch.float32, params: Optional[Rebo2ScrParameters] = None,
                 with_dihedral: bool = False):
        self.device = select_device(device)
        if self.device.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS does not support float64; use float32 (fast mode) or cpu float64 (reference mode)")
        self.dtype = dtype
        self.p = params or Rebo2ScrParameters()
        if with_dihedral:
            raise NotImplementedError("The dihedral term (with_dihedral=True) is not part of the default "
                                      "Rebo2Scr potential and is not implemented in the Torch port.")
        p = self.p
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)
        self.g_c1 = t(p.g_coeff1)          # [6, 3]
        self.g_c2 = t(p.g_coeff2)          # [6, 3]
        self.g_th1 = float(p.g_theta[1])
        self.g_th2 = float(p.g_theta[2])
        self.F_coeff = t(p.Fcc_coeff.reshape(4 * 4 * 9, 64))   # box-major: bx*36 + by*9 + bz
        self.P_coeff = t(p.Pcc_coeff.reshape(5 * 5, 16))
        self.rc = p.r_cut                      # 4.0 A : screening / neighbour-count range
        self.rc2 = p.r_cut ** 2
        self.rs = max(p.ar_r2, p.bo_r2)        # 2.82 A: everything else (pair energy, bond order)
        self.rs2 = self.rs ** 2
        self.ln_thr = p.screening_threshold
        self.info = EngineInfo(parameter_checksum=p.checksum(), device=str(self.device), dtype=str(dtype))
        self.timings = {"n_calls": 0, "forward_s": 0.0, "backward_s": 0.0}

    # ------------------------------------------------------------------ helpers
    def _trig(self, r, r1, r2):
        x = torch.clamp((r - r1) / (r2 - r1), 0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def _fconj(self, x):
        u = torch.clamp(x - 2.0, 0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * u))

    def _g_spline(self, coeff, c):
        """coeff [6,3]; c = cos(theta) tensor -> polynomial value."""
        j = torch.where(c < self.g_th1, 0, torch.where(c < self.g_th2, 1, 2))
        cf = coeff[:, j]            # [6, ...]
        out = cf[5]
        for pw in range(4, -1, -1):
            out = out * c + cf[pw]
        return out

    def _g(self, c, n):
        """angular function with coordination-dependent switching (carbon)."""
        v1 = self._g_spline(self.g_c1, c)
        v2 = self._g_spline(self.g_c2, c)
        u = torch.clamp((n - 3.2) / 0.5, 0.0, 1.0)
        s = 0.5 * (1.0 + torch.cos(math.pi * u))
        return v1 * (1.0 - s) + v2 * s

    def _powers(self, x):
        one = torch.ones_like(x)
        return torch.stack([one, x, x * x, x * x * x], dim=-1)   # [..., 4]

    def _F(self, ni, nj, nc):
        """tricubic F_CC(N_i^t, N_j^t, N^conj); table boxes 4 x 4 x 9."""
        bi = torch.clamp(ni.detach().floor().long(), 0, 3)
        bj = torch.clamp(nj.detach().floor().long(), 0, 3)
        bc = torch.clamp(nc.detach().floor().long(), 0, 8)
        x1 = ni - bi.to(ni.dtype)
        x2 = nj - bj.to(nj.dtype)
        x3 = nc - bc.to(nc.dtype)
        box = bi * 36 + bj * 9 + bc
        cf = self.F_coeff[box].reshape(*box.shape, 4, 4, 4)
        return torch.einsum("...ijk,...i,...j,...k->...", cf, self._powers(x1), self._powers(x2), self._powers(x3))

    def _P(self, nh, nc):
        """bicubic P_CC(N_H, N_C); boxes 5 x 5."""
        bh = torch.clamp(nh.detach().floor().long(), 0, 4)
        bc = torch.clamp(nc.detach().floor().long(), 0, 4)
        x1 = nh - bh.to(nh.dtype)
        x2 = nc - bc.to(nc.dtype)
        cf = self.P_coeff[bh * 5 + bc].reshape(*bh.shape, 4, 4)
        return torch.einsum("...ij,...i,...j->...", cf, self._powers(x1), self._powers(x2))

    # ------------------------------------------------------------------ neighbour list
    def build_neighbor_list(self, positions_list, cells, pbc, skin=0.3):
        return build_dense_neighbor_list(positions_list, cells, pbc, cutoff=self.rc, skin=skin,
                                         device=self.device, dtype=self.dtype, short_cutoff=self.rs)

    def needs_rebuild(self, R, cell, sid, nl: DenseNeighborList):
        """True if any atom moved (relative to the affine map of the reference cell) by more than skin/2."""
        with torch.no_grad():
            # map reference positions through the current cell (fractional coordinates preserved)
            inv_ref = torch.linalg.inv(nl.ref_cell.cpu()).to(nl.ref_cell.device, nl.ref_cell.dtype)
            frac = torch.einsum("ni,nij->nj", nl.ref_positions, inv_ref[sid])
            mapped = torch.einsum("ni,nij->nj", frac, cell[sid])
            disp = (R - mapped).norm(dim=1).max()
            return bool(disp > 0.5 * nl.skin)

    # ------------------------------------------------------------------ energy
    def energy(self, R, cell, sid, nl: DenseNeighborList, eps=None, return_pair_terms=False):
        """Total energy of the super-system (sum over all structures).

        R    [N,3] positions (may require grad)
        cell [B,3,3]
        sid  [N] structure index per atom
        eps  optional [B,3,3] strain tensor applied to all pair vectors (for the virial)
        """
        p = self.p
        N, M = nl.idx.shape
        Rj = R[nl.idx]                                                # [N,M,3]
        d = Rj - R[:, None, :] + torch.einsum("nmk,nkl->nml", nl.shift, cell[sid])
        if eps is not None:
            d = d + torch.einsum("nml,nkl->nmk", d, eps[sid])
        r2 = (d * d).sum(-1)
        mask = nl.mask & (r2 < self.rc2)
        r2s = torch.where(mask, r2, torch.ones_like(r2))
        r = torch.sqrt(r2s)
        u = torch.where(mask[..., None], d / r[..., None], torch.zeros_like(d))   # unit vectors (0 for padding)

        # ---------------- screening function S_ij (only needed for r_ij >= in_r1, harmless elsewhere)
        dij = d[:, :, None, :]
        dik = d[:, None, :, :]
        rij2 = r2s[:, :, None]
        rik2 = r2s[:, None, :]
        dot_ij_ik = (dij * dik).sum(-1)                                # [N,M,M]
        rjk2 = ((dik - dij) ** 2).sum(-1)
        dot_ij_jk = dot_ij_ik - rij2
        eye = torch.eye(M, dtype=torch.bool, device=R.device)[None]
        tmask = (mask[:, :, None] & mask[:, None, :] & ~eye
                 & (rik2 < p.C_dr_cut * rij2)
                 & (dot_ij_ik > p.dot_threshold) & (dot_ij_jk < -p.dot_threshold))
        xik = rik2 / rij2
        xjk = rjk2 / rij2
        xm = xik - xjk
        xp = xik + xjk
        den = 1.0 - xm * xm
        # Degenerate geometries (k projecting exactly onto i or j) give den -> 0 and C -> +inf, i.e. an
        # unscreened contribution of exactly 1.  They are excluded from the product (identical value)
        # to keep the autograd backward free of 0/0 (this only matters in float32 on perfect lattices).
        tmask = tmask & (den > 1e-6)
        den_s = torch.where(tmask, den, torch.ones_like(den))
        C = (2.0 * xp - xm * xm - 1.0) / den_s
        screened = (tmask & (C <= p.Cmin)).any(-1)                    # [N,M]
        pmask = tmask & (C > p.Cmin) & (C < p.Cmax)
        Cs = torch.where(pmask, C, torch.full_like(C, 0.5 * (p.Cmin + p.Cmax)))
        ratio = (p.Cmax - Cs) / torch.clamp(Cs - p.Cmin, min=1e-6)
        logS = torch.where(pmask, -(ratio * ratio), torch.zeros_like(C)).sum(-1)   # [N,M]
        below = logS < self.ln_thr
        S = torch.where(screened | below, torch.zeros_like(logS),
                        torch.exp(torch.where(below, torch.zeros_like(logS), logS)))

        # ---------------- cut-off functions (long list: only the neighbour-count cutoff is needed)
        fin = self._trig(r, p.in_r1, p.in_r2)
        fnc = self._trig(r, p.nc_r1, p.nc_r2)
        one_m_fin = 1.0 - fin
        mf = mask.to(r.dtype)
        fc_nc = torch.where(mask, one_m_fin * S * fnc + fin, torch.zeros_like(r))
        NN = fc_nc.sum(1)                                              # [N] total neighbour count (uncapped)

        # ---------------- short list (first Mb slots, sorted by distance at build time)
        Mb = nl.Mb
        ds, rs_, us = d[:, :Mb], r[:, :Mb], u[:, :Mb]
        masks = mask[:, :Mb] & (r2[:, :Mb] < self.rs2)
        mfs = masks.to(r.dtype)
        Ss, fins, omfs = S[:, :Mb], fin[:, :Mb], one_m_fin[:, :Mb]
        far = self._trig(rs_, p.ar_r1, p.ar_r2)
        fbo = self._trig(rs_, p.bo_r1, p.bo_r2)
        zero_s = torch.zeros_like(rs_)
        fc_ar = torch.where(masks, omfs * Ss * far + fins, zero_s)
        fc_bo = torch.where(masks, omfs * Ss * fbo + fins, zero_s)
        fc_nc_s = fc_nc[:, :Mb]

        # ---------------- coordination numbers seen from each short pair (i,j)
        Ni_excl = NN[:, None] - fc_nc_s                                # [N,Mb] N_i without j
        Ni_cap = clamp_max_st(Ni_excl, 4.0)                            # species cap (C)
        nti = Ni_cap                                                   # N_H = 0 for pure carbon

        # ---------------- bond order b_ij
        cos = torch.einsum("nmk,nlk->nml", us, us)                     # cos(theta_jik), j=m, k=l  [N,Mb,Mb]
        eyes = torch.eye(Mb, dtype=torch.bool, device=R.device)[None]
        kmask = (masks[:, None, :] & ~eyes).to(r.dtype)                # k != j, valid
        gval = self._g(cos, nti[:, :, None])
        zeta = (fc_bo[:, None, :] * gval * kmask).sum(-1)              # [N,Mb]
        Pij = self._P(torch.zeros_like(Ni_cap), Ni_cap)
        b = torch.rsqrt(torch.clamp(1.0 + zeta + Pij, min=1e-12))      # (1 + zeta)^(-1/2); clamp only guards padding slots

        # ---------------- conjugation  N^conj = (sum_{k!=j} fc_nc_ik F(x_ik))^2 + (j-side)^2
        NNk = NN[nl.idx]                                               # [N,M] neighbour count of k
        fx = self._fconj(NNk - fc_nc)                                  # F_conj(x_ik)
        conj_tot = (fc_nc * fx).sum(1)                                 # [N]
        nconj_dir = conj_tot[:, None] - (fc_nc * fx)[:, :Mb]           # [N,Mb] exclude j

        revs = nl.rev_short.reshape(-1)
        b_rev = b.reshape(-1)[revs].reshape(N, Mb)
        nconj_rev = nconj_dir.reshape(-1)[revs].reshape(N, Mb)
        nti_rev = nti.reshape(-1)[revs].reshape(N, Mb)
        nconj = clamp_max_st(nconj_dir * nconj_dir + nconj_rev * nconj_rev, 8.0)
        Fij = self._F(clamp_max_st(nti, 3.0), clamp_max_st(nti_rev, 3.0), nconj)
        bbar = 0.5 * (b + b_rev + Fij)

        # ---------------- pair potentials
        VR = (1.0 + p.Q / rs_) * p.A * torch.exp(-p.alpha * rs_)
        VA = -(p.B1 * torch.exp(-p.beta1 * rs_) + p.B2 * torch.exp(-p.beta2 * rs_) + p.B3 * torch.exp(-p.beta3 * rs_))
        Epair = torch.where(masks, fc_ar * (VR + bbar * VA), zero_s)   # directed pair energy (counted twice) [N,Mb]
        E = 0.5 * Epair.sum()
        if return_pair_terms:
            return E, dict(Epair=Epair, d=d, r=r, mask=mask, S=S, fc_ar=fc_ar, fc_bo=fc_bo, fc_nc=fc_nc,
                           b=b, bbar=bbar, Fij=Fij, nconj=nconj, nti=nti, NN=NN, zeta=zeta, Mb=Mb)
        return E

    # ------------------------------------------------------------------ energy + forces (+ virial)
    def evaluate(self, R, cell, sid, nl: DenseNeighborList, compute_stress=True, per_atom=False,
                 n_structures=None):
        """Return dict with total energy per structure, forces, virial dE/deps per structure,
        optionally per-atom energies and per-atom virials (pair-vector decomposition)."""
        t0 = time.perf_counter()
        B = int(cell.shape[0]) if n_structures is None else n_structures
        R = R.detach().requires_grad_(True)
        eps = None
        if compute_stress:
            eps = torch.zeros((B, 3, 3), dtype=R.dtype, device=R.device, requires_grad=True)
        E, terms = self.energy(R, cell, sid, nl, eps=eps, return_pair_terms=True)
        t1 = time.perf_counter()
        inputs = [R] + ([eps] if compute_stress else [])
        if per_atom:
            inputs.append(terms["d"])
        grads = torch.autograd.grad(E, inputs, allow_unused=True)
        t2 = time.perf_counter()
        out = {}
        F = -grads[0]
        out["forces"] = F
        e_atom = 0.5 * terms["Epair"].detach().sum(1)
        out["energy_per_atom"] = e_atom
        out["energy"] = torch.zeros(B, dtype=R.dtype, device=R.device).index_add_(0, sid, e_atom)
        out["energy_total"] = E.detach()
        if compute_stress:
            out["dE_deps"] = grads[1].detach()                          # [B,3,3]  (V * sigma)
        if per_atom:
            G = grads[-1]                                                # dE/dd  [N,M,3]
            d = terms["d"].detach()
            w = d[..., :, None] * G[..., None, :]                        # [N,M,3,3] per directed pair
            w_rev = w.reshape(-1, 3, 3)[nl.rev.reshape(-1)].reshape(w.shape)
            out["virial_per_atom"] = 0.5 * (w + w_rev).sum(1)           # [N,3,3]; sum over atoms = dE/deps
        out["coordination"] = terms["NN"].detach()
        out["S"] = terms["S"].detach()
        out["r"] = terms["r"].detach()
        out["pair_mask"] = terms["mask"]
        self.timings["n_calls"] += 1
        self.timings["forward_s"] += t1 - t0
        self.timings["backward_s"] += t2 - t1
        return out

    # ------------------------------------------------------------------ convenience for a single ASE Atoms
    def evaluate_atoms(self, atoms, per_atom=False, skin=0.0, nl=None):
        cell = torch.as_tensor(np.asarray(atoms.cell, float)[None], dtype=self.dtype, device=self.device)
        pbc = list(atoms.pbc)
        if nl is None:
            nl = self.build_neighbor_list([atoms.positions], np.asarray(atoms.cell, float)[None], pbc, skin=skin)
        R = torch.as_tensor(atoms.positions, dtype=self.dtype, device=self.device)
        sid = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        out = self.evaluate(R, cell, sid, nl, compute_stress=True, per_atom=per_atom)
        return out, nl
