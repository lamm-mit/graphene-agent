"""
Published parameters and lookup tables of the screened second-generation REBO potential
(REBO2+S) for carbon, transcribed from the authoritative Atomistica implementation
(src/potentials/bop/rebo2/rebo2_type.f90, rebo2_default_tables.f90, rebo2_db.f90,
table2d.f90, table3d.f90; Atomistica commit 14a86f2c, 2025-10-21).

References
----------
Brenner, Shenderova, Harrison, Stuart, Ni, Sinnott, J. Phys.: Condens. Matter 14, 783 (2002)
Pastewka, Pou, Perez, Gumbsch, Moseler, Phys. Rev. B 78, 161402(R) (2008)  [screening]
Pastewka, Klemenz, Gumbsch, Moseler, Phys. Rev. B 87, 205410 (2013)         [screening, details]

The spline/table coefficient construction below replicates the Fortran routines
`rebo2_db_make_cc_g_spline`, `table2d_init` and `table3d_init` exactly (same linear systems),
so the resulting piecewise polynomials agree with Atomistica to floating-point round-off.

Units: eV, Angstrom.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np

# --------------------------------------------------------------------------------------
# Scalar parameters (C-C only; the production study contains only carbon atoms)
# --------------------------------------------------------------------------------------

CC_B1 = 12388.79197798
CC_B2 = 17.56740646509
CC_B3 = 30.71493208065
CC_BETA1 = 4.7204523127
CC_BETA2 = 1.4332132499
CC_BETA3 = 1.3826912506
CC_Q = 0.3134602960833
CC_A = 10953.544162170
CC_ALPHA = 4.7465390606595

# g(cos theta) knot data, Brenner 2002 Table 3 (as stored in Atomistica)
CC_G_THETA = np.array([-1.0, -0.5, -1.0 / 3.0, 0.0, 0.5, 1.0])
# NOTE: in rebo2_type.f90 the knot values below are written as default-kind (single precision)
# Fortran literals and are therefore rounded to float32 before being promoted to double.  We
# replicate that rounding deliberately so that the angular spline is identical to Atomistica's.
def _f32(vals):
    return np.asarray(np.asarray(vals, dtype=np.float32), dtype=np.float64)
CC_G_G1 = _f32([-0.01, 0.05280, 0.09733, 0.37545, 2.0014, 8.0])
CC_G_DG1 = _f32([0.10400, 0.17000, 0.40000, 0.0, 0.0, 0.0])
CC_G_D2G1 = _f32([0.00000, 0.37000, 1.98000, 0.0, 0.0, 0.0])
CC_G_G2 = _f32([0.0, 0.0, 0.09733, 0.271856, 0.416335, 1.0])

# Screening parameters (Pastewka 2008)
CMIN = 1.00
CMAX = 2.00
SCREENING_THRESHOLD = math.log(1e-6)
DOT_THRESHOLD = 1e-10

# Cut-off radii (screened variant, rebo2_type.f90 under #ifdef SCREENING)
CC_IN_R1, CC_IN_R2 = 1.95, 2.25          # inner (unscreened-type) cutoff
CC_AR_R1, CC_AR_R2 = 2.179347, 2.819732  # attractive/repulsive pair cutoff
CC_BO_R1, CC_BO_R2 = 1.866344, 2.758372  # bond-order (angular sum) cutoff
CC_NC_R1, CC_NC_R2 = 1.217335, 4.000000  # neighbour-count / conjugation cutoff

# Derived screening constants (rebo2_db.f90)
DC = CMAX - CMIN
C_DR_CUT = CMAX ** 2 / (4.0 * (CMAX - 1.0))   # = 1.0 for Cmax = 2

# Maximum interaction range of the potential (all cutoffs)
R_CUT = max(CC_IN_R2, CC_AR_R2, CC_BO_R2, CC_NC_R2)  # 4.0 A

# Neighbour-list range requested by Atomistica: sqrt(C_dr_cut) * max cutoff = 4.0 A
R_NEIGHBOR = math.sqrt(C_DR_CUT) * R_CUT

# Bond-order exponent (conpe = -0.5 for carbon)
CONPE = -0.5

# g(theta) switching between the two carbon angular splines (rebo2_func.f90, subroutine g)
G_N_LOW, G_N_HIGH = 3.2, float(np.float32(3.7))  # 3.7 is a single-precision literal in the Fortran source


# --------------------------------------------------------------------------------------
# Default lookup tables (rebo2_default_tables.f90) -- reproduced statement by statement
# --------------------------------------------------------------------------------------

def default_Fcc_table():
    """Return F, dFdi, dFdj, dFdk with shape (5, 5, 10) (indices 0..4, 0..4, 0..9)."""
    F = np.zeros((5, 5, 10))
    dFdi = np.zeros((5, 5, 10))
    dFdj = np.zeros((5, 5, 10))
    dFdk = np.zeros((5, 5, 10))
    # Values from Table 4 (with Atomistica-specific modifications, see Fortran source)
    F[1, 1, 0] = 0.105000
    F[1, 1, 1] = -0.0041775
    F[1, 1, 2:9] = -0.0160856
    F[2, 2, 0] = 0.09444957
    F[2, 2, 1] = 0.02200000
    F[2, 2, 2] = 0.03970587
    F[2, 2, 3] = 0.03308822
    F[2, 2, 4] = 0.02647058
    F[2, 2, 5] = 0.01985293
    F[2, 2, 6] = 0.01323529
    F[2, 2, 7] = 0.00661764
    F[2, 2, 8] = 0.0
    F[0, 1, 0] = 0.04338699
    F[0, 1, 1] = 0.0099172158
    F[0, 1, 1:9] = 0.0099172158
    F[0, 2, 0] = 0.0493976637
    F[0, 2, 1] = -0.011942669
    F[0, 2, 2:9] = F[0, 1, 1]
    F[0, 3, 0:9] = -0.119798935
    F[0, 3, 0:2] = -0.119798935
    F[0, 3, 2:9] = F[0, 1, 1]
    F[1, 2, 0] = 0.0096495698
    F[1, 2, 1] = 0.030
    F[1, 2, 2] = -0.0200
    F[1, 2, 3] = -0.0233778774
    F[1, 2, 4] = -0.0267557548
    F[1, 2, 5:9] = -0.030133632
    F[1, 3, 1:9] = -0.124836752
    F[2, 3, 0:9] = -0.044709383
    for i in range(3, 8):
        F[2, 2, i] = F[2, 2, 2] + (i - 2) * (F[2, 2, 8] - F[2, 2, 2]) / 6.0
    for i in range(3, 5):
        F[1, 2, i] = F[1, 2, 2] + (i - 2) * (F[1, 2, 5] - F[1, 2, 2]) / 3.0
    dFdi[2, 1, 0] = -0.052500
    dFdi[2, 1, 4:9] = -0.054376
    dFdi[2, 3, 0] = 0.0
    dFdi[2, 3, 1:6] = 0.062418
    dFdk[2, 2, 3:8] = -0.006618
    dFdi[2, 3, 6:9] = 0.062418
    dFdk[1, 1, 1] = -0.060543
    dFdk[1, 2, 3] = -0.020044
    dFdk[1, 2, 4] = -0.020044
    # Symmetrize
    for k in range(0, 10):
        for i in range(0, 4):
            for j in range(i + 1, 4):
                x = F[i, j, k] + F[j, i, k]
                F[i, j, k] = x
                F[j, i, k] = x
                x = dFdi[i, j, k] + dFdj[j, i, k]
                dFdi[i, j, k] = x
                dFdj[j, i, k] = x
                x = dFdi[j, i, k] + dFdj[i, j, k]
                dFdi[j, i, k] = x
                dFdj[i, j, k] = x
                x = dFdk[i, j, k] + dFdk[j, i, k]
                dFdk[i, j, k] = x
                dFdk[j, i, k] = x
    return F, dFdi, dFdj, dFdk


def default_Pcc_table():
    """P_CC(N_H, N_C), shape (6, 6); first index is the hydrogen count."""
    P = np.zeros((6, 6))
    P[1, 1] = 0.003026697473481
    P[2, 0] = 0.007860700254745
    P[3, 0] = 0.016125364564267
    P[1, 2] = 0.003179530830731
    P[2, 1] = 0.006326248241119
    return P


def default_Tcc_table():
    T = np.zeros((5, 5, 10))
    T[2, 2, 0] = -0.070280085
    T[2, 2, 1:9] = -0.00809675
    return T


# --------------------------------------------------------------------------------------
# Spline / table coefficient construction (replicates the Fortran linear systems)
# --------------------------------------------------------------------------------------

def make_cc_g_spline_coefficients():
    """Replicates rebo2_db_make_cc_g_spline. Returns (c1, c2), each of shape (6, 3):
    c[p, j] is the coefficient of cos(theta)**p in interval j (0-based)."""
    th = CC_G_THETA
    c1 = np.zeros((6, 3))
    c2 = np.zeros((6, 3))
    # third interval (knots 3..6, 1-based) : quintic through 4 points + 2 derivative constraints
    A = np.zeros((6, 6))
    for i in range(3, 7):                    # Fortran i = 3..6
        z = th[i - 1]
        for j in range(1, 7):                # Fortran j = 1..6
            A[i - 3, j - 1] = z ** (j - 1)
    z = th[2]
    A[4, :] = 0.0
    A[5, :] = 0.0
    A[4, 1] = 1.0
    A[5, 2] = 2.0
    for j in range(3, 7):
        A[4, j - 1] = (j - 1) * z ** (j - 2)
        if j >= 4:
            A[5, j - 1] = (j - 2) * (j - 1) * z ** (j - 3)
    B = np.zeros(6)
    B[0:4] = CC_G_G1[2:6]
    B[4] = CC_G_DG1[2]
    B[5] = CC_G_D2G1[2]
    c1[:, 2] = np.linalg.solve(A, B)
    B = np.zeros(6)
    B[0:4] = CC_G_G2[2:6]
    B[4] = CC_G_DG1[2]
    B[5] = CC_G_D2G1[2]
    c2[:, 2] = np.linalg.solve(A, B)
    # first and second interval: Hermite quintic with value, 1st and 2nd derivative at both ends
    for k in range(0, 2):
        A = np.zeros((6, 6))
        for i in range(0, 2):
            z = th[k] * (1 - i) + th[k + 1] * i
            A[3 * i + 0, 0] = 1.0
            A[3 * i + 1, 1] = 1.0
            A[3 * i + 2, 2] = 2.0
            for j in range(2, 7):
                A[3 * i + 0, j - 1] = z ** (j - 1)
                if j >= 3:
                    A[3 * i + 1, j - 1] = (j - 1) * z ** (j - 2)
                if j >= 4:
                    A[3 * i + 2, j - 1] = (j - 2) * (j - 1) * z ** (j - 3)
        B = np.array([CC_G_G1[k], CC_G_DG1[k], CC_G_D2G1[k],
                      CC_G_G1[k + 1], CC_G_DG1[k + 1], CC_G_D2G1[k + 1]])
        sol = np.linalg.solve(A, B)
        c1[:, k] = sol
        c2[:, k] = sol
    return c1, c2


def _ipow(base: int, p: int) -> int:
    """Integer power with the Fortran convention 0**0 = 1."""
    return int(base) ** int(p)


def fit_table2d(values, dvdx=None, dvdy=None):
    """Replicates table2d_init. values has shape (nx+1, ny+1). Returns coeff[nx, ny, 4, 4]
    with coeff[bx, by, p1, p2] the coefficient of x1**p1 * x2**p2 inside box (bx, by)."""
    values = np.asarray(values, dtype=float)
    nx, ny = values.shape[0] - 1, values.shape[1] - 1
    ix1 = [0, 1, 1, 0]
    ix2 = [0, 0, 1, 1]
    A = np.zeros((16, 16))
    for icorn in range(4):
        n1, n2 = ix1[icorn], ix2[icorn]
        for p1 in range(4):
            for p2 in range(4):
                p1m = max(p1 - 1, 0)
                p2m = max(p2 - 1, 0)
                icol = 4 * p1 + p2
                A[icorn, icol] = _ipow(n1, p1) * _ipow(n2, p2)
                A[icorn + 4, icol] = p1 * _ipow(n1, p1m) * _ipow(n2, p2)
                A[icorn + 8, icol] = _ipow(n1, p1) * p2 * _ipow(n2, p2m)
                A[icorn + 12, icol] = p1 * _ipow(n1, p1m) * p2 * _ipow(n2, p2m)
    coeff = np.zeros((nx, ny, 4, 4))
    for bx in range(nx):
        for by in range(ny):
            B = np.zeros(16)
            for icorn in range(4):
                n1, n2 = ix1[icorn] + bx, ix2[icorn] + by
                B[icorn] = values[n1, n2]
                if dvdx is not None:
                    B[icorn + 4] = dvdx[n1, n2]
                if dvdy is not None:
                    B[icorn + 8] = dvdy[n1, n2]
            sol = np.linalg.solve(A, B)
            coeff[bx, by] = sol.reshape(4, 4)
    return coeff


def fit_table3d(values, dvdx=None, dvdy=None, dvdz=None):
    """Replicates table3d_init. values has shape (nx+1, ny+1, nz+1).
    Returns coeff[nx, ny, nz, 4, 4, 4]."""
    values = np.asarray(values, dtype=float)
    nx, ny, nz = values.shape[0] - 1, values.shape[1] - 1, values.shape[2] - 1
    ix1 = [0, 1, 1, 0, 0, 1, 1, 0]
    ix2 = [0, 0, 1, 1, 0, 0, 1, 1]
    ix3 = [0, 0, 0, 0, 1, 1, 1, 1]
    A = np.zeros((64, 64))
    for icorn in range(8):
        n1, n2, n3 = ix1[icorn], ix2[icorn], ix3[icorn]
        for p1 in range(4):
            for p2 in range(4):
                for p3 in range(4):
                    p1m, p2m, p3m = max(p1 - 1, 0), max(p2 - 1, 0), max(p3 - 1, 0)
                    icol = 16 * p1 + 4 * p2 + p3
                    A[icorn, icol] = _ipow(n1, p1) * _ipow(n2, p2) * _ipow(n3, p3)
                    A[icorn + 8, icol] = p1 * _ipow(n1, p1m) * _ipow(n2, p2) * _ipow(n3, p3)
                    A[icorn + 16, icol] = _ipow(n1, p1) * p2 * _ipow(n2, p2m) * _ipow(n3, p3)
                    A[icorn + 24, icol] = _ipow(n1, p1) * _ipow(n2, p2) * p3 * _ipow(n3, p3m)
                    A[icorn + 32, icol] = p1 * _ipow(n1, p1m) * p2 * _ipow(n2, p2m) * _ipow(n3, p3)
                    A[icorn + 40, icol] = p1 * _ipow(n1, p1m) * _ipow(n2, p2) * p3 * _ipow(n3, p3m)
                    A[icorn + 48, icol] = _ipow(n1, p1) * p2 * _ipow(n2, p2m) * p3 * _ipow(n3, p3m)
                    A[icorn + 56, icol] = p1 * _ipow(n1, p1m) * p2 * _ipow(n2, p2m) * p3 * _ipow(n3, p3m)
    Ainv_solver = np.linalg.inv(A)  # A is the same for every box; solve exactly like gaussn
    coeff = np.zeros((nx, ny, nz, 4, 4, 4))
    for bx in range(nx):
        for by in range(ny):
            for bz in range(nz):
                B = np.zeros(64)
                for icorn in range(8):
                    n1, n2, n3 = ix1[icorn] + bx, ix2[icorn] + by, ix3[icorn] + bz
                    B[icorn] = values[n1, n2, n3]
                    if dvdx is not None:
                        B[icorn + 8] = dvdx[n1, n2, n3]
                    if dvdy is not None:
                        B[icorn + 16] = dvdy[n1, n2, n3]
                    if dvdz is not None:
                        B[icorn + 24] = dvdz[n1, n2, n3]
                sol = np.linalg.solve(A, B)
                coeff[bx, by, bz] = sol.reshape(4, 4, 4)
    return coeff


@dataclass
class Rebo2ScrParameters:
    """Complete carbon parameter set of REBO2+S as used by Atomistica's Rebo2Scr (defaults)."""
    B1: float = CC_B1
    B2: float = CC_B2
    B3: float = CC_B3
    beta1: float = CC_BETA1
    beta2: float = CC_BETA2
    beta3: float = CC_BETA3
    Q: float = CC_Q
    A: float = CC_A
    alpha: float = CC_ALPHA
    Cmin: float = CMIN
    Cmax: float = CMAX
    screening_threshold: float = SCREENING_THRESHOLD
    dot_threshold: float = DOT_THRESHOLD
    in_r1: float = CC_IN_R1
    in_r2: float = CC_IN_R2
    ar_r1: float = CC_AR_R1
    ar_r2: float = CC_AR_R2
    bo_r1: float = CC_BO_R1
    bo_r2: float = CC_BO_R2
    nc_r1: float = CC_NC_R1
    nc_r2: float = CC_NC_R2
    with_dihedral: bool = False
    g_theta: np.ndarray = field(default_factory=lambda: CC_G_THETA.copy())
    g_coeff1: np.ndarray = None
    g_coeff2: np.ndarray = None
    Fcc: np.ndarray = None
    dFdi: np.ndarray = None
    dFdj: np.ndarray = None
    dFdk: np.ndarray = None
    Pcc: np.ndarray = None
    Tcc: np.ndarray = None
    Fcc_coeff: np.ndarray = None
    Pcc_coeff: np.ndarray = None
    Tcc_coeff: np.ndarray = None

    def __post_init__(self):
        self.g_coeff1, self.g_coeff2 = make_cc_g_spline_coefficients()
        self.Fcc, self.dFdi, self.dFdj, self.dFdk = default_Fcc_table()
        self.Pcc = default_Pcc_table()
        self.Tcc = default_Tcc_table()
        self.Fcc_coeff = fit_table3d(self.Fcc, self.dFdi, self.dFdj, self.dFdk)
        self.Pcc_coeff = fit_table2d(self.Pcc)
        self.Tcc_coeff = fit_table3d(self.Tcc)

    @property
    def C_dr_cut(self):
        return self.Cmax ** 2 / (4.0 * (self.Cmax - 1.0))

    @property
    def dC(self):
        return self.Cmax - self.Cmin

    @property
    def r_cut(self):
        return max(self.in_r2, self.ar_r2, self.bo_r2, self.nc_r2)

    def as_dict(self):
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                d[k] = np.round(v, 14).tolist()
            else:
                d[k] = v
        return d

    def checksum(self) -> str:
        """SHA-256 over the published scalar parameters and raw tables (not the derived coefficients)."""
        keys = ["B1", "B2", "B3", "beta1", "beta2", "beta3", "Q", "A", "alpha", "Cmin", "Cmax",
                "in_r1", "in_r2", "ar_r1", "ar_r2", "bo_r1", "bo_r2", "nc_r1", "nc_r2", "with_dihedral"]
        payload = {k: getattr(self, k) for k in keys}
        payload["g_theta"] = CC_G_THETA.tolist()
        payload["g1"] = CC_G_G1.tolist()
        payload["dg1"] = CC_G_DG1.tolist()
        payload["d2g1"] = CC_G_D2G1.tolist()
        payload["g2"] = CC_G_G2.tolist()
        payload["Fcc"] = self.Fcc.tolist()
        payload["dFdi"] = self.dFdi.tolist()
        payload["dFdj"] = self.dFdj.tolist()
        payload["dFdk"] = self.dFdk.tolist()
        payload["Pcc"] = self.Pcc.tolist()
        payload["Tcc"] = self.Tcc.tolist()
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()


CITATIONS = [
    "D. W. Brenner, O. A. Shenderova, J. A. Harrison, S. J. Stuart, B. Ni, S. B. Sinnott, "
    "J. Phys.: Condens. Matter 14, 783 (2002).",
    "L. Pastewka, P. Pou, R. Perez, P. Gumbsch, M. Moseler, Phys. Rev. B 78, 161402(R) (2008).",
    "L. Pastewka, A. Klemenz, P. Gumbsch, M. Moseler, Phys. Rev. B 87, 205410 (2013).",
    "Atomistica, https://github.com/Atomistica/atomistica (L. Pastewka et al.), version 1.2.7.",
]

if __name__ == "__main__":
    p = Rebo2ScrParameters()
    print("checksum", p.checksum())
    print("g1 coeff\n", p.g_coeff1)
    print("g2 coeff\n", p.g_coeff2)
