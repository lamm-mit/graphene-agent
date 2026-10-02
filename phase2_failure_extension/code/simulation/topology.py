"""Connectivity / damage analysis from atomic geometry.

Bond criterion (documented): two carbon atoms are bonded if r_ij < R_BOND = 2.0 A.  This lies
between the graphene equilibrium bond length (1.42 A) and the inner cut-off of REBO2+S
(fCin decays between 1.95 and 2.25 A); stretched-but-intact bonds at the tensile instability
(~1.7-1.8 A) are still bonds, while ruptured bonds relax to > 2.5 A within one AQS step.
Persistence: a bond is counted as broken only if it stays absent in all later recorded steps
(post-processed), which removes transient reopening/reclosing noise.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

R_BOND = 2.0


def bond_pairs_from_neighbor_data(idx, shift, r, mask, n, r_bond=R_BOND):
    """idx [n,M], shift [n,M,3] (ints), r [n,M], mask [n,M] -> unique bonds (i<j or (i==j, S>0)) as
    arrays i, j, S (shift of j relative to i)."""
    I = np.repeat(np.arange(n)[:, None], idx.shape[1], 1)
    sel = mask & (r < r_bond)
    i, j, S = I[sel], idx[sel], shift[sel]
    keep = (i < j) | ((i == j) & ((S[:, 0] > 0) | ((S[:, 0] == 0) & (S[:, 1] > 0))))
    return i[keep], j[keep], S[keep]


def bond_keys(i, j, S):
    """Integer keys for bonds (unordered pair + image) usable in set operations."""
    return (i.astype(np.int64) * 10_000_000 + j.astype(np.int64)) * 1000 + (S[:, 0] + 5) * 100 + (S[:, 1] + 5) * 10 + (S[:, 2] + 5)


def spanning_x(i, j, S, n):
    """True if the bond network percolates across the periodic x boundary (load-bearing path)."""
    if n == 0 or len(i) == 0:
        return False
    rows, cols = [], []
    sx = S[:, 0]
    m0 = sx == 0
    rows += [i[m0], i[m0] + n]
    cols += [j[m0], j[m0] + n]
    mp = sx > 0
    rows += [i[mp]]
    cols += [j[mp] + n]
    mm = sx < 0
    rows += [i[mm] + n]
    cols += [j[mm]]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, 2 * n))
    ncomp, labels = connected_components(g, directed=False)
    return bool(np.any(labels[:n] == labels[n:]))


def largest_fragment_fraction(i, j, n):
    if n == 0:
        return 0.0
    g = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    counts = np.bincount(labels)
    return float(counts.max() / n), int(ncomp)


def coordination_from_bonds(i, j, n):
    c = np.zeros(n, int)
    np.add.at(c, i, 1)
    np.add.at(c, j, 1)
    return c


def damage_localization(damage_positions, cell, nbins=8):
    """Entropy-based localisation L = 1 - H/Hmax of irreversible damage events binned on an
    nbins x nbins grid over the periodic cell (in the reference frame).  L=0 uniform, L=1 all in one bin.
    Returns (L, H, counts)."""
    if len(damage_positions) == 0:
        return float("nan"), float("nan"), None
    Lx, Ly = cell[0, 0], cell[1, 1]
    x = np.mod(damage_positions[:, 0], Lx) / Lx
    y = np.mod(damage_positions[:, 1], Ly) / Ly
    ix = np.minimum((x * nbins).astype(int), nbins - 1)
    iy = np.minimum((y * nbins).astype(int), nbins - 1)
    counts = np.zeros((nbins, nbins))
    np.add.at(counts, (ix, iy), 1)
    p = counts.ravel() / counts.sum()
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    Hmax = float(np.log(nbins * nbins))
    return 1.0 - H / Hmax, H, counts
