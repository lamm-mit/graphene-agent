"""
Dense (padded) neighbour lists for the TorchRebo2Scr engine.

* `build_dense_neighbor_list`  -- cell-list based (ASE `primitive_neighbor_list`, O(N)),
  supports periodic / non-periodic directions and a Verlet skin.
* `brute_force_pairs`          -- O(N^2) reference enumeration with explicit image loops,
  used only to validate the optimised list (validation TEST 9).

Several independent structures can be concatenated into one "super-system" (each atom
carries a structure id); neighbour lists are built per structure and concatenated with
index offsets, so a single force evaluation handles a whole batch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

SHIFT_BASE = 21          # shifts are encoded in [-10, 10]
SHIFT_OFF = 10


@dataclass
class DenseNeighborList:
    idx: torch.Tensor      # [N, M] long   neighbour atom index (padded with self index)
    shift: torch.Tensor    # [N, M, 3]     integer cell shifts (as float dtype of engine)
    mask: torch.Tensor     # [N, M] bool   valid entry
    rev: torch.Tensor      # [N, M] long   flat index (i*M+m) of the reverse pair (j -> i, -S)
    count: torch.Tensor    # [N] long      number of valid neighbours per atom
    cutoff: float          # build cutoff (interaction cutoff + skin)
    skin: float
    ref_positions: torch.Tensor   # positions at build time (for displacement-based rebuild)
    ref_cell: torch.Tensor        # cell at build time [B,3,3]
    n_pairs: int
    max_neighbors: int
    Mb: int = 0            # width of the short (bond-order) list: neighbours within short_cutoff+skin come first
    rev_short: torch.Tensor = None   # [N, Mb] long  flat index (j*Mb+m') of the reverse pair inside the short list
    short_cutoff: float = 0.0

    @property
    def M(self):
        return self.idx.shape[1]

    def average_neighbors(self):
        return float(self.count.float().mean())


def _encode(i, j, S, N):
    """Unique integer key for a directed pair (i, j, S)."""
    code = ((S[:, 0] + SHIFT_OFF) * SHIFT_BASE + (S[:, 1] + SHIFT_OFF)) * SHIFT_BASE + (S[:, 2] + SHIFT_OFF)
    return (i.astype(np.int64) * N + j.astype(np.int64)) * (SHIFT_BASE ** 3) + code.astype(np.int64)


def pairs_for_structure(positions, cell, pbc, cutoff):
    """Cell-list neighbour enumeration (ASE, O(N)). Returns directed pairs i, j, S (int arrays)."""
    n = len(positions)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int)
    i, j, S = primitive_neighbor_list("ijS", pbc=np.asarray(pbc, bool), cell=np.asarray(cell, float),
                                      positions=np.asarray(positions, float), cutoff=float(cutoff),
                                      numbers=None, self_interaction=False, use_scaled_positions=False)
    return i.astype(np.int64), j.astype(np.int64), S.astype(np.int64)


def brute_force_pairs(positions, cell, pbc, cutoff, max_image=3):
    """O(N^2) reference enumeration over explicit periodic images (validation only)."""
    positions = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    n = len(positions)
    ii, jj, SS = [], [], []
    rng = [range(-max_image, max_image + 1) if p else [0] for p in pbc]
    for sx in rng[0]:
        for sy in rng[1]:
            for sz in rng[2]:
                S = np.array([sx, sy, sz])
                disp = S @ cell
                d = positions[None, :, :] - positions[:, None, :] + disp[None, None, :]
                r = np.linalg.norm(d, axis=-1)
                m = r < cutoff
                if sx == 0 and sy == 0 and sz == 0:
                    np.fill_diagonal(m, False)
                a, b = np.nonzero(m)
                ii.append(a)
                jj.append(b)
                SS.append(np.tile(S, (len(a), 1)))
    if not ii:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(SS)


def build_dense_neighbor_list(positions_list, cells, pbc, cutoff, skin=0.5, device="cpu",
                              dtype=torch.float32, pad_multiple=4, pair_fn=None, short_cutoff=None):
    """Build a dense padded neighbour list for a batch of structures.

    positions_list : list of (n_b, 3) arrays (one per structure)
    cells          : (B, 3, 3) array
    pbc            : length-3 bool sequence (shared by the batch)
    cutoff         : interaction cutoff (skin is added on top)
    """
    pair_fn = pair_fn or pairs_for_structure
    rc = float(cutoff) + float(skin)
    rs = (float(short_cutoff) + float(skin)) if short_cutoff is not None else rc
    offsets = np.cumsum([0] + [len(p) for p in positions_list])
    Ntot = int(offsets[-1])
    I, J, S, D = [], [], [], []
    for b, pos in enumerate(positions_list):
        i, j, s = pair_fn(pos, cells[b], pbc, rc)
        with np.errstate(all="ignore"):   # Accelerate emits spurious FP warnings for small mixed-type matmuls; results are exact
            dvec = np.asarray(pos)[j] - np.asarray(pos)[i] + s.astype(float) @ np.asarray(cells[b], float)
        I.append(i + offsets[b])
        J.append(j + offsets[b])
        S.append(s)
        D.append(np.linalg.norm(dvec, axis=1))
    I = np.concatenate(I) if I else np.zeros(0, np.int64)
    J = np.concatenate(J) if J else np.zeros(0, np.int64)
    S = np.concatenate(S) if S else np.zeros((0, 3), np.int64)
    D = np.concatenate(D) if D else np.zeros(0, float)
    n_pairs = len(I)
    # sort by (i, distance) so that per-atom slots are contiguous and short-range neighbours come first
    order = np.lexsort((D, I))
    I, J, S, D = I[order], J[order], S[order], D[order]
    counts = np.bincount(I, minlength=Ntot)
    maxn = int(counts.max()) if Ntot > 0 else 0
    M = max(pad_multiple, int(np.ceil(maxn / pad_multiple) * pad_multiple))
    short_counts = np.bincount(I[D < rs], minlength=Ntot)
    maxs = int(short_counts.max()) if Ntot > 0 else 0
    Mb = max(pad_multiple, int(np.ceil(maxs / pad_multiple) * pad_multiple))
    Mb = min(Mb, M)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(n_pairs) - starts[I]
    idx = np.tile(np.arange(Ntot)[:, None], (1, M))
    shift = np.zeros((Ntot, M, 3), np.int64)
    mask = np.zeros((Ntot, M), bool)
    idx[I, slot] = J
    shift[I, slot] = S
    mask[I, slot] = True
    # reverse-pair index
    keys = _encode(I, J, S, Ntot)
    rkeys = _encode(J, I, -S, Ntot)
    ks = np.argsort(keys)
    pos_in_sorted = np.searchsorted(keys[ks], rkeys)
    if n_pairs > 0:
        assert np.all(keys[ks][pos_in_sorted] == rkeys), "neighbour list is not symmetric"
    rev_pair = ks[pos_in_sorted]                       # index into the pair arrays
    rev_flat = np.full((Ntot, M), -1, np.int64)
    flat_of_pair = I * M + slot
    rev_flat[I, slot] = flat_of_pair[rev_pair]
    self_flat = (np.arange(Ntot)[:, None] * M + np.arange(M)[None, :])
    rev_flat = np.where(mask, rev_flat, self_flat)
    # reverse index restricted to the short list (slots < Mb): flat index j*Mb + m'
    rev_j = rev_flat // M
    rev_m = rev_flat % M
    is_short = np.zeros((Ntot, M), bool)
    is_short[I, slot] = D < rs
    fallback = (np.arange(Ntot)[:, None] * Mb + np.minimum(np.arange(M)[None, :], Mb - 1))
    rev_short = np.where(is_short & (rev_m < Mb), rev_j * Mb + rev_m, fallback)
    rev_short = rev_short[:, :Mb]
    assert np.all((rev_m < Mb) | ~is_short), "short-list reverse pair outside short list (should not happen: same distance)"
    allpos = np.concatenate(positions_list) if Ntot > 0 else np.zeros((0, 3))
    nl = DenseNeighborList(
        idx=torch.as_tensor(idx, device=device),
        shift=torch.as_tensor(shift, dtype=dtype, device=device),
        mask=torch.as_tensor(mask, device=device),
        rev=torch.as_tensor(rev_flat, device=device),
        count=torch.as_tensor(counts, device=device),
        cutoff=rc,
        skin=float(skin),
        ref_positions=torch.as_tensor(allpos, dtype=dtype, device=device),
        ref_cell=torch.as_tensor(np.asarray(cells, float), dtype=dtype, device=device),
        n_pairs=n_pairs,
        max_neighbors=maxn,
        Mb=Mb,
        rev_short=torch.as_tensor(rev_short, device=device),
        short_cutoff=float(short_cutoff) if short_cutoff is not None else rc,
    )
    return nl
