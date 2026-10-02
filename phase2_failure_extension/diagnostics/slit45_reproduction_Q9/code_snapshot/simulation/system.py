"""Batched super-system: several independent 2D carbon structures concatenated into one tensor set."""
from __future__ import annotations

import numpy as np
import torch
from ase import Atoms


class BatchedSystem:
    """Holds B independent structures on the engine device.

    positions [N,3], cell [B,3,3] (orthorhombic, a||x, b||y), sid [N] structure id per atom.
    """

    def __init__(self, atoms_list, engine, names=None):
        self.engine = engine
        self.device = engine.device
        self.dtype = engine.dtype
        self.B = len(atoms_list)
        self.names = list(names) if names is not None else [f"s{b}" for b in range(self.B)]
        self.natoms = np.array([len(a) for a in atoms_list], int)
        self.offsets = np.concatenate([[0], np.cumsum(self.natoms)])
        self.N = int(self.offsets[-1])
        pbc0 = list(atoms_list[0].pbc)
        for a in atoms_list:
            assert list(a.pbc) == pbc0, "all structures in a batch must share pbc"
            c = np.asarray(a.cell, float)
            assert abs(c[0, 1]) < 1e-8 and abs(c[1, 0]) < 1e-8 and abs(c[0, 2]) < 1e-8 and abs(c[1, 2]) < 1e-8, \
                "orthorhombic cells required (a || x, b || y)"
        self.pbc = pbc0
        pos = np.concatenate([np.asarray(a.positions, float) for a in atoms_list])
        cells = np.stack([np.asarray(a.cell, float) for a in atoms_list])
        self.positions = torch.as_tensor(pos, dtype=self.dtype, device=self.device)
        self.cell = torch.as_tensor(cells, dtype=self.dtype, device=self.device)
        self.cell0 = self.cell.clone()
        sid = np.repeat(np.arange(self.B), self.natoms)
        self.sid = torch.as_tensor(sid, dtype=torch.long, device=self.device)
        # padded index helpers for per-structure reductions
        local = np.concatenate([np.arange(n) for n in self.natoms])
        self.local_idx = torch.as_tensor(local, dtype=torch.long, device=self.device)
        self.Nmax = int(self.natoms.max())
        self.symbols = [a.get_chemical_symbols() for a in atoms_list]

    # ------------------------------------------------------------------ reductions
    def seg_sum(self, x):
        """x [N] or [N,k] -> [B] or [B,k]"""
        out = torch.zeros((self.B,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        return out.index_add_(0, self.sid, x)

    def seg_max(self, x):
        """x [N] -> [B] (max over atoms of each structure)"""
        pad = torch.full((self.B, self.Nmax), -float("inf"), dtype=x.dtype, device=x.device)
        pad[self.sid, self.local_idx] = x
        return pad.max(dim=1).values

    # ------------------------------------------------------------------ geometry
    def areas(self):
        c = self.cell
        return (c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]).abs()

    def positions_numpy(self, b=None):
        P = self.positions.detach().cpu().numpy()
        if b is None:
            return [P[self.offsets[i]:self.offsets[i + 1]] for i in range(self.B)]
        return P[self.offsets[b]:self.offsets[b + 1]]

    def cells_numpy(self):
        return self.cell.detach().cpu().numpy()

    def to_atoms(self, b, positions=None, cell=None):
        pos = positions if positions is not None else self.positions_numpy(b)
        c = cell if cell is not None else self.cells_numpy()[b]
        return Atoms(symbols=self.symbols[b], positions=np.asarray(pos, float), cell=np.asarray(c, float), pbc=self.pbc)

    def positions_list_numpy(self):
        return self.positions_numpy()

    def build_neighbor_list(self, skin=0.2):
        return self.engine.build_neighbor_list(self.positions_numpy(), self.cells_numpy(), self.pbc, skin=skin)

    def slice(self, b):
        return slice(int(self.offsets[b]), int(self.offsets[b + 1]))
