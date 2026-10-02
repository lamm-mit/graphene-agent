"""ASE Calculator wrapper around TorchRebo2Scr (single structure)."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .torch_rebo2scr import TorchRebo2Scr

EV_PER_A2_TO_N_PER_M = 16.0217663  # 1 eV/A^2 = 16.0218 N/m


class TorchRebo2ScrCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress", "stresses", "free_energy"]

    def __init__(self, device="auto", dtype=torch.float32, skin=0.5, engine=None, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine or TorchRebo2Scr(device=device, dtype=dtype)
        self.skin = skin
        self.nl = None
        self._nl_natoms = -1

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        eng = self.engine
        cell = np.asarray(atoms.cell, float)
        R = torch.as_tensor(atoms.positions, dtype=eng.dtype, device=eng.device)
        cell_t = torch.as_tensor(cell[None], dtype=eng.dtype, device=eng.device)
        sid = torch.zeros(len(atoms), dtype=torch.long, device=eng.device)
        rebuild = (self.nl is None or self._nl_natoms != len(atoms) or "pbc" in system_changes
                   or eng.needs_rebuild(R, cell_t, sid, self.nl))
        if rebuild:
            self.nl = eng.build_neighbor_list([atoms.positions], cell[None], list(atoms.pbc), skin=self.skin)
            self._nl_natoms = len(atoms)
        per_atom = ("stresses" in properties) or ("energies" in properties)
        out = eng.evaluate(R, cell_t, sid, self.nl, compute_stress=True, per_atom=per_atom)
        vol = abs(np.linalg.det(cell))
        self.results["energy"] = float(out["energy_total"].cpu())
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = out["forces"].detach().cpu().numpy().astype(float)
        dEde = out["dE_deps"][0].cpu().numpy().astype(float)
        sig = 0.5 * (dEde + dEde.T) / vol
        self.results["stress"] = np.array([sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]])
        self.results["energies"] = out["energy_per_atom"].detach().cpu().numpy().astype(float)
        if per_atom:
            w = out["virial_per_atom"].cpu().numpy().astype(float)
            w = 0.5 * (w + np.transpose(w, (0, 2, 1))) / vol * len(atoms)  # per-atom stress in "per atomic volume" convention
            self.results["stresses"] = np.stack([w[:, 0, 0], w[:, 1, 1], w[:, 2, 2], w[:, 1, 2], w[:, 0, 2], w[:, 0, 1]], 1)
