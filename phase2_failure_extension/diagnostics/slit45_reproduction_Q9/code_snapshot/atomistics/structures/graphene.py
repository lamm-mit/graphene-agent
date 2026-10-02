"""Rectangular (orthorhombic) graphene supercells with chosen lattice orientation.

Loading axis x can be aligned with the zigzag or the armchair direction.

The 4-atom rectangular unit cell of graphene with lattice constant a:
    zigzag along x : cell = (a, sqrt(3) a)
    armchair along x: cell = (sqrt(3) a, a)
REBO2 equilibrium lattice constant a0 = 2.4602 A (bond 1.4204 A) -- determined in validation TEST 5.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

A0_REBO2 = 2.460177  # relaxed REBO2+S lattice constant (validation TEST 5, bond 1.420384 A)


def rectangular_unit_cell(a=A0_REBO2, orientation="zigzag"):
    """Return positions (4,3) and cell (3,3) of the rectangular graphene cell.
    orientation: direction of the lattice along x."""
    acc = a / np.sqrt(3.0)
    if orientation == "zigzag":
        Lx, Ly = a, np.sqrt(3.0) * a
        pos = np.array([[0.0, 0.0, 0.0],
                        [a / 2.0, acc / 2.0, 0.0],
                        [a / 2.0, acc / 2.0 + acc, 0.0],
                        [0.0, 2.0 * acc, 0.0]])
    elif orientation == "armchair":
        Lx, Ly = np.sqrt(3.0) * a, a
        pos = np.array([[0.0, 0.0, 0.0],
                        [acc / 2.0, a / 2.0, 0.0],
                        [acc / 2.0 + acc, a / 2.0, 0.0],
                        [2.0 * acc, 0.0, 0.0]])
    else:
        raise ValueError(orientation)
    cell = np.diag([Lx, Ly, 20.0])
    return pos, cell


def graphene_sheet(nx, ny, a=A0_REBO2, orientation="zigzag", vacuum=10.0):
    """Periodic rectangular graphene sheet with nx x ny rectangular cells."""
    pos, cell = rectangular_unit_cell(a, orientation)
    P = []
    for i in range(nx):
        for j in range(ny):
            P.append(pos + np.array([i * cell[0, 0], j * cell[1, 1], 0.0]))
    P = np.concatenate(P)
    P[:, 2] = vacuum
    c = np.diag([nx * cell[0, 0], ny * cell[1, 1], 2 * vacuum])
    atoms = Atoms("C" * len(P), positions=P, cell=c, pbc=[True, True, False])
    atoms.info["orientation_x"] = orientation
    atoms.info["lattice_constant"] = a
    return atoms


def graphene_sheet_for_size(Lx_target, Ly_target, a=A0_REBO2, orientation="zigzag", vacuum=10.0):
    """Sheet whose periodic lengths are the closest multiples of the unit cell to the targets."""
    pos, cell = rectangular_unit_cell(a, orientation)
    nx = max(1, int(round(Lx_target / cell[0, 0])))
    ny = max(1, int(round(Ly_target / cell[1, 1])))
    return graphene_sheet(nx, ny, a, orientation, vacuum)
