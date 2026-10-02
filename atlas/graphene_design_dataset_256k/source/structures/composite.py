"""Composite hierarchy (phase 3, test T5): a coarse architecture whose solid regions are themselves built from a fine
architecture of a DIFFERENT design (building blocks made of building blocks).

  coarse level   'veins_x'  load-parallel veins of width vein_w every `domain` (domains between them)
                 'square'   square coarse grid: bars of width bar_w every `domain` along x and y (coarse pores between)
  fine level     applied separately to the coarse solid along the load (fill_solid), to the coarse solid across the load
                 (fill_solid_y, square grid only; default = fill_solid) and to the domains / coarse pores (fill_domain):
                   'none'      pristine graphene            'empty'     no material (open coarse pore)
                   'slits_x'   load-parallel slits (fibre bundle: strips of width period_y - slit_w)
                   'round'     round pores                  'ellipse_x' pores elongated along the load (aspect 2)
                   'subveins_ellipse'  a third level: solid sub-veins of width sub_w every sub_domain along the load with
                                       elongated pores between them
  registry       offset (fine lattices) and vein_offset (coarse grid) shift the whole pattern; cracks through generate()
Registered as family 'composite' (design_space imports this module at the end)."""
from __future__ import annotations
import math
import numpy as np
from .graphene import graphene_sheet_for_size, A0_REBO2
from . import design_space as D


def _fine_mask(P, L, kind, period=12.0, pore_d=5.0, slit_w=3.0, slit_len=26.0, period_y=9.0, stagger=True, offset=(0.0, 0.0),
               sub_domain=33.3, sub_w=8.0, vein_offset=0.0):
    """Boolean mask of atoms REMOVED by the fine pattern (before intersection with the region it applies to)."""
    if kind == "none":
        return np.zeros(len(P), bool)
    if kind == "empty":
        return np.ones(len(P), bool)
    if kind == "slits_x":
        nx = max(1, int(round(L[0] / (slit_len + 3.5)))); ny = max(1, int(round(L[1] / period_y)))
        px, py = L[0] / nx, L[1] / ny
        C = np.array([(((ix * px + (0.5 * px if (stagger and iy % 2) else 0.0) + offset[0]) % L[0]), ((iy * py + offset[1]) % L[1])) for iy in range(ny) for ix in range(nx)])
        return D._in_shape(P, C, "slit", slit_len, aspect=slit_w, angle=0.0, L=L)
    C, _ = D._hex_lattice_centers(L, period, offset=tuple(offset), stagger=stagger)
    if kind == "round":
        return D._in_shape(P, C, "circle", pore_d, L=L)
    if kind == "ellipse_x":
        return D._in_shape(P, C, "circle", pore_d, aspect=2.0, angle=math.radians(90.0), L=L)
    if kind == "subveins_ellipse":
        n = max(1, int(round(L[1] / sub_domain))); sy = L[1] / n
        sub = np.abs(D._wrap_delta(P[:, 1] - vein_offset, sy)) < sub_w / 2.0
        return D._in_shape(P, C, "circle", pore_d, aspect=2.0, angle=math.radians(90.0), L=L) & ~sub
    raise ValueError(kind)


def composite(Lx=300.0, Ly=300.0, coarse="veins_x", domain=100.0, vein_w=30.0, bar_w=55.0, fill_solid="slits_x", fill_solid_y=None, fill_domain="round",
              solid_params=None, solid_y_params=None, domain_params=None, orientation="zigzag", seed=0, a=A0_REBO2, vein_offset=0.0, offset=(0.0, 0.0)):
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms); L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]]); P = atoms.positions
    ny = max(1, int(round(L[1] / domain))); sy = L[1] / ny
    w = vein_w if coarse == "veins_x" else bar_w
    solid_x = np.abs(D._wrap_delta(P[:, 1] - vein_offset, sy)) < w / 2.0          # coarse solid running along the load
    if coarse == "veins_x":
        solid_y = np.zeros(len(P), bool)
    elif coarse == "square":
        nx = max(1, int(round(L[0] / domain))); sx = L[0] / nx
        solid_y = (np.abs(D._wrap_delta(P[:, 0] - vein_offset, sx)) < w / 2.0) & ~solid_x   # bars across the load (crossings belong to the x bars)
    else:
        raise ValueError(coarse)
    fill_solid_y = fill_solid_y or fill_solid
    common = dict(offset=tuple(offset), vein_offset=vein_offset)
    mX = _fine_mask(P, L, fill_solid, **{**common, **(solid_params or {})})
    mY = _fine_mask(P, L, fill_solid_y, **{**common, **(solid_y_params if solid_y_params is not None else (solid_params or {}))})
    mD = _fine_mask(P, L, fill_domain, **{**common, **(domain_params or {})})
    remove = (solid_x & mX) | (solid_y & mY) | (~solid_x & ~solid_y & mD)
    atoms.info["n_initial"] = n0
    design = dict(family="composite", hierarchy_levels=3 if fill_domain == "subveins_ellipse" else 2, coarse=coarse, domain=domain, vein_w=vein_w, bar_w=bar_w,
                  fill_solid=fill_solid, fill_solid_y=fill_solid_y, fill_domain=fill_domain, solid_params=solid_params or {}, solid_y_params=solid_y_params,
                  domain_params=domain_params or {}, orientation=orientation, seed=seed, Lx=float(L[0]), Ly=float(L[1]),
                  vein_dirs="x" if coarse == "veins_x" else "xy", vein_offset=vein_offset, offset=list(offset), scale_ratio=domain / float((domain_params or {}).get("period", 12.0)))
    return D._finish(D._remove(atoms, remove, design), design)


D.FAMILIES["composite"] = composite
