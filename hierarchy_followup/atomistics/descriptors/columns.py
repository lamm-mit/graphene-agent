"""Raster column analysis (phase 3): solid fraction of the weakest column of the occupancy raster, with the seam of the
raster excluded and the solid split into load-parallel vein bands and the rest.

The occupancy raster of atomistics/descriptors/descriptors.py is ceil(Lx/res)*res long, i.e. up to one column longer than
the periodic cell; its last column lies mostly outside the cell and is painted only by the discs of neighbouring atoms, so
it reads emptier than any real column (e.g. 0.72 against 0.85 for the 120 A ellipse mesh).  The phase-1 descriptor
min_solid_fraction_across_x takes the minimum over all columns and therefore carries this artefact whenever Lx is not a
multiple of res (all 120 A cells of phase 1).  Here the minimum is taken over interior columns only."""
from __future__ import annotations
import numpy as np
from .descriptors import _raster


def vein_bands(design):
    """[(y_centre, half_width, spacing)] of the load-parallel veins of a nanomesh_hier design (None otherwise)."""
    fam = design.get("family")
    if fam == "composite":
        w = design["vein_w"] if design.get("coarse") == "veins_x" else design["bar_w"]
    elif fam == "nanomesh_hier" and "x" in str(design.get("vein_dirs", "xy")):
        w = design["vein_w"]
    else:
        return None
    Ly = design["Ly"]; n = max(1, int(round(Ly / design["domain"]))); sy = Ly / n; off = design.get("vein_offset", 0.0)
    return [(((k * sy + off) % Ly), w / 2.0, sy) for k in range(n)]


def in_vein(y, bands):
    if not bands:
        return False
    for yc, hw, sy in bands:
        d = y - yc; d -= sy * round(d / sy)
        if abs(d) < hw:
            return True
    return False


def column_analysis(atoms, x0=None, halfwidth=1.5, seam=1.5, res=0.5, r_atom=1.1):
    """Solid fraction of the weakest raster column.  With x0 (A): the weakest column among those within +-halfwidth of x0
    (the crack column); without x0: the minimum over the interior columns seam < x < Lx - seam (the raster seam excluded).
    The solid rows of that column are split into vein bands and the rest (fine level / domains)."""
    grid, res, L = _raster(atoms, res=res, r_atom=r_atom)
    nx, ny = grid.shape
    xs = (np.arange(nx) + 0.5) * res
    if x0 is None:
        cols = np.where((xs > seam) & (xs < L[0] - seam))[0]
    else:
        lo, hi = int(np.floor((x0 - halfwidth) / res)), int(np.ceil((x0 + halfwidth) / res))
        cols = np.array([i % nx for i in range(lo, hi + 1)])
    widths = grid[cols].sum(axis=1) * res
    j = int(np.argmin(widths)); col = cols[j]
    bands = vein_bands(atoms.info.get("design", {}))
    ys = (np.arange(ny) + 0.5) * res
    solid = grid[col]
    vein = np.array([in_vein(y, bands) for y in ys]) if bands else np.zeros(ny, bool)
    return dict(SF=float(widths[j] / L[1]), SF_vein=float((solid & vein).sum() * res / L[1]), SF_fine=float((solid & ~vein).sum() * res / L[1]), x_col=float(col * res),
                SF_seam_artefact=float(grid.sum(axis=1).min() * res / L[1]))


def min_sf_interior(atoms):
    return column_analysis(atoms)["SF"]
