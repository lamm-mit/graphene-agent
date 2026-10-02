"""Structural descriptors of 2D carbon architectures (computed from the atomic structure, not from
the generator parameters), intended to explain mechanisms:

geometry     porosity, areal density, footprint
pores        pore-size distribution (distance transform of the empty phase on a raster), number of
             distinct pore scales, scale ratio (largest/smallest characteristic pore size)
ligaments    ligament-width distribution (distance transform of the solid phase), minimum width
             (bottleneck) along the loading direction
topology     coordination distribution, bond-graph cyclomatic number per atom (loops/redundancy),
             spanning check, load-path tortuosity (shortest periodic path across x vs Lx)
anisotropy   orientation tensor of ligaments (solid-phase structure tensor) and its anisotropy index
hierarchy    hierarchy depth estimate from the multi-scale pore/ligament distributions and
             localisation of the coarse scale (fraction of solid belonging to the widest ligaments)
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from ase.neighborlist import primitive_neighbor_list

R_BOND = 1.9
RHO_GRAPHENE = 0.38177   # atoms / A^2 at a0 = 2.4602 A


def _raster(atoms, res=0.5, r_atom=1.0):
    """Occupancy raster of the solid phase (periodic in x, y)."""
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    nx, ny = int(np.ceil(L[0] / res)), int(np.ceil(L[1] / res))
    grid = np.zeros((nx, ny), bool)
    P = atoms.positions[:, :2]
    rr = int(np.ceil(r_atom / res))
    ix = np.floor(P[:, 0] / res).astype(int) % nx
    iy = np.floor(P[:, 1] / res).astype(int) % ny
    for dx in range(-rr, rr + 1):
        for dy in range(-rr, rr + 1):
            if dx * dx + dy * dy <= rr * rr:
                grid[(ix + dx) % nx, (iy + dy) % ny] = True
    return grid, res, L


def _periodic_edt(mask, res):
    """Euclidean distance transform of `mask` (True = phase of interest) with periodic wrap."""
    t = np.tile(mask, (3, 3))
    d = ndi.distance_transform_edt(t) * res
    nx, ny = mask.shape
    return d[nx:2 * nx, ny:2 * ny]


def _local_maxima_sizes(dist, mask, res, min_sep):
    """Characteristic sizes: local maxima of the distance transform (twice the inscribed radius)."""
    if not mask.any():
        return np.zeros(0)
    fp = int(max(1, min_sep / res))
    mx = ndi.maximum_filter(np.tile(dist, (3, 3)), size=2 * fp + 1)
    nx, ny = dist.shape
    mx = mx[nx:2 * nx, ny:2 * ny]
    peaks = (dist == mx) & mask & (dist > res)
    return 2.0 * dist[peaks]


def bond_graph(atoms, r_bond=R_BOND):
    i, j, S, D = primitive_neighbor_list("ijSD", pbc=atoms.pbc, cell=np.asarray(atoms.cell, float), positions=atoms.positions, cutoff=r_bond)
    keep = (i < j) | ((i == j) & ((S[:, 0] > 0) | ((S[:, 0] == 0) & (S[:, 1] > 0))))
    return i[keep], j[keep], S[keep], np.linalg.norm(D[keep], axis=1)


def tortuosity_x(atoms, i, j, S, d):
    """Shortest bond-path length needed to cross the periodic cell once along x, divided by Lx.
    Computed on a 2x1 supercell graph: distance from each atom to its own +x image (min over atoms)."""
    n = len(atoms)
    Lx = atoms.cell[0, 0]
    rows, cols, w = [], [], []
    sx = S[:, 0]
    for a, b, s, dd in zip(i, j, sx, d):
        if s == 0:
            rows += [a, a + n]; cols += [b, b + n]; w += [dd, dd]
        elif s > 0:
            rows += [a]; cols += [b + n]; w += [dd]
        else:
            rows += [a + n]; cols += [b]; w += [dd]
    g = sp.coo_matrix((w, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    g = g + g.T
    # sample a subset of source atoms for speed
    rng = np.random.default_rng(0)
    src = rng.choice(n, min(n, 40), replace=False)
    dist = dijkstra(g, directed=False, indices=src)
    dd = np.array([dist[k, s + n] for k, s in enumerate(src)])
    dd = dd[np.isfinite(dd)]
    return float(dd.min() / Lx) if len(dd) else float("inf")


def compute_descriptors(atoms, design=None):
    n = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    A = float(L[0] * L[1])
    desc = {"n_atoms": n, "footprint_A2": A, "Lx": float(L[0]), "Ly": float(L[1]),
            "areal_density": n / A, "porosity": 1.0 - (n / A) / RHO_GRAPHENE,
            "carbon_mass_amu": 12.011 * n}
    # topology
    i, j, S, d = bond_graph(atoms)
    coord = np.zeros(n, int); np.add.at(coord, i, 1); np.add.at(coord, j, 1)
    desc["coordination_mean"] = float(coord.mean())
    desc["coordination_hist"] = {int(k): int((coord == k).sum()) for k in np.unique(coord)}
    desc["fraction_undercoordinated"] = float((coord < 3).sum() / n)
    desc["edge_atoms"] = int((coord < 3).sum())
    g = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    desc["n_fragments"] = int(ncomp)
    desc["cyclomatic_per_atom"] = float((len(i) - n + ncomp) / n)    # independent loops per atom (redundancy)
    desc["bonds_per_atom"] = float(len(i) / n)
    desc["tortuosity_x"] = tortuosity_x(atoms, i, j, S, d) if n > 0 else float("nan")
    # rasters
    grid, res, _ = _raster(atoms, res=0.5, r_atom=1.1)
    solid = grid
    pore = ~grid
    if pore.any():
        dpore = _periodic_edt(pore, res)
        psz = _local_maxima_sizes(dpore, pore, res, min_sep=3.0)
        desc["pore_size_max_A"] = float(2 * dpore.max())
        desc["pore_size_mean_A"] = float(psz.mean()) if len(psz) else 0.0
        desc["pore_size_std_A"] = float(psz.std()) if len(psz) else 0.0
        desc["n_pores_detected"] = int(len(psz))
        # scale analysis: cluster the characteristic sizes in log space
        if len(psz) > 1:
            ls = np.log(psz[psz > 2.5])
            if len(ls) > 1:
                hist, edges = np.histogram(ls, bins=np.arange(ls.min() - 0.01, ls.max() + 0.35, 0.35))
                peaks = [(edges[k] + edges[k + 1]) / 2 for k in range(len(hist)) if hist[k] > 0 and (k == 0 or hist[k] >= hist[k - 1]) and (k == len(hist) - 1 or hist[k] >= hist[k + 1])]
                # merge peaks closer than a factor 1.6
                merged = []
                for p in peaks:
                    if merged and abs(p - merged[-1]) < np.log(1.6):
                        continue
                    merged.append(p)
                desc["n_pore_scales"] = int(len(merged))
                desc["pore_scale_ratio"] = float(np.exp(max(merged) - min(merged))) if len(merged) > 1 else 1.0
            else:
                desc["n_pore_scales"] = 1; desc["pore_scale_ratio"] = 1.0
        else:
            desc["n_pore_scales"] = int(len(psz) > 0); desc["pore_scale_ratio"] = 1.0
    else:
        desc.update(pore_size_max_A=0.0, pore_size_mean_A=0.0, pore_size_std_A=0.0, n_pores_detected=0, n_pore_scales=0, pore_scale_ratio=1.0)
    if solid.any():
        dsol = _periodic_edt(solid, res)
        lw = _local_maxima_sizes(dsol, solid, res, min_sep=2.0)
        desc["ligament_width_max_A"] = float(2 * dsol.max())
        desc["ligament_width_mean_A"] = float(lw.mean()) if len(lw) else float(2 * dsol.max())
        desc["ligament_width_std_A"] = float(lw.std()) if len(lw) else 0.0
        # bottleneck: for each x-column of the raster, total solid width (min over x) -> minimal load-bearing width
        widths = solid.sum(axis=1) * res
        desc["min_solid_width_across_x_A"] = float(widths.min())
        desc["min_solid_fraction_across_x"] = float(widths.min() / L[1])
        # anisotropy: structure tensor of the solid phase (gradient-based) -> orientation of ligaments
        gx = np.gradient(dsol, axis=0); gy = np.gradient(dsol, axis=1)
        w = solid & (dsol > 0)
        J = np.array([[np.sum(gx[w] ** 2), np.sum(gx[w] * gy[w])], [np.sum(gx[w] * gy[w]), np.sum(gy[w] ** 2)]])
        ev, evec = np.linalg.eigh(J)
        desc["anisotropy_index"] = float((ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-12))
        vmin = evec[:, 0]   # direction of least gradient = ligament direction
        desc["ligament_orientation_deg"] = float(np.degrees(np.arctan2(vmin[1], vmin[0])) % 180)
        # fraction of solid in "coarse" ligaments (width > 2x the mean width) -> hierarchy localisation
        wm = desc["ligament_width_mean_A"]
        coarse = solid & (dsol > wm)   # half-width > mean width -> width > 2*mean
        desc["coarse_ligament_solid_fraction"] = float(coarse.sum() / solid.sum())
        lws = lw[lw > 2.0]
        if len(lws) > 1:
            ls = np.log(lws)
            hist, edges = np.histogram(ls, bins=np.arange(ls.min() - 0.01, ls.max() + 0.35, 0.35))
            peaks = [(edges[k] + edges[k + 1]) / 2 for k in range(len(hist)) if hist[k] > 0 and (k == 0 or hist[k] >= hist[k - 1]) and (k == len(hist) - 1 or hist[k] >= hist[k + 1])]
            merged = []
            for p in peaks:
                if merged and abs(p - merged[-1]) < np.log(1.6):
                    continue
                merged.append(p)
            desc["n_ligament_scales"] = int(len(merged))
            desc["ligament_scale_ratio"] = float(np.exp(max(merged) - min(merged))) if len(merged) > 1 else 1.0
        else:
            desc["n_ligament_scales"] = 1; desc["ligament_scale_ratio"] = 1.0
    desc["hierarchy_depth_measured"] = int(max(desc.get("n_pore_scales", 0), desc.get("n_ligament_scales", 1)))
    if design:
        desc["hierarchy_levels_design"] = design.get("hierarchy_levels", None)
        desc["family"] = design.get("family")
    return desc
