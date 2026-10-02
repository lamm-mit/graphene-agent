"""Image-derived design language -> parametric families of graphene-derived architectures.

All generators start from a periodic rectangular graphene sheet and remove atoms (pores, slits,
cracks, vacancies), producing explicit atomistic candidates (ASE Atoms).  Every generator records
its parameters and random seed in atoms.info["design"].

Families (see report section "Reference-image interpretation"):
  nanomesh_single       single-scale periodic pore lattice (no hierarchy) -- baseline mesh
  nanomesh_hier         nested hierarchy: thick "vein" ligaments partition the sheet into domains
                        (level 1); inside domains a fine pore lattice (level 2); optionally level 3
  voronoi_network       disordered polygonal ligament network (Voronoi cell walls) -- image 3
  slit_array            oriented slit-like pores (anisotropy / load-path alignment) -- image 4
  strut_lattice         straight through-going ligaments crossing at set angles -- image 4
  graded_pores          pore size / spacing gradient along x or radial -- image 5
  ring_around_hole      central hole surrounded by concentric pore rings (flaw shielding) -- image 5
  precrack              central slit crack of given length (fracture-mechanics reference)
  vacancies             random or clustered vacancies (defect organisation)
"""
from __future__ import annotations

import math
import numpy as np
from ase import Atoms

from .graphene import graphene_sheet_for_size, A0_REBO2


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _wrap_delta(d, L):
    return d - L * np.round(d / L)


def _remove(atoms, remove_mask, design):
    keep = ~remove_mask
    new = atoms[keep]
    new.info["design"] = design
    new.info["n_removed"] = int(remove_mask.sum())
    new.info["n_initial"] = int(len(atoms))
    return new


def _in_shape(P, centers, shape, size, aspect=1.0, angle=0.0, L=None):
    """Boolean mask of atoms inside any of the given shapes (periodic)."""
    x, y = P[:, 0], P[:, 1]
    mask = np.zeros(len(P), bool)
    ca, sa = math.cos(angle), math.sin(angle)
    for (cx, cy), s in zip(centers, np.broadcast_to(size, (len(centers),))):
        dx = _wrap_delta(x - cx, L[0])
        dy = _wrap_delta(y - cy, L[1])
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        if shape == "circle":
            # NOTE convention: for aspect != 1 the ellipse's LONG axis is perpendicular to the `angle` direction
            # (angle_deg = 0 -> long axis along y, i.e. perpendicular to the loading axis x; angle_deg = 90 -> long axis along x).
            # For 'slit' the length is along the `angle` direction.  Kept as is for reproducibility of stored designs.
            mask |= (u ** 2 + (v / aspect) ** 2) < (s / 2.0) ** 2
        elif shape == "hexagon":
            # regular hexagon with flat sides, "diameter" s across flats
            r = s / 2.0
            m = np.abs(v) < r
            m &= np.abs(0.5 * np.abs(v) + (math.sqrt(3) / 2) * np.abs(u)) < r * (math.sqrt(3) / 2) * 2 / math.sqrt(3)
            m &= np.abs(u) < r * 2 / math.sqrt(3)
            mask |= m
        elif shape == "square":
            mask |= (np.abs(u) < s / 2.0) & (np.abs(v) < s * aspect / 2.0)
        elif shape == "slit":
            mask |= (np.abs(u) < s / 2.0) & (np.abs(v) < aspect / 2.0)   # aspect = slit width in A
        else:
            raise ValueError(shape)
    return mask


def _prune_dangling(atoms, r_bond=1.9, min_coord=2, iterations=10):
    """Remove atoms with fewer than min_coord neighbours (dangling chains) iteratively, then remove
    isolated fragments (keep the largest connected component)."""
    from ase.neighborlist import primitive_neighbor_list
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    a = atoms.copy()
    removed_total = 0
    for _ in range(iterations):
        if len(a) == 0:
            break
        i, j = primitive_neighbor_list("ij", pbc=a.pbc, cell=np.asarray(a.cell, float), positions=a.positions, cutoff=r_bond)
        coord = np.bincount(i, minlength=len(a))
        bad = coord < min_coord
        if not bad.any():
            break
        a = a[~bad]
        removed_total += int(bad.sum())
    if len(a) > 0:
        i, j = primitive_neighbor_list("ij", pbc=a.pbc, cell=np.asarray(a.cell, float), positions=a.positions, cutoff=r_bond)
        g = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(a), len(a)))
        ncomp, labels = connected_components(g, directed=False)
        if ncomp > 1:
            big = np.argmax(np.bincount(labels))
            keep = labels == big
            removed_total += int((~keep).sum())
            a = a[keep]
    a.info = dict(atoms.info)
    a.info["n_pruned"] = removed_total
    return a


def _finish(atoms, design, prune=True):
    atoms.info["design"] = design
    if prune:
        atoms = _prune_dangling(atoms)
    atoms.info["design"] = design
    atoms.info["porosity"] = 1.0 - len(atoms) / atoms.info.get("n_initial", len(atoms))
    return atoms


def _hex_lattice_centers(L, period, offset=(0.0, 0.0), stagger=True):
    """Centers of a (staggered) lattice of pores with spacing `period` filling the periodic cell."""
    nx = max(1, int(round(L[0] / period)))
    ny = max(1, int(round(L[1] / (period * (math.sqrt(3) / 2 if stagger else 1.0)))))
    px = L[0] / nx
    py = L[1] / ny
    C = []
    for iy in range(ny):
        for ix in range(nx):
            sx = 0.5 * px if (stagger and iy % 2 == 1) else 0.0
            C.append(((ix * px + sx + offset[0]) % L[0], (iy * py + offset[1]) % L[1]))
    return np.array(C), (px, py)


# --------------------------------------------------------------------------------------
# family generators
# --------------------------------------------------------------------------------------

def pristine(Lx=80.0, Ly=80.0, orientation="zigzag", a=A0_REBO2):
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    atoms.info["n_initial"] = len(atoms)
    design = dict(family="pristine", orientation=orientation, Lx=float(atoms.cell[0, 0]), Ly=float(atoms.cell[1, 1]))
    return _finish(atoms, design, prune=False)


def nanomesh_single(Lx=80.0, Ly=80.0, pore_d=8.0, period=16.0, shape="circle", orientation="zigzag", stagger=True,
                    aspect=1.0, angle_deg=0.0, seed=0, jitter=0.0, size_disorder=0.0, a=A0_REBO2, offset=(0.0, 0.0)):
    """Single-scale pore lattice.  Ligament width w = period - pore_d.  jitter/size_disorder in A / fraction."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    rng = np.random.default_rng(seed)
    C, (px, py) = _hex_lattice_centers(L, period, offset=tuple(offset), stagger=stagger)
    if jitter > 0:
        C = C + rng.normal(0, jitter, C.shape)
    sizes = pore_d * (1.0 + size_disorder * rng.normal(0, 1, len(C))) if size_disorder > 0 else np.full(len(C), pore_d)
    sizes = np.clip(sizes, 2.0, None)
    mask = _in_shape(atoms.positions, C, shape, sizes, aspect=aspect, angle=math.radians(angle_deg), L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="nanomesh_single", hierarchy_levels=1, pore_d=pore_d, period=period, shape=shape,
                  ligament_w=period - pore_d, orientation=orientation, stagger=stagger, aspect=aspect,
                  angle_deg=angle_deg, seed=seed, jitter=jitter, size_disorder=size_disorder, offset=list(offset),
                  Lx=float(L[0]), Ly=float(L[1]), n_pores=int(len(C)), actual_period=(float(px), float(py)))
    return _finish(_remove(atoms, mask, design), design)


def nanomesh_hier(Lx=120.0, Ly=120.0, levels=2, pore_d=6.0, period=12.0, domain=40.0, vein_w=8.0,
                  shape="circle", orientation="zigzag", stagger=True, seed=0, level3_domain=None, level3_vein=None,
                  fine_fill="pores", a=A0_REBO2, jitter=0.0, size_disorder=0.0, aspect=1.0, angle_deg=0.0, vein_dirs="xy",
                  vein_offset=0.0, offset=(0.0, 0.0)):
    """Nested hierarchy (images 1-2): thick veins (width vein_w) form a rectangular/hexagonal net of
    domains of size `domain`; inside each domain a fine pore lattice (pore_d, period).  With
    levels=3 an even coarser vein net (level3_domain, level3_vein) is superimposed.
    fine_fill: 'pores' (closed-cell fine net) or 'none' (domains left pristine -> veins only)."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    P = atoms.positions
    rng = np.random.default_rng(seed)
    mask = np.zeros(n0, bool)
    if fine_fill == "pores":
        C, _ = _hex_lattice_centers(L, period, offset=tuple(offset), stagger=stagger)   # offset: lattice-registry replicate (phase 3)
        if jitter > 0:
            C = C + rng.normal(0, jitter, C.shape)
        sizes = pore_d * (1.0 + size_disorder * rng.normal(0, 1, len(C))) if size_disorder > 0 else np.full(len(C), pore_d)
        sizes = np.clip(sizes, 2.0, None)
        mask |= _in_shape(P, C, shape, sizes, aspect=aspect, angle=math.radians(angle_deg), L=L)
    # level-2 veins: keep atoms within vein_w/2 of a rectangular grid of lines with spacing `domain`
    # (vein_dirs selects which line families exist: "xy" = closed rectangular cells, "x" = veins along x only, "y" = along y only)
    def near_grid(P, spacing, width, dirs="xy"):
        nx = max(1, int(round(L[0] / spacing)))
        ny = max(1, int(round(L[1] / spacing)))
        sx, sy = L[0] / nx, L[1] / ny
        dx = np.abs(_wrap_delta(P[:, 0] - vein_offset, sx))
        dy = np.abs(_wrap_delta(P[:, 1] - vein_offset, sy))
        m = np.zeros(len(P), bool)
        if "y" in dirs: m |= dx < width / 2.0     # lines of constant x (veins running along y)
        if "x" in dirs: m |= dy < width / 2.0     # lines of constant y (veins running along x)
        return m
    veins = near_grid(P, domain, vein_w, vein_dirs)
    mask &= ~veins          # veins are solid: no fine pores inside veins
    if levels >= 3:
        d3 = level3_domain or 3 * domain
        w3 = level3_vein or 2 * vein_w
        veins3 = near_grid(P, d3, w3)
        mask &= ~veins3
    atoms.info["n_initial"] = n0
    design = dict(family="nanomesh_hier", hierarchy_levels=levels, pore_d=pore_d, period=period, domain=domain,
                  vein_w=vein_w, shape=shape, orientation=orientation, stagger=stagger, seed=seed,
                  level3_domain=level3_domain, level3_vein=level3_vein, fine_fill=fine_fill,
                  scale_ratio=domain / period, Lx=float(L[0]), Ly=float(L[1]), ligament_w=period - pore_d,
                  jitter=jitter, size_disorder=size_disorder, aspect=aspect, angle_deg=angle_deg, vein_dirs=vein_dirs, vein_offset=vein_offset, offset=list(offset))
    return _finish(_remove(atoms, mask, design), design)


def voronoi_network(Lx=100.0, Ly=100.0, n_cells=30, ligament_w=6.0, seed=0, regularity=0.0, orientation="zigzag",
                    a=A0_REBO2):
    """Disordered polygonal ligament network (image 3): graphene is kept only within ligament_w/2 of
    the edges of a periodic Voronoi tessellation of n_cells random seeds.  regularity in [0,1]
    interpolates seeds from random (0) towards a hexagonal lattice (1)."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, 1, (n_cells, 2)) * L
    if regularity > 0:
        C, _ = _hex_lattice_centers(L, math.sqrt(L[0] * L[1] / n_cells * 2 / math.sqrt(3)))
        C = C[:n_cells] if len(C) >= n_cells else np.concatenate([C, seeds[len(C):]])
        seeds = regularity * C + (1 - regularity) * seeds[:len(C)]
    # periodic Voronoi: distance to nearest and second nearest seed (via images)
    P = atoms.positions[:, :2]
    d = np.full((len(P), 2), np.inf)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            S = seeds + np.array([sx * L[0], sy * L[1]])
            dd = np.sqrt(((P[:, None, :] - S[None, :, :]) ** 2).sum(-1))   # [N, n_cells]
            allD = np.concatenate([d, dd], 1)
            allD.sort(axis=1)
            d = allD[:, :2]
    # atoms near a cell boundary: d2 - d1 small  (perpendicular distance to the bisector ~ (d2-d1)/2 * ...)
    # exact distance to the bisector between the two nearest seeds:
    keep = (d[:, 1] - d[:, 0]) < ligament_w * 1.0   # (d2-d1) ~ 2*distance to boundary for far seeds
    mask = ~keep
    atoms.info["n_initial"] = n0
    design = dict(family="voronoi_network", hierarchy_levels=1, n_cells=n_cells, ligament_w=ligament_w, seed=seed,
                  regularity=regularity, orientation=orientation, Lx=float(L[0]), Ly=float(L[1]),
                  mean_cell_size=float(math.sqrt(L[0] * L[1] / n_cells)))
    return _finish(_remove(atoms, mask, design), design)


def slit_array(Lx=80.0, Ly=80.0, slit_len=16.0, slit_w=4.0, period_x=30.0, period_y=16.0, angle_deg=0.0, stagger=True,
               orientation="zigzag", seed=0, a=A0_REBO2, offset=(0.0, 0.0)):
    """Oriented slit pores (image 4: aligned straight load paths / anisotropy). angle_deg is the
    slit axis relative to the loading (x) axis."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    nx = max(1, int(round(L[0] / period_x)))
    ny = max(1, int(round(L[1] / period_y)))
    px, py = L[0] / nx, L[1] / ny
    C = []
    for iy in range(ny):
        for ix in range(nx):
            sx = 0.5 * px if (stagger and iy % 2 == 1) else 0.0
            C.append(((ix * px + sx + offset[0]) % L[0], (iy * py + offset[1]) % L[1]))   # offset: lattice-registry replicate (phase 3)
    C = np.array(C)
    mask = _in_shape(atoms.positions, C, "slit", slit_len, aspect=slit_w, angle=math.radians(angle_deg), L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="slit_array", hierarchy_levels=1, slit_len=slit_len, slit_w=slit_w, period_x=period_x,
                  period_y=period_y, angle_deg=angle_deg, stagger=stagger, orientation=orientation, seed=seed,
                  Lx=float(L[0]), Ly=float(L[1]), n_slits=int(len(C)), offset=list(offset))
    return _finish(_remove(atoms, mask, design), design)


def strut_lattice(Lx=100.0, Ly=100.0, strut_w=6.0, spacing=25.0, angles_deg=(0.0, 60.0, 120.0), orientation="zigzag",
                  seed=0, a=A0_REBO2):
    """Straight through-going ligaments (image 4): graphene kept within strut_w/2 of periodic families
    of parallel lines at the given angles; everything else removed.  Requires the line families to
    be commensurate with the cell (angles 0/60/120 with hexagonal spacing, or 0/90)."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    P = atoms.positions
    keep = np.zeros(n0, bool)
    for ang in angles_deg:
        th = math.radians(ang)
        nvec = np.array([-math.sin(th), math.cos(th)])       # normal of the line family
        # periodic spacing along the normal must divide the cell: use the nearest commensurate spacing
        proj = P[:, 0] * nvec[0] + P[:, 1] * nvec[1]
        # commensurate period of projection: |n . L| combos -> pick lattice of lines with spacing s
        # (use spacing that divides the projected cell length)
        Lp = abs(nvec[0] * L[0]) if abs(nvec[1]) < 1e-9 else (abs(nvec[1] * L[1]) if abs(nvec[0]) < 1e-9 else None)
        if Lp is None:
            # generic angle: lines y = tan(th) x + c ; periodic if the family maps onto itself: use spacing from Ly
            Lp = abs(nvec[1]) * L[1]
        m = max(1, int(round(Lp / spacing)))
        s = Lp / m
        dist = np.abs(_wrap_delta(proj, s))
        keep |= dist < strut_w / 2.0
    mask = ~keep
    atoms.info["n_initial"] = n0
    design = dict(family="strut_lattice", hierarchy_levels=1, strut_w=strut_w, spacing=spacing, angles_deg=list(angles_deg),
                  orientation=orientation, seed=seed, Lx=float(L[0]), Ly=float(L[1]))
    return _finish(_remove(atoms, mask, design), design)


def graded_pores(Lx=100.0, Ly=100.0, pore_d_min=4.0, pore_d_max=12.0, period=16.0, mode="x", orientation="zigzag",
                 seed=0, shape="circle", a=A0_REBO2):
    """Pore-size gradient (image 5): pore diameter varies linearly with x (mode='x', periodic
    triangle profile) or with distance from the cell centre (mode='radial')."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    C, _ = _hex_lattice_centers(L, period)
    if mode == "x":
        t = np.abs(_wrap_delta(C[:, 0] - L[0] / 2, L[0])) / (L[0] / 2)     # 0 at centre, 1 at boundary
        sizes = pore_d_max + (pore_d_min - pore_d_max) * t
    else:
        rr = np.sqrt(_wrap_delta(C[:, 0] - L[0] / 2, L[0]) ** 2 + _wrap_delta(C[:, 1] - L[1] / 2, L[1]) ** 2)
        t = np.clip(rr / (0.5 * min(L)), 0, 1)
        sizes = pore_d_max + (pore_d_min - pore_d_max) * t
    mask = _in_shape(atoms.positions, C, shape, sizes, L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="graded_pores", hierarchy_levels=1, pore_d_min=pore_d_min, pore_d_max=pore_d_max, period=period,
                  mode=mode, shape=shape, orientation=orientation, seed=seed, Lx=float(L[0]), Ly=float(L[1]),
                  gradient=(pore_d_max - pore_d_min) / (L[0] / 2))
    return _finish(_remove(atoms, mask, design), design)


def ring_around_hole(Lx=100.0, Ly=100.0, hole_d=20.0, n_rings=2, ring_pore_d=5.0, ring_gap=8.0, pores_per_ring=None,
                     orientation="zigzag", seed=0, a=A0_REBO2, background=None):
    """Central hole surrounded by concentric rings of small pores (image 5 'nested rings around a void');
    tests whether a graded/compliant halo around a flaw changes crack initiation.  background: optional
    dict(pore_d, period) to also fill the far field with a uniform pore lattice."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    c0 = L / 2
    centers = [tuple(c0)]
    sizes = [hole_d]
    for k in range(1, n_rings + 1):
        R = hole_d / 2 + k * ring_gap
        npr = pores_per_ring or max(6, int(round(2 * math.pi * R / (ring_pore_d + 3.0))))
        for m in range(npr):
            ang = 2 * math.pi * m / npr + (k % 2) * math.pi / npr
            centers.append((c0[0] + R * math.cos(ang), c0[1] + R * math.sin(ang)))
            sizes.append(ring_pore_d)
    mask = _in_shape(atoms.positions, np.array(centers), "circle", np.array(sizes), L=L)
    if background:
        C, _ = _hex_lattice_centers(L, background["period"])
        far = np.sqrt(_wrap_delta(C[:, 0] - c0[0], L[0]) ** 2 + _wrap_delta(C[:, 1] - c0[1], L[1]) ** 2) > hole_d / 2 + (n_rings + 1) * ring_gap
        mask |= _in_shape(atoms.positions, C[far], "circle", background["pore_d"], L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="ring_around_hole", hierarchy_levels=1 + (n_rings > 0), hole_d=hole_d, n_rings=n_rings,
                  ring_pore_d=ring_pore_d, ring_gap=ring_gap, orientation=orientation, seed=seed, background=background,
                  Lx=float(L[0]), Ly=float(L[1]))
    return _finish(_remove(atoms, mask, design), design)


def precrack(Lx=100.0, Ly=100.0, crack_len=20.0, crack_w=3.0, angle_deg=90.0, orientation="zigzag", seed=0, a=A0_REBO2):
    """Central slit crack; angle_deg=90 -> crack perpendicular to the loading (x) axis."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    mask = _in_shape(atoms.positions, np.array([L / 2]), "slit", crack_len, aspect=crack_w, angle=math.radians(angle_deg), L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="precrack", hierarchy_levels=0, crack_len=crack_len, crack_w=crack_w, angle_deg=angle_deg,
                  orientation=orientation, seed=seed, Lx=float(L[0]), Ly=float(L[1]))
    return _finish(_remove(atoms, mask, design), design)


def vacancies(Lx=80.0, Ly=80.0, fraction=0.02, organisation="random", cluster_size=6, seed=0, orientation="zigzag", a=A0_REBO2):
    """Vacancy defects: 'random' (uniform), 'clustered' (removed in compact clusters of ~cluster_size atoms),
    or 'lines' (short vacancy lines perpendicular to x)."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    rng = np.random.default_rng(seed)
    n_rm = int(round(fraction * n0))
    mask = np.zeros(n0, bool)
    if organisation == "random":
        idx = rng.choice(n0, n_rm, replace=False)
        mask[idx] = True
    elif organisation == "clustered":
        P = atoms.positions
        n_clusters = max(1, n_rm // cluster_size)
        centers = P[rng.choice(n0, n_clusters, replace=False)]
        r_c = math.sqrt(cluster_size / 0.3819 / math.pi)
        mask = _in_shape(P, centers[:, :2], "circle", 2 * r_c, L=L)
    elif organisation == "lines":
        P = atoms.positions
        n_lines = max(1, n_rm // cluster_size)
        centers = P[rng.choice(n0, n_lines, replace=False)]
        mask = _in_shape(P, centers[:, :2], "slit", cluster_size * 1.2, aspect=1.5, angle=math.pi / 2, L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="vacancies", hierarchy_levels=0, fraction=fraction, organisation=organisation,
                  cluster_size=cluster_size, seed=seed, orientation=orientation, Lx=float(L[0]), Ly=float(L[1]))
    return _finish(_remove(atoms, mask, design), design)


def repeat(base_family="nanomesh_single", base_params=None, reps=(2, 2), a=A0_REBO2, seed=None):
    """Exact periodic repetition of a base structure (identical pattern, larger cell) -- for size-convergence tests."""
    base_params = dict(base_params or {})
    if seed is not None and base_family != "pristine":
        base_params.setdefault("seed", seed)
    base = FAMILIES[base_family](**base_params)
    atoms = base.repeat((int(reps[0]), int(reps[1]), 1))
    atoms.info = dict(base.info)
    design = dict(base.info.get("design", {}))
    design.update(family="repeat", base_family=base_family, base_params=base_params, reps=list(reps), Lx=float(atoms.cell[0, 0]), Ly=float(atoms.cell[1, 1]))
    atoms.info["design"] = design
    atoms.info["n_initial"] = base.info.get("n_initial", len(base)) * int(reps[0]) * int(reps[1])
    atoms.info["porosity"] = base.info.get("porosity", 0.0)
    return atoms


FAMILIES = {
    "pristine": pristine,
    "repeat": repeat,
    "nanomesh_single": nanomesh_single,
    "nanomesh_hier": nanomesh_hier,
    "voronoi_network": voronoi_network,
    "slit_array": slit_array,
    "strut_lattice": strut_lattice,
    "graded_pores": graded_pores,
    "ring_around_hole": ring_around_hole,
    "precrack": precrack,
    "vacancies": vacancies,
}


# --------------------------------------------------------------------------------------
# hierarchy follow-up (phase 3) additions.  New keys / a new family only: every phase-1 design is regenerated
# unchanged (checked against the stored atom counts in tests/test_followup.py).
# --------------------------------------------------------------------------------------

_CRACK_KEYS = ("crack_len", "crack_w", "crack_angle_deg", "crack_center")


def add_crack(atoms, crack_len, crack_w=3.0, crack_angle_deg=90.0, crack_center=(0.5, 0.5)):
    """Cut a slit crack of length crack_len (A) and width crack_w into an already generated structure.  The slit is
    centred at the fractional cell position crack_center; crack_angle_deg = 90 puts it perpendicular to the loading
    axis x (same convention and same slit mask as the 'precrack' family).  Dangling atoms at the flanks are pruned by
    _finish, so the realised crack is slightly longer than the nominal one: the realised geometry (extent of all atoms
    removed by the cut and the pruning, projected on the crack axis, plus the tip positions) is stored in
    atoms.info['crack_actual'] and in the design dict.  The porosity is re-referenced to the pristine sheet
    (n_initial of the base design); the porosity of the base design before the crack is kept in design['porosity_base']."""
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    cx, cy = crack_center[0] * L[0], crack_center[1] * L[1]
    ang = math.radians(crack_angle_deg)
    mask = _in_shape(atoms.positions, np.array([[cx, cy]]), "slit", crack_len, aspect=crack_w, angle=ang, L=L)
    design = dict(atoms.info["design"])
    design.update(crack_len=float(crack_len), crack_w=float(crack_w), crack_angle_deg=float(crack_angle_deg),
                  crack_center=[float(crack_center[0]), float(crack_center[1])], cracked=True,
                  porosity_base=float(atoms.info.get("porosity", 0.0)), n_atoms_base=int(len(atoms)))
    n0 = int(atoms.info.get("n_initial", len(atoms)))
    n_removed_before = int(atoms.info.get("n_removed", 0)) + int(atoms.info.get("n_pruned", 0))
    base_pos = atoms.positions.copy()
    new = _finish(_remove(atoms, mask, design), design)
    # realised crack: every base atom that is no longer present (cut + pruned), projected on the crack axis
    keep = {tuple(np.round(q, 3)) for q in new.positions}
    gone = np.array([q for q in base_pos if tuple(np.round(q, 3)) not in keep]) if len(new) < len(base_pos) else np.zeros((0, 3))
    if len(gone):
        du = _wrap_delta(gone[:, 0] - cx, L[0]); dv = _wrap_delta(gone[:, 1] - cy, L[1])
        u = math.cos(ang) * du + math.sin(ang) * dv          # along the crack
        v = -math.sin(ang) * du + math.cos(ang) * dv         # across the crack
        actual = dict(crack_len_actual=float(u.max() - u.min()), crack_w_actual=float(v.max() - v.min()),
                      tip_lo=[float(cx + u.min() * math.cos(ang)), float(cy + u.min() * math.sin(ang))],
                      tip_hi=[float(cx + u.max() * math.cos(ang)), float(cy + u.max() * math.sin(ang))],
                      n_removed_cut=int(mask.sum()), n_pruned=int(new.info.get("n_pruned", 0)))
    else:
        actual = dict(crack_len_actual=0.0, crack_w_actual=0.0, tip_lo=[cx, cy], tip_hi=[cx, cy], n_removed_cut=0, n_pruned=0)
    design.update(crack_actual=actual)
    new.info["design"] = design
    new.info["crack_actual"] = actual
    new.info["n_initial"] = n0
    new.info["n_removed"] = n_removed_before + int(mask.sum()) + int(new.info.get("n_pruned", 0))
    new.info["porosity"] = 1.0 - len(new) / n0
    return new


def seam_pores(Lx=120.0, Ly=120.0, pore_d=4.0, pitch=8.0, seam_spacing=20.0, seam_dir="x", seam_anchor=(0.5, 0.5),
               placement="rows", skip_near=None, orientation="zigzag", seed=0, a=A0_REBO2):
    """Rows ('seams') of small round pores in an otherwise pristine sheet (Cook-Gordon test, T2).
    seam_dir='x': rows run along the load (constant y), spaced seam_spacing along y (rounded to a divisor of Ly) and
    anchored so that one row passes through y = seam_anchor[1]*Ly; along a row the pores sit at
    x = seam_anchor[0]*Lx + pitch/2 + k*pitch, i.e. the line x = seam_anchor[0]*Lx (the crack plane in T2) meets a
    ligament between two pores, not a pore.  seam_dir='y': the same pattern with the roles of x and y exchanged, the
    columns shifted by half a spacing so that none coincides with x = seam_anchor[0]*Lx (across-the-load control).
    placement='random': the same number of pores at random positions with minimum separation `pitch` (control).
    skip_near=(x, y, half_w, half_h) in A: pores whose centres fall inside this rectangle are omitted (keeps the flanks
    of a crack cut afterwards with add_crack free of pores).  Combine with a crack through generate(..., crack_len=...)."""
    atoms = graphene_sheet_for_size(Lx, Ly, a=a, orientation=orientation)
    n0 = len(atoms)
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]])
    ax, ay = seam_anchor[0] * L[0], seam_anchor[1] * L[1]
    rng = np.random.default_rng(seed)
    if seam_dir == "x":
        n_rows = max(1, int(round(L[1] / seam_spacing))); s = L[1] / n_rows
        n_per = max(1, int(round(L[0] / pitch))); p = L[0] / n_per
        C = [((ax + 0.5 * p + k * p) % L[0], (ay + j * s) % L[1]) for j in range(n_rows) for k in range(n_per)]
    elif seam_dir == "y":
        n_rows = max(1, int(round(L[0] / seam_spacing))); s = L[0] / n_rows
        n_per = max(1, int(round(L[1] / pitch))); p = L[1] / n_per
        C = [((ax + 0.5 * s + j * s) % L[0], (ay + k * p) % L[1]) for j in range(n_rows) for k in range(n_per)]
    else:
        raise ValueError(seam_dir)
    C = np.array(C)
    if skip_near is not None:
        x0, y0, hw, hh = skip_near
        inside = (np.abs(_wrap_delta(C[:, 0] - x0, L[0])) < hw) & (np.abs(_wrap_delta(C[:, 1] - y0, L[1])) < hh)
        C = C[~inside]
    if placement == "random":
        n_target = len(C); pts = []; tries = 0
        while len(pts) < n_target and tries < 200000:
            tries += 1
            q = rng.uniform(0, 1, 2) * L
            if skip_near is not None and abs(_wrap_delta(q[0] - skip_near[0], L[0])) < skip_near[2] and abs(_wrap_delta(q[1] - skip_near[1], L[1])) < skip_near[3]:
                continue
            if pts:
                P = np.array(pts); dd = np.hypot(_wrap_delta(P[:, 0] - q[0], L[0]), _wrap_delta(P[:, 1] - q[1], L[1]))
                if dd.min() < pitch:
                    continue
            pts.append(q)
        C = np.array(pts)
    elif placement != "rows":
        raise ValueError(placement)
    mask = _in_shape(atoms.positions, C, "circle", pore_d, L=L)
    atoms.info["n_initial"] = n0
    design = dict(family="seam_pores", hierarchy_levels=1, pore_d=pore_d, pitch=pitch, seam_spacing=float(s), seam_spacing_nominal=seam_spacing,
                  seam_dir=seam_dir, seam_anchor=list(seam_anchor), placement=placement, skip_near=list(skip_near) if skip_near is not None else None,
                  n_pores=int(len(C)), n_rows=int(n_rows), orientation=orientation, seed=seed, Lx=float(L[0]), Ly=float(L[1]))
    return _finish(_remove(atoms, mask, design), design)


FAMILIES["seam_pores"] = seam_pores


def generate(family, **params):
    """Generate a structure; crack keys (crack_len, crack_w, crack_angle_deg, crack_center) are applied to the generated
    structure afterwards with add_crack (phase 3).  The 'precrack' family keeps its own crack parameters."""
    if family != "precrack" and any(k in params for k in _CRACK_KEYS):
        crack = {k: params.pop(k) for k in _CRACK_KEYS if k in params}
        atoms = FAMILIES[family](**params)
        return add_crack(atoms, **crack) if crack.get("crack_len") else atoms
    return FAMILIES[family](**params)



from . import composite  # noqa: E402,F401  (phase 3, T5: registers the family "composite")
