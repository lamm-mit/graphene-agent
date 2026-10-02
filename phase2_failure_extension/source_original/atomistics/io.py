"""Structure/trajectory export in standard formats via ASE (canonical representation = ase.Atoms).

Every export embeds the design parameters, seed and run id in atoms.info (extxyz keeps them as
key=value pairs in the comment line; other formats carry only geometry, so a JSON sidecar is written)."""
from __future__ import annotations
import json, os
import numpy as np
from ase import Atoms
from ase.io import write, read
from ase.io.trajectory import Trajectory

FORMATS = {"extxyz": ".extxyz", "xyz": ".xyz", "traj": ".traj", "lammps-data": ".lmp", "cif": ".cif", "vasp": ".vasp"}


def atoms_with_metadata(atoms, design=None, seed=None, run_id=None, extra=None):
    a = atoms.copy()
    info = {}
    if design:
        info["design"] = json.dumps(design)
        info["family"] = design.get("family", "")
    if seed is not None:
        info["seed"] = int(seed)
    if run_id is not None:
        info["run_id"] = run_id
    if extra:
        info.update(extra)
    a.info.update(info)
    return a


def export_structure(atoms, path_no_ext, formats=("extxyz", "xyz", "traj", "lammps-data", "cif", "vasp"), design=None, seed=None, run_id=None):
    a = atoms_with_metadata(atoms, design, seed, run_id)
    written = {}
    for f in formats:
        p = path_no_ext + FORMATS[f]
        try:
            if f == "lammps-data":
                write(p, a, format="lammps-data", atom_style="atomic", masses=True)
            elif f == "vasp":
                write(p, a, format="vasp", direct=False, sort=False)
            elif f == "traj":
                write(p, a, format="traj")
            else:
                write(p, a, format=f)
            written[f] = p
        except Exception as e:
            written[f] = f"error: {e}"
    meta = {"design": design, "seed": seed, "run_id": run_id, "cell": np.asarray(a.cell).tolist(), "pbc": list(map(bool, a.pbc)), "n_atoms": len(a)}
    json.dump(meta, open(path_no_ext + ".meta.json", "w"), indent=1)
    return written


def trajectory_to_ase(npz_path, out_path, every=1, fields=True):
    """Convert a stored AQS trajectory (npz) into an ASE .traj + extxyz with per-atom arrays."""
    d = np.load(npz_path)
    P, C = d["positions"], d["cells"]
    n = P.shape[1]
    frames = []
    for k in range(0, len(P), every):
        a = Atoms("C" * n, positions=P[k].astype(float), cell=C[k], pbc=[True, True, False])
        a.info.update(eps_x=float(d["eps_x"][k]), eps_y=float(d["eps_y"][k]), sigma_xx_eV_A2=float(d["sigma_xx"][k]),
                      sigma_yy_eV_A2=float(d["sigma_yy"][k]), energy=float(d["energy"][k]), n_bonds=int(d["n_bonds"][k]))
        if fields:
            a.set_array("energy_per_atom", d["peratom_energy"][k].astype(float))
            a.set_array("virial_xx_yy_xy", d["peratom_virial"][k].astype(float))
            a.set_array("coordination", d["coordination"][k].astype(int))
        frames.append(a)
    write(out_path + ".extxyz", frames, format="extxyz")
    with Trajectory(out_path + ".traj", "w") as t:
        for a in frames:
            t.write(a)
    return len(frames)


def roundtrip_check(atoms, tmpdir):
    """TEST 20: write in every format and read back; report max position/cell deviations."""
    os.makedirs(tmpdir, exist_ok=True)
    res = {}
    base = os.path.join(tmpdir, "rt")
    written = export_structure(atoms, base, design={"family": "test"}, seed=0, run_id="test")
    for f, p in written.items():
        if p.startswith("error"):
            res[f] = {"ok": False, "error": p}
            continue
        try:
            fmt = {"vasp": "vasp", "lammps-data": "lammps-data"}.get(f, None)
            b = read(p, format=fmt) if fmt else read(p)
            if f == "lammps-data":
                b.set_chemical_symbols(["C"] * len(b))
            dp = np.abs(b.positions - atoms.positions)
            # positions may be wrapped/reordered in some formats; compare sorted distances to origin as a robust check
            d1 = np.sort(np.linalg.norm(atoms.positions, axis=1)); d2 = np.sort(np.linalg.norm(b.positions, axis=1))
            res[f] = {"ok": True, "n_atoms": len(b), "max_dpos_direct": float(dp.max()) if len(b) == len(atoms) else None,
                      "max_dpos_sorted_norms": float(np.abs(d1 - d2).max()) if len(b) == len(atoms) else None,
                      "max_dcell": float(np.abs(np.asarray(b.cell) - np.asarray(atoms.cell)).max())}
        except Exception as e:
            res[f] = {"ok": False, "error": repr(e)}
    return res
