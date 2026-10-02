"""Machine-readable experiment database: one directory per run, JSON records + NPZ trajectories,
plus a flat index (JSONL) that is regenerated from the run directories."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "experiments", "database")
RUNS_DIR = os.path.join(DB_DIR, "runs")
TRAJ_DIR = os.path.join(ROOT, "trajectories")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o):
        return asdict(o)
    return str(o)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def new_run_id(prefix="run"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def environment_record():
    import torch, ase
    try:
        import atomistica
        atv = getattr(atomistica, "__version__", None)
    except Exception:
        atv = None
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        cpu = platform.processor()
    gpu = None
    try:
        sp = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode()
        for line in sp.splitlines():
            if "Chipset Model" in line:
                gpu = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return {
        "os": platform.platform(),
        "cpu": cpu,
        "gpu": gpu,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ase": ase.__version__,
        "atomistica": atv,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def code_version():
    """Best-effort git commit of the project; falls back to a content hash of the source tree."""
    try:
        c = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return {"git_commit": c}
    except Exception:
        h = hashlib.sha256()
        for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
            if "/app/frontend/" in p:
                continue
            h.update(open(p, "rb").read())
        return {"git_commit": None, "source_sha256": h.hexdigest()}


def save_run(run_id, record, result, atoms_initial, atoms_relaxed, atoms_final=None, save_trajectory=True):
    """Persist one AQS simulation: JSON record, metrics, stress-strain CSV, structures (extxyz), trajectory (npz)."""
    from ase.io import write
    rdir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(rdir, exist_ok=True)
    # stress-strain table
    cols = ["eps_x", "eps_y", "sigma_xx", "sigma_yy", "sigma_xy", "energy", "n_bonds", "n_broken_new", "n_formed_new",
            "n_broken_cum", "fire_iterations", "fire_converged", "transverse_iterations", "spanning", "largest_fragment",
            "n_fragments", "wall_time"]
    arr = np.stack([np.asarray(result[c], float) for c in cols], 1)
    np.savetxt(os.path.join(rdir, "stress_strain.csv"), arr, delimiter=",", header=",".join(cols) + "  # sigma in eV/A^2 (x16.0218 = N/m)", comments="")
    # structures
    write(os.path.join(rdir, "initial.extxyz"), atoms_initial)
    write(os.path.join(rdir, "relaxed.extxyz"), atoms_relaxed)
    if atoms_final is not None:
        write(os.path.join(rdir, "final.extxyz"), atoms_final)
    # trajectory (compact npz) -- positions float32 per recorded frame + per-atom fields
    traj_path = None
    existing = os.path.join(TRAJ_DIR, f"{run_id}.npz")
    if not save_trajectory and os.path.exists(existing):
        traj_path = existing
    if save_trajectory and len(result["positions"]) > 0:
        os.makedirs(TRAJ_DIR, exist_ok=True)
        traj_path = os.path.join(TRAJ_DIR, f"{run_id}.npz")
        nb = {f"bonds_{k}": b for k, b in enumerate(result["bonds"])}
        np.savez_compressed(traj_path, positions=np.stack(result["positions"]), cells=np.stack(result["cells"]),
                            peratom_energy=np.stack(result["peratom_energy"]), peratom_virial=np.stack(result["peratom_virial"]),
                            coordination=np.stack(result["coordination"]), eps_x=np.asarray(result["eps_x"]),
                            eps_y=np.asarray(result["eps_y"]), sigma_xx=np.asarray(result["sigma_xx"]),
                            sigma_yy=np.asarray(result["sigma_yy"]), sigma_xy=np.asarray(result["sigma_xy"]),
                            energy=np.asarray(result["energy"]), n_bonds=np.asarray(result["n_bonds"]),
                            n_broken_cum=np.asarray(result["n_broken_cum"]), **nb)
    record = dict(record)
    record["run_id"] = run_id
    record["trajectory"] = os.path.relpath(traj_path, ROOT) if traj_path else None
    record["run_dir"] = os.path.relpath(rdir, ROOT)
    dump_json(record, os.path.join(rdir, "record.json"))
    return rdir


def load_record(run_id):
    return json.load(open(os.path.join(RUNS_DIR, run_id, "record.json")))


def all_records():
    recs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "record.json"))):
        try:
            recs.append(json.load(open(p)))
        except Exception as e:
            print("bad record", p, e)
    return recs


def rebuild_index():
    recs = all_records()
    os.makedirs(DB_DIR, exist_ok=True)
    with open(os.path.join(DB_DIR, "index.jsonl"), "w") as f:
        for r in recs:
            slim = {k: v for k, v in r.items() if k not in ("broken_bond_events",)}
            f.write(json.dumps(slim, default=_json_default) + "\n")
    # CSV summary of key metrics
    import csv
    keys = ["run_id", "campaign", "stage", "family", "name", "seed", "n_atoms", "porosity", "hierarchy_levels", "orientation",
            "device", "precision", "viability", "modulus_2d_Nm", "strength_Nm", "strain_at_peak", "first_damage_strain",
            "failure_strain", "work_to_failure_J_m2", "specific_work_eV_per_atom", "damage_localization", "fracture_mode",
            "n_broken_total", "termination", "wall_time_s"]
    with open(os.path.join(DB_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in recs:
            m = r.get("metrics", {})
            d = r.get("design", {})
            row = []
            for k in keys:
                if k in r:
                    row.append(r[k])
                elif k in m:
                    row.append(m[k])
                elif k in d:
                    row.append(d[k])
                else:
                    row.append("")
            w.writerow(row)
    return len(recs)
