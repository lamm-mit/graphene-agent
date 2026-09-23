"""Local FastAPI backend for the carbon-discovery platform.

Endpoints (JSON):
  GET  /api/status                  force engine / device / validation status, campaign counters
  GET  /api/validation              validation table (expected / observed / error / tolerance / PASS-FAIL)
  GET  /api/interpretation          image interpretation document (editable)
  PUT  /api/interpretation          save an edited interpretation document
  GET  /api/families                available generator families and their parameters
  POST /api/generate                generate a structure -> geometry, descriptors, viability pre-check
  POST /api/simulate                launch an AQS simulation of a generated structure (background job)
  GET  /api/jobs                    running / finished jobs
  GET  /api/runs                    experiment database index (all records, slim)
  GET  /api/runs/{run_id}           full record
  GET  /api/runs/{run_id}/trajectory?fields=1   frames (positions, cells, per-atom fields, bonds)
  GET  /api/runs/{run_id}/export/{fmt}          export initial/relaxed/final structure in a given format
  GET  /api/figures                 list of generated figures
  GET  /api/top                     top structures selection + fracture progression metadata
  GET  /api/performance             engine benchmark data
  GET  /api/hypotheses              hypotheses / discriminating experiments / holdout predictions
Static: /  (frontend), /files/... (figures, movies, exports)
"""
from __future__ import annotations
import sys, os, json, glob, time, threading, uuid, io, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from experiments import db
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
from atomistics.io import export_structure, FORMATS
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr, select_device
from potentials.rebo2scr.reference import atomistica_reference as aref
from potentials.rebo2scr.reference.rebo2scr_parameters import Rebo2ScrParameters

app = FastAPI(title="Carbon architecture discovery platform")
FRONT = os.path.join(ROOT, "app", "frontend")
STATE = {"engine": None, "engine_error": None, "jobs": {}}


def get_engine():
    if STATE["engine"] is None and STATE["engine_error"] is None:
        try:
            STATE["engine"] = TorchRebo2Scr(device="auto", dtype=torch.float32)
        except Exception as e:
            STATE["engine_error"] = repr(e)
    return STATE["engine"]


def _json_safe(o):
    return json.loads(json.dumps(o, default=db._json_default))


@app.get("/api/status")
def status():
    eng = get_engine()
    dev = select_device("auto")
    recs = db.all_records()
    val = os.path.join(ROOT, "validation", "results", "validation_table.json")
    vt = json.load(open(val)) if os.path.exists(val) else []
    n_fail = sum(1 for r in vt if r["status"] == "FAIL")
    info = {
        "force_engine": {
            "potential": "screened second-generation REBO (REBO2+S, Rebo2Scr)",
            "implementation": "TorchRebo2Scr (PyTorch autograd port, exact to Atomistica Rebo2Scr)",
            "reference_implementation": aref.REFERENCE_INFO | {"available": aref.available()},
            "parameterization": "Brenner 2002 + Pastewka 2008 screening; Atomistica default tables; dihedral=False",
            "parameter_checksum": Rebo2ScrParameters().checksum(),
            "device": str(dev), "device_fast": str(dev), "device_reference": "cpu",
            "precision_fast": "float32", "precision_reference": "float64",
            "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
            "status": "ready" if eng is not None else "unavailable: " + str(STATE["engine_error"]),
            "units": "eV, Angstrom; 2D stress in N/m (1 eV/A^2 = 16.0218 N/m)",
        },
        "validation": {"n_tests": len(vt), "n_fail": n_fail, "essential_pass": n_fail == 0 and len(vt) >= 16},
        "database": {"n_runs": len(recs), "n_completed": sum(1 for r in recs if r.get("status") == "completed"),
                     "campaigns": sorted(set(str(r.get("campaign")) for r in recs)), "stages": sorted(set(str(r.get("stage")) for r in recs))},
        "environment": db.environment_record(),
        "jobs": {k: {kk: vv for kk, vv in v.items() if kk != "thread"} for k, v in STATE["jobs"].items()},
    }
    return _json_safe(info)


@app.get("/api/validation")
def validation():
    p = os.path.join(ROOT, "validation", "results", "validation_table.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p))


@app.get("/api/interpretation")
def interpretation():
    return json.load(open(os.path.join(ROOT, "analysis", "image_interpretation.json")))


@app.put("/api/interpretation")
def put_interpretation(doc: dict = Body(...)):
    p = os.path.join(ROOT, "analysis", "image_interpretation.json")
    os.replace(p, p + ".bak") if os.path.exists(p) else None
    json.dump(doc, open(p, "w"), indent=1)
    return {"ok": True}


FAMILY_PARAMS = {
    "pristine": {"Lx": 80, "Ly": 80, "orientation": "zigzag"},
    "nanomesh_single": {"Lx": 80, "Ly": 80, "pore_d": 8.0, "period": 16.0, "shape": "circle", "orientation": "zigzag", "stagger": True, "aspect": 1.0, "angle_deg": 0.0, "seed": 0, "jitter": 0.0, "size_disorder": 0.0},
    "nanomesh_hier": {"Lx": 120, "Ly": 120, "levels": 2, "pore_d": 6.0, "period": 12.0, "domain": 40.0, "vein_w": 8.0, "shape": "circle", "orientation": "zigzag", "stagger": True, "seed": 0, "level3_domain": None, "level3_vein": None, "fine_fill": "pores", "jitter": 0.0, "size_disorder": 0.0, "aspect": 1.0, "angle_deg": 0.0, "vein_dirs": "xy"},
    "voronoi_network": {"Lx": 100, "Ly": 100, "n_cells": 30, "ligament_w": 6.0, "seed": 0, "regularity": 0.0, "orientation": "zigzag"},
    "slit_array": {"Lx": 80, "Ly": 80, "slit_len": 16.0, "slit_w": 4.0, "period_x": 30.0, "period_y": 16.0, "angle_deg": 0.0, "stagger": True, "orientation": "zigzag", "seed": 0},
    "strut_lattice": {"Lx": 100, "Ly": 100, "strut_w": 6.0, "spacing": 25.0, "angles_deg": [0.0, 60.0, 120.0], "orientation": "zigzag", "seed": 0},
    "graded_pores": {"Lx": 100, "Ly": 100, "pore_d_min": 4.0, "pore_d_max": 12.0, "period": 16.0, "mode": "x", "orientation": "zigzag", "seed": 0, "shape": "circle"},
    "ring_around_hole": {"Lx": 100, "Ly": 100, "hole_d": 20.0, "n_rings": 2, "ring_pore_d": 5.0, "ring_gap": 8.0, "orientation": "zigzag", "seed": 0, "background": None},
    "precrack": {"Lx": 100, "Ly": 100, "crack_len": 20.0, "crack_w": 3.0, "angle_deg": 90.0, "orientation": "zigzag", "seed": 0},
    "vacancies": {"Lx": 80, "Ly": 80, "fraction": 0.02, "organisation": "random", "cluster_size": 6, "seed": 0, "orientation": "zigzag"},
}


@app.get("/api/families")
def families():
    doc = json.load(open(os.path.join(ROOT, "analysis", "image_interpretation.json")))
    return {"families": FAMILY_PARAMS, "principles": doc.get("families", {})}


class GenerateRequest(BaseModel):
    family: str
    params: dict = {}


def _bonds(atoms):
    from atomistics.descriptors.descriptors import bond_graph
    i, j, S, d = bond_graph(atoms)
    return np.stack([i, j], 1).tolist(), S.tolist()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        params = dict(FAMILY_PARAMS.get(req.family, {}))
        params.update({k: v for k, v in req.params.items() if v is not None})
        atoms = D.generate(req.family, **params)
    except Exception as e:
        raise HTTPException(400, f"generation failed: {e}")
    from scripts.run_batch import precheck
    pre = precheck(atoms)
    desc = compute_descriptors(atoms, atoms.info.get("design"))
    bonds, shifts = _bonds(atoms)
    sid = uuid.uuid4().hex[:8]
    STATE.setdefault("structures", {})[sid] = atoms
    return _json_safe({"structure_id": sid, "n_atoms": len(atoms), "cell": np.asarray(atoms.cell).tolist(), "positions": atoms.positions.round(4).tolist(),
                       "bonds": bonds, "design": atoms.info.get("design"), "porosity": atoms.info.get("porosity"), "precheck": pre, "descriptors": desc})


class SimulateRequest(BaseModel):
    structure_id: str
    name: str = "app_run"
    d_strain: float = 0.005
    max_strain: float = 0.35
    fmax: float = 0.02
    transverse_relax: bool = True
    refine_first_damage: bool = True
    device: str = "auto"
    dtype: str = "float32"
    reason: str = "interactive run from the application"


def _run_job(job_id, atoms, req: SimulateRequest):
    job = STATE["jobs"][job_id]
    try:
        spec = {"campaign": "interactive", "stage": "app", "batch_name": f"app_{job_id}", "device": req.device if req.device != "auto" else str(select_device("auto")),
                "dtype": req.dtype, "aqs": {"d_strain": req.d_strain, "max_strain": req.max_strain, "fmax": req.fmax, "transverse_relax": req.transverse_relax,
                                            "refine_first_damage": req.refine_first_damage, "d_strain_min": req.d_strain / 4},
                "structures": [{"name": req.name, "family": atoms.info["design"]["family"], "params": {k: v for k, v in atoms.info["design"].items() if k in FAMILY_PARAMS.get(atoms.info["design"]["family"], {})},
                                "seed": atoms.info["design"].get("seed", 0), "reason": req.reason}]}
        os.makedirs(os.path.join(ROOT, "experiments", "specs", "interactive"), exist_ok=True)
        sp = os.path.join(ROOT, "experiments", "specs", "interactive", f"{job_id}.json")
        json.dump(spec, open(sp, "w"), indent=1)
        import subprocess
        job["status"] = "running"; job["log"] = os.path.join(ROOT, "experiments", "specs", "interactive", f"{job_id}.log")
        with open(job["log"], "w") as lf:
            p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "run_batch.py"), sp], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            job["pid"] = p.pid
            rc = p.wait()
        job["status"] = "finished" if rc == 0 else f"failed (rc={rc})"
        # find the run id from the log
        try:
            txt = open(job["log"]).read()
            import re
            m = re.findall(r"-> (run_[0-9_a-f]+)", txt)
            job["run_id"] = m[-1] if m else None
        except Exception:
            pass
        job["finished"] = time.time()
    except Exception as e:
        job["status"] = "error: " + repr(e); job["traceback"] = traceback.format_exc()


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    atoms = STATE.get("structures", {}).get(req.structure_id)
    if atoms is None:
        raise HTTPException(404, "structure not found; generate first")
    if get_engine() is None:
        raise HTTPException(503, "force engine unavailable: " + str(STATE["engine_error"]))
    job_id = uuid.uuid4().hex[:8]
    STATE["jobs"][job_id] = {"status": "queued", "started": time.time(), "name": req.name, "n_atoms": len(atoms)}
    t = threading.Thread(target=_run_job, args=(job_id, atoms, req), daemon=True)
    STATE["jobs"][job_id]["thread"] = t
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs():
    out = {}
    for k, v in STATE["jobs"].items():
        d = {kk: vv for kk, vv in v.items() if kk != "thread"}
        if d.get("log") and os.path.exists(d["log"]):
            lines = open(d["log"]).read().splitlines()
            d["log_tail"] = lines[-6:]
        out[k] = d
    return _json_safe(out)


@app.get("/api/runs")
def runs():
    recs = db.all_records()
    slim = []
    for r in recs:
        m = r.get("metrics", {}); d = r.get("design", {}); desc = r.get("descriptors", {})
        slim.append({k: r.get(k) for k in ["run_id", "name", "campaign", "stage", "family", "seed", "n_atoms", "porosity", "hierarchy_levels", "orientation",
                                            "device", "precision", "viability", "status", "termination", "reason", "hypothesis", "prediction", "tags", "wall_time_s", "anomaly_flags"]}
                    | {"metrics": m, "design": d, "descriptors": {k: desc.get(k) for k in ["pore_size_mean_A", "ligament_width_mean_A", "min_solid_fraction_across_x", "anisotropy_index", "tortuosity_x", "cyclomatic_per_atom", "hierarchy_depth_measured", "pore_scale_ratio", "fraction_undercoordinated"]}})
    return _json_safe(slim)


@app.get("/api/runs/{run_id}")
def run_record(run_id: str):
    try:
        return _json_safe(db.load_record(run_id))
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.get("/api/runs/{run_id}/trajectory")
def run_trajectory(run_id: str, fields: int = 1, every: int = 1, max_frames: int = 400):
    rec = db.load_record(run_id)
    tp = rec.get("trajectory")
    if not tp:
        raise HTTPException(404, "no trajectory stored")
    d = np.load(os.path.join(ROOT, tp))
    P = d["positions"]; nfr = len(P)
    step = max(every, int(np.ceil(nfr / max_frames)))
    idx = list(range(0, nfr, step))
    if idx[-1] != nfr - 1:
        idx.append(nfr - 1)
    out = {"frames": idx, "n_atoms": int(P.shape[1]), "eps_x": d["eps_x"].tolist(), "eps_y": d["eps_y"].tolist(),
           "sigma_xx_Nm": (d["sigma_xx"] * 16.0217663).tolist(), "sigma_yy_Nm": (d["sigma_yy"] * 16.0217663).tolist(),
           "energy": d["energy"].tolist(), "n_bonds": d["n_bonds"].tolist(), "n_broken_cum": d["n_broken_cum"].tolist(),
           "cells": [d["cells"][k].tolist() for k in idx], "positions": [P[k].round(3).tolist() for k in idx],
           "bonds": [d[f"bonds_{k}"].tolist() for k in idx]}
    if fields:
        out["energy_per_atom"] = [d["peratom_energy"][k].round(4).tolist() for k in idx]
        out["virial_per_atom"] = [d["peratom_virial"][k].round(4).tolist() for k in idx]
        out["coordination"] = [d["coordination"][k].tolist() for k in idx]
        # displacement relative to the affine map of frame 0
        P0 = P[0]; c0 = d["cells"][0]
        disp = []
        for k in idx:
            c = d["cells"][k]
            aff = P0 * np.array([c[0, 0] / c0[0, 0], c[1, 1] / c0[1, 1], 1.0])
            dd = P[k] - aff
            for ax in range(2):
                dd[:, ax] -= c[ax, ax] * np.round(dd[:, ax] / c[ax, ax])
            disp.append(np.linalg.norm(dd, axis=1).round(3).tolist())
        out["nonaffine_displacement"] = disp
    events = rec.get("broken_bond_events", [])
    out["damage_events"] = events
    return JSONResponse(content=out)


@app.get("/api/runs/{run_id}/export/{which}/{fmt}")
def run_export(run_id: str, which: str, fmt: str):
    from ase.io import read
    rec = db.load_record(run_id)
    rdir = os.path.join(ROOT, rec["run_dir"])
    src = {"initial": "initial.extxyz", "relaxed": "relaxed.extxyz", "final": "final.extxyz"}[which]
    atoms = read(os.path.join(rdir, src))
    outdir = os.path.join(rdir, "exports"); os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{which}")
    if fmt == "trajectory":
        from atomistics.io import trajectory_to_ase
        trajectory_to_ase(os.path.join(ROOT, rec["trajectory"]), os.path.join(outdir, "trajectory"))
        return FileResponse(os.path.join(outdir, "trajectory.extxyz"), filename=f"{run_id}_trajectory.extxyz")
    if fmt == "npz":
        return FileResponse(os.path.join(ROOT, rec["trajectory"]), filename=f"{run_id}.npz")
    if fmt == "csv":
        return FileResponse(os.path.join(rdir, "stress_strain.csv"), filename=f"{run_id}_stress_strain.csv")
    if fmt == "json":
        return FileResponse(os.path.join(rdir, "record.json"), filename=f"{run_id}_record.json")
    if fmt not in FORMATS:
        raise HTTPException(400, "unknown format")
    export_structure(atoms, base, formats=(fmt,), design=rec.get("design"), seed=rec.get("seed"), run_id=run_id)
    return FileResponse(base + FORMATS[fmt], filename=f"{run_id}_{which}{FORMATS[fmt]}")


@app.get("/api/figures")
def figures():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "figures", "**", "*.png"), recursive=True)):
        if os.path.basename(p).startswith("_"):
            continue
        rel = os.path.relpath(p, ROOT)
        cap = p.replace(".png", ".caption.txt")
        out.append({"file": rel, "name": os.path.basename(p), "caption": open(cap).read() if os.path.exists(cap) else "",
                    "svg": os.path.exists(p.replace(".png", ".svg")), "pdf": os.path.exists(p.replace(".png", ".pdf"))})
    movies = [os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(ROOT, "movies", "*.mp4")))]
    return {"figures": out, "movies": movies}


@app.get("/api/top")
def top():
    p = os.path.join(ROOT, "final_designs", "top_structures.json")
    return json.load(open(p)) if os.path.exists(p) else {"selected": [], "note": "top-structure analysis not yet generated"}


@app.get("/api/performance")
def performance():
    p = os.path.join(ROOT, "validation", "results", "performance.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@app.get("/api/hypotheses")
def hypotheses():
    out = {}
    for name in ["hypotheses.json", "stage4_predictions.json", "holdout_predictions.json", "holdout_evaluation.json", "campaign_summary.json"]:
        for folder in ["predictions", "holdouts", "database"]:
            p = os.path.join(ROOT, "experiments", folder, name)
            if os.path.exists(p):
                out[name.replace(".json", "")] = json.load(open(p))
    return out


@app.get("/api/report")
def report():
    p = os.path.join(ROOT, "report", "report.pdf")
    return {"available": os.path.exists(p), "path": "files/report/report.pdf" if os.path.exists(p) else None}


app.mount("/files", StaticFiles(directory=ROOT), name="files")


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(FRONT, "index.html")).read())


app.mount("/static", StaticFiles(directory=os.path.join(FRONT, "static")), name="static")
