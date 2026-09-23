"""Helpers for the later campaign stages: hypothesis records, prediction files written BEFORE the
simulations (timestamped + hashed), holdout evaluation (prediction vs observation), spec writers."""
from __future__ import annotations
import sys, os, json, time, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from experiments import db

PRED_DIR = os.path.join(ROOT, "experiments", "predictions")
HOLD_DIR = os.path.join(ROOT, "experiments", "holdouts")
os.makedirs(PRED_DIR, exist_ok=True); os.makedirs(HOLD_DIR, exist_ok=True)


def write_predictions(name, predictions, notes=""):
    """predictions: list of dicts with keys name, family, params, predicted: {modulus_2d_Nm, strength_Nm, failure_strain,
    work_to_failure_J_m2, damage_localization, fracture_mode, crack_path_or_mechanism}, rationale.
    The file is timestamped and hashed so that it can be shown to predate the simulations."""
    doc = {"name": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes, "predictions": predictions}
    s = json.dumps(doc, sort_keys=True, indent=1, default=db._json_default)
    doc["sha256_of_content"] = hashlib.sha256(s.encode()).hexdigest()
    path = os.path.join(PRED_DIR, f"{name}.json")
    json.dump(doc, open(path, "w"), indent=1, default=db._json_default)
    return path, doc["sha256_of_content"]


def evaluate_predictions(pred_file, stage):
    """Compare the predictions of pred_file with the completed runs of the given stage (matched by structure name)."""
    doc = json.load(open(pred_file))
    recs = {r["name"]: r for r in db.all_records() if r.get("stage") == stage and r.get("status") == "completed"}
    rows = []
    for p in doc["predictions"]:
        r = recs.get(p["name"])
        row = {"name": p["name"], "family": p.get("family"), "predicted": p["predicted"], "rationale": p.get("rationale", ""), "run_id": r["run_id"] if r else None}
        if r:
            m = r["metrics"]
            obs = {k: m.get(k) for k in ["modulus_2d_Nm", "strength_Nm", "failure_strain", "work_to_failure_J_m2", "damage_localization", "fracture_mode", "first_damage_strain"]}
            obs["viability"] = r.get("viability")
            row["observed"] = obs
            errs = {}
            for k in ["modulus_2d_Nm", "strength_Nm", "failure_strain", "work_to_failure_J_m2", "damage_localization"]:
                pv, ov = p["predicted"].get(k), obs.get(k)
                if pv is not None and ov is not None and np.isfinite(ov) and np.isfinite(pv):
                    errs[k] = {"abs": float(ov - pv), "rel": float((ov - pv) / ov) if ov != 0 else None}
            row["errors"] = errs
            row["mode_match"] = (p["predicted"].get("fracture_mode_class") == classify_mode_class(obs.get("fracture_mode")))
        rows.append(row)
    summary = {}
    for k in ["modulus_2d_Nm", "strength_Nm", "failure_strain", "work_to_failure_J_m2", "damage_localization"]:
        rel = [abs(x["errors"][k]["rel"]) for x in rows if "errors" in x and k in x["errors"] and x["errors"][k]["rel"] is not None]
        ab = [abs(x["errors"][k]["abs"]) for x in rows if "errors" in x and k in x["errors"]]
        if rel:
            summary[k] = {"median_abs_rel_error": float(np.median(rel)), "mean_abs_rel_error": float(np.mean(rel)), "max_abs_rel_error": float(np.max(rel)), "mean_abs_error": float(np.mean(ab)), "n": len(rel)}
    modes = [x["mode_match"] for x in rows if "mode_match" in x]
    summary["fracture_mode_class_accuracy"] = float(np.mean(modes)) if modes else None
    out = {"prediction_file": os.path.relpath(pred_file, ROOT), "prediction_written": doc["written"], "prediction_sha256": doc.get("sha256_of_content"), "evaluated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows, "summary": summary}
    return out


def classify_mode_class(mode):
    if mode is None: return None
    if "abrupt" in mode: return "abrupt"
    if "progressive" in mode: return "progressive"
    if "stepwise" in mode: return "stepwise"
    return "none"


def spec(campaign, stage, batch_name, structures, aqs):
    return {"campaign": campaign, "stage": stage, "batch_name": batch_name, "device": "mps", "dtype": "float32", "aqs": aqs, "structures": structures}


def write_specs(stage_dir, batches):
    os.makedirs(stage_dir, exist_ok=True)
    for b in batches:
        json.dump(b, open(os.path.join(stage_dir, b["batch_name"] + ".json"), "w"), indent=1)
