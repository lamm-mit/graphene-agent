"""Reproduce a stored run from its record (family, parameters, seed, AQS configuration, device/precision)
and compare the key metrics with the stored ones.  Usage: python scripts/reproduce_run.py <run_id> [--device cpu --dtype float64]"""
from __future__ import annotations
import sys, os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from experiments import db
from atomistics.structures import design_space as D
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner, AQSConfig
from analysis.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("--device", default=None); ap.add_argument("--dtype", default=None); ap.add_argument("--max-strain", type=float, default=None)
    a = ap.parse_args()
    rec = db.load_record(a.run_id)
    params = dict(rec["params"]); 
    if rec.get("seed") is not None and rec["family"] != "pristine":
        params["seed"] = rec["seed"]
    atoms = D.generate(rec["family"], **params)
    assert len(atoms) == rec["n_atoms"], f"regenerated structure has {len(atoms)} atoms, record has {rec['n_atoms']}"
    device = a.device or rec["device"]; dtype = {"float32": torch.float32, "float64": torch.float64}[a.dtype or rec["precision"]]
    cfg_d = {k: v for k, v in rec["aqs_config"].items() if k in AQSConfig.__dataclass_fields__}
    if a.max_strain: cfg_d["max_strain"] = a.max_strain
    cfg = AQSConfig(**cfg_d)
    eng = TorchRebo2Scr(device=device, dtype=dtype)
    sysb = BatchedSystem([atoms], eng, names=[rec["name"]])
    res = AQSRunner(eng, sysb, cfg, log=print).run()[0]
    m = compute_metrics(res); m0 = rec["metrics"]
    print("\nmetric                     stored        reproduced")
    for k in ["modulus_2d_Nm", "strength_Nm", "strain_at_peak", "first_damage_strain", "failure_strain", "work_to_failure_J_m2", "damage_localization"]:
        print(f"{k:26s} {m0.get(k, float('nan')):12.4f} {m.get(k, float('nan')):12.4f}")
    print("stored fracture mode:", m0.get("fracture_mode"), "| reproduced:", m.get("fracture_mode"))


if __name__ == "__main__":
    main()
