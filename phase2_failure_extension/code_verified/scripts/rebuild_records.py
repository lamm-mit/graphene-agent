"""Rebuild experiment records from retained raw trajectories + batch specs + queue logs.
(Used once after the run directories were accidentally deleted on 2026-09-05; every rebuilt record is
flagged `reconstructed_from_trajectory: true`.)  Physics quantities (stress-strain, bonds, per-atom
fields, positions) come from the stored trajectory; metrics/descriptors are recomputed with the same code."""
from __future__ import annotations
import sys, os, json, glob, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from ase import Atoms
from experiments import db
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors, bond_graph
from analysis.metrics import compute_metrics
from simulation.topology import spanning_x, largest_fragment_fraction
from simulation.aqs.aqs import AQSConfig, EV_A2_TO_N_M
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from scripts.run_batch import precheck, viability


def spec_index():
    idx = {}
    for f in glob.glob(os.path.join(ROOT, "experiments", "specs", "*", "**", "*.json*"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for s in d.get("structures", []):
            idx[s["name"]] = (s, d)
    return idx


def log_index():
    m = {}
    for f in glob.glob(os.path.join(ROOT, "experiments", "specs", "*", "logs", "*.log")):
        for line in open(f):
            mm = re.match(r"\[done\] (\S+) -> (run_\S+): (.*?); strength .*?, (.*)$", line.strip())
            if mm:
                name, rid, via, rest = mm.groups()
                term = rest.split(", ", 2)[-1] if "mode" in rest else rest
                m[rid] = {"name": name, "viability_logged": via, "termination": rest.split("), ")[-1] if "), " in rest else rest.split(", ")[-1], "log": f}
    return m


def rebuild(rid, meta, specs, eng):
    s, batch = specs[meta["name"]]
    d = np.load(os.path.join(ROOT, "trajectories", rid + ".npz"))
    P, C = d["positions"], d["cells"]; nfr = len(P); n = P.shape[1]
    params = dict(s.get("params", {}))
    if s.get("seed") is not None and s["family"] != "pristine":
        params["seed"] = s["seed"]
    atoms = D.generate(s["family"], **params)
    assert len(atoms) == n, (meta["name"], len(atoms), n)
    bonds = [d[f"bonds_{k}"] for k in range(nfr)]
    # bond graphs with shifts per frame (for spanning / fragments) + damage events from consecutive frames
    spanning, lf, nfrag, events, nb_new = [], [], [], [], []
    prev = None
    for k in range(nfr):
        a = Atoms("C" * n, positions=P[k].astype(float), cell=C[k], pbc=[True, True, False])
        i, j, S, _ = bond_graph(a, r_bond=2.0)
        spanning.append(spanning_x(i, j, S, n)); f_, nf = largest_fragment_fraction(i, j, n); lf.append(f_); nfrag.append(nf)
        cur = set(map(tuple, np.sort(bonds[k], axis=1).tolist()))
        if prev is not None:
            lost = prev - cur
            nb_new.append(len(lost))
            for (ii, jj) in sorted(lost):
                mid = 0.5 * (P[k][ii] + P[k][jj]); events.append((float(d["eps_x"][k]), int(ii), int(jj), [0, 0, 0], [float(mid[0]), float(mid[1])]))
        else:
            nb_new.append(0)
        prev = cur
    res = {"name": meta["name"], "n_atoms": n, "eps_x": d["eps_x"], "eps_y": d["eps_y"], "sigma_xx": d["sigma_xx"], "sigma_yy": d["sigma_yy"], "sigma_xy": d["sigma_xy"],
           "energy": d["energy"], "n_bonds": d["n_bonds"], "n_broken_new": np.array(nb_new), "n_formed_new": np.zeros(nfr, int), "n_broken_cum": d["n_broken_cum"],
           "fire_iterations": np.full(nfr, -1), "fire_converged": np.ones(nfr, bool), "transverse_iterations": np.zeros(nfr, int), "spanning": np.array(spanning),
           "largest_fragment": np.array(lf), "n_fragments": np.array(nfrag), "wall_time": np.full(nfr, np.nan), "L0": np.array([C[0][0, 0], C[0][1, 1]]),
           "peak_sigma": float(d["sigma_xx"].max()), "peak_strain": float(d["eps_x"][int(np.argmax(d["sigma_xx"]))]),
           "first_damage_strain": float(d["eps_x"][np.where(d["n_broken_cum"] > 0)[0][0]]) if (d["n_broken_cum"] > 0).any() else float("nan"),
           "termination": meta["termination"], "broken_bond_events": events, "positions": [P[k] for k in range(nfr)], "cells": [C[k] for k in range(nfr)],
           "peratom_energy": [d["peratom_energy"][k] for k in range(nfr)], "peratom_virial": [d["peratom_virial"][k] for k in range(nfr)], "bonds": bonds,
           "coordination": [d["coordination"][k] for k in range(nfr)], "stats": {"note": "not retained"}, "config": {**AQSConfig().__dict__, **batch.get("aqs", {})}}
    atoms_relaxed = Atoms("C" * n, positions=P[0].astype(float), cell=C[0], pbc=[True, True, False])
    atoms_final = Atoms("C" * n, positions=P[-1].astype(float), cell=C[-1], pbc=[True, True, False])
    sysb = BatchedSystem([atoms], eng); nl = sysb.build_neighbor_list(skin=0.3)
    out0 = eng.evaluate(sysb.positions, sysb.cell, sysb.sid, nl, compute_stress=False)
    E_init = float(out0["energy"][0]); Fmax_init = float(np.sqrt((out0["forces"] ** 2).sum(1).max()))
    pre = precheck(atoms)
    via = viability(atoms, atoms_relaxed, E_init, float(d["energy"][0]), True, pre); via["max_force_initial_eV_A"] = Fmax_init
    metrics = compute_metrics(res); desc = compute_descriptors(atoms_relaxed, atoms.info.get("design"))
    rec = {k: s.get(k) for k in ["name", "family", "params", "seed", "reason", "hypothesis", "prediction", "parent", "tags", "notes"]}
    rec.update(campaign=batch.get("campaign"), stage=batch.get("stage"), batch_name=batch.get("batch_name"), precheck=pre, design=atoms.info.get("design", {}),
               porosity=atoms.info.get("porosity"), n_atoms=n, hierarchy_levels=atoms.info.get("design", {}).get("hierarchy_levels"), orientation=atoms.info.get("design", {}).get("orientation"),
               run_id=rid, environment=db.environment_record(), code=db.code_version(), device=batch.get("device"), precision=batch.get("dtype"), engine=eng.info.__dict__,
               aqs_config=res["config"], boundary_conditions={"pbc": [True, True, False], "loading": "uniaxial engineering strain along x (periodic)", "transverse": "relaxed to sigma_yy = 0 (secant)", "orientation_x": rec.get("orientation"), "cell0_A": res["L0"].tolist()},
               viability=via["classification"], viability_details=via, metrics=metrics, descriptors=desc, termination=meta["termination"], wall_time_s=None, n_frames=nfr,
               stats=res["stats"], status="completed", broken_bond_events=events[:5000], first_damage_strain=res["first_damage_strain"], peak_sigma_Nm=res["peak_sigma"] * EV_A2_TO_N_M,
               peak_strain=res["peak_strain"], anomaly_flags=["record reconstructed from retained trajectory (2026-09-05); FIRE iteration counts and wall time not retained"],
               reconstructed_from_trajectory=True, viability_logged=meta["viability_logged"])
    db.save_run(rid, rec, res, atoms, atoms_relaxed, atoms_final, save_trajectory=False)
    return rec


if __name__ == "__main__":
    specs = spec_index(); logs = log_index()
    eng = TorchRebo2Scr(device="cpu", dtype=torch.float64)
    done = 0
    for rid, meta in sorted(logs.items()):
        if not os.path.exists(os.path.join(ROOT, "trajectories", rid + ".npz")) or os.path.exists(os.path.join(db.RUNS_DIR, rid, "record.json")):
            continue
        if meta["name"] not in specs:
            print("no spec for", meta["name"]); continue
        rec = rebuild(rid, meta, specs, eng)
        m = rec["metrics"]
        print(f"rebuilt {rid} {meta['name']}: {rec['viability']} (logged: {meta['viability_logged']}) strength {m['strength_Nm']:.2f} fail {m['failure_strain']:.3f} {m['fracture_mode']}", flush=True)
        done += 1
    db.rebuild_index(); print("rebuilt", done, "records")
