"""Run one batch specification (JSON) of AQS tensile simulations and store everything in the
experiment database.

spec = {
  "campaign": str, "stage": str, "batch_name": str,
  "device": "mps"|"cpu", "dtype": "float32"|"float64",
  "aqs": {AQSConfig overrides},
  "structures": [ {"name", "family", "params": {...}, "seed", "reason", "hypothesis", "prediction", "parent", "tags"} ... ]
}
"""
from __future__ import annotations
import sys, os, json, time, traceback, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from ase.neighborlist import primitive_neighbor_list
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors, bond_graph
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner, AQSConfig, EV_A2_TO_N_M
from simulation.topology import spanning_x
from analysis.metrics import compute_metrics
from experiments import db


def precheck(atoms):
    n = len(atoms)
    info = {"n_atoms": n}
    if n < 10:
        info["status"] = "geometry generation failure (too few atoms)"
        return info
    i, j, d = primitive_neighbor_list("ijd", pbc=atoms.pbc, cell=np.asarray(atoms.cell, float), positions=atoms.positions, cutoff=1.9)
    info["min_distance"] = float(d.min()) if len(d) else float("inf")
    info["duplicates"] = int((d < 0.5).sum() // 2)
    coord = np.bincount(i, minlength=n)
    info["coordination_hist_initial"] = {int(k): int((coord == k).sum()) for k in np.unique(coord)}
    bi, bj, bS, _ = bond_graph(atoms)
    info["spanning_x_initial"] = bool(spanning_x(bi, bj, bS, n))
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    g = sp.coo_matrix((np.ones(len(bi)), (bi, bj)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    info["n_fragments_initial"] = int(ncomp)
    info["isolated_atoms_initial"] = int((coord == 0).sum())
    if info["duplicates"] > 0 or info["min_distance"] < 0.9:
        info["status"] = "geometry generation failure (overlapping atoms)"
    elif not info["spanning_x_initial"]:
        info["status"] = "disconnected (no load-bearing path)"
    else:
        info["status"] = "ok"
    return info


def viability(atoms_initial, atoms_relaxed, E_initial, E_relaxed, fire_ok, pre):
    n = len(atoms_initial)
    disp = atoms_relaxed.positions - atoms_initial.positions
    cell = np.asarray(atoms_initial.cell, float)
    # remove drift, wrap displacements periodically
    for k in range(2):
        disp[:, k] -= cell[k, k] * np.round(disp[:, k] / cell[k, k])
    disp -= disp.mean(0)
    dn = np.linalg.norm(disp, axis=1)
    bi0, bj0, S0, _ = bond_graph(atoms_initial)
    bi1, bj1, S1, _ = bond_graph(atoms_relaxed)
    k0 = set(zip(bi0.tolist(), bj0.tolist())); k1 = set(zip(bi1.tolist(), bj1.tolist()))
    c0 = np.zeros(n, int); np.add.at(c0, bi0, 1); np.add.at(c0, bj0, 1)
    c1 = np.zeros(n, int); np.add.at(c1, bi1, 1); np.add.at(c1, bj1, 1)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    g = sp.coo_matrix((np.ones(len(bi1)), (bi1, bj1)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    v = {"energy_initial_eV": float(E_initial), "energy_relaxed_eV": float(E_relaxed),
         "relaxation_energy_eV_per_atom": float((E_relaxed - E_initial) / n),
         "max_displacement_A": float(dn.max()), "rms_displacement_A": float(np.sqrt((dn ** 2).mean())),
         "lost_neighbors": int(len(k0 - k1)), "new_neighbors": int(len(k1 - k0)),
         "coordination_changes": int((c0 != c1).sum()), "isolated_atoms": int((c1 == 0).sum()),
         "n_fragments": int(ncomp), "spanning_x": bool(spanning_x(bi1, bj1, S1, n)),
         "coordination_hist_relaxed": {int(k): int((c1 == k).sum()) for k in np.unique(c1)},
         "minimizer_converged": bool(fire_ok)}
    if pre["status"].startswith("geometry"):
        cls = pre["status"]
    elif not fire_ok:
        cls = "failed during minimization"
    elif ncomp > 1 and v["isolated_atoms"] > 0:
        cls = "fragmented"
    elif not v["spanning_x"]:
        cls = "disconnected"
    elif v["max_displacement_A"] > 1.0 or v["coordination_changes"] > 0.05 * n:
        cls = "strongly reconstructed"
    elif v["max_displacement_A"] > 0.3 or v["coordination_changes"] > 0:
        cls = "reconstructed but viable"
    else:
        cls = "stable"
    v["classification"] = cls
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--no-traj", action="store_true")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    device = args.device or spec.get("device", "mps")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype or spec.get("dtype", "float32")]
    t_start = time.time()
    log_lines = []
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_lines.append(s)
    eng = TorchRebo2Scr(device=device, dtype=dtype)
    env = db.environment_record()
    code = db.code_version()
    cfg = AQSConfig(**spec.get("aqs", {}))
    # ---- build structures
    built, records, failed = [], [], []
    for s in spec["structures"]:
        rec = {k: s.get(k) for k in ["name", "family", "params", "seed", "reason", "hypothesis", "prediction", "parent", "tags", "notes"]}
        rec.update(campaign=spec.get("campaign"), stage=spec.get("stage"), batch_name=spec.get("batch_name"))
        try:
            params = dict(s.get("params", {}))
            if "seed" in s and s["seed"] is not None and s["family"] != "pristine":
                params["seed"] = s["seed"]
            atoms = D.generate(s["family"], **params)
            pre = precheck(atoms)
            rec["precheck"] = pre
            rec["design"] = atoms.info.get("design", {})
            rec["porosity"] = atoms.info.get("porosity")
            rec["n_atoms"] = len(atoms)
            rec["hierarchy_levels"] = rec["design"].get("hierarchy_levels")
            rec["orientation"] = rec["design"].get("orientation")
            if pre["status"] != "ok":
                rec["viability"] = pre["status"]
                rec["status"] = "not simulated"
                failed.append((rec, atoms))
                log(f"[skip] {s['name']}: {pre['status']}")
                continue
            built.append((rec, atoms))
        except Exception as e:
            rec["viability"] = "geometry generation failure"
            rec["status"] = "generation error: " + repr(e)
            rec["traceback"] = traceback.format_exc()
            failed.append((rec, None))
            log(f"[fail] {s['name']}: {e}")
    for rec, atoms in failed:
        run_id = db.new_run_id("fail")
        rec.update(run_id=run_id, environment=env, code=code, device=device, precision=str(dtype)[6:])
        db.dump_json(rec, os.path.join(db.RUNS_DIR, run_id, "record.json"))
    if not built:
        log("nothing to simulate")
        return
    names = [r["name"] for r, _ in built]
    atoms_list = [a for _, a in built]
    log(f"batch {spec.get('batch_name')}: {len(built)} structures, N = {[len(a) for a in atoms_list]}, device={device} {dtype}")
    sysb = BatchedSystem(atoms_list, eng, names=names)
    # energies of the generated (unrelaxed) geometries
    nl0 = sysb.build_neighbor_list(skin=cfg.skin)
    out0 = eng.evaluate(sysb.positions, sysb.cell, sysb.sid, nl0, compute_stress=False)
    E_init = out0["energy"].cpu().numpy(); Fmax_init = np.sqrt(sysb.seg_max((out0["forces"] ** 2).sum(1)).cpu().numpy())
    saved = {}
    def postprocess(b, res):
        rec, atoms = built[b]
        run_id = db.new_run_id("run")
        try:
            atoms_relaxed = sysb.to_atoms(b, positions=res["positions"][0], cell=res["cells"][0])
            atoms_final = sysb.to_atoms(b, positions=res["positions"][-1], cell=res["cells"][-1])
            fire_ok = bool(res["fire_converged"][0]) if len(res["fire_converged"]) else False
            via = viability(atoms, atoms_relaxed, E_init[b], res["energy"][0], fire_ok, rec["precheck"])
            via["max_force_initial_eV_A"] = float(Fmax_init[b])
            metrics = compute_metrics(res)
            desc = compute_descriptors(atoms_relaxed, rec["design"])
            rec.update(run_id=run_id, environment=env, code=code, device=device, precision=str(dtype)[6:],
                       engine=eng.info.__dict__, aqs_config=res["config"], boundary_conditions={
                           "pbc": [True, True, False], "loading": "uniaxial engineering strain along x (periodic)",
                           "transverse": "relaxed to sigma_yy = 0 (secant)" if cfg.transverse_relax else "fixed transverse strain",
                           "orientation_x": rec["design"].get("orientation"), "cell0_A": res["L0"].tolist()},
                       viability=via["classification"], viability_details=via, metrics=metrics, descriptors=desc,
                       termination=res["termination"], wall_time_s=float(res["wall_time"][-1]) if len(res["wall_time"]) else None,
                       n_frames=len(res["eps_x"]), stats=res["stats"], status="completed",
                       broken_bond_events=res["broken_bond_events"][:5000], first_damage_strain=res["first_damage_strain"],
                       peak_sigma_Nm=res["peak_sigma"] * EV_A2_TO_N_M, peak_strain=res["peak_strain"])
            # per-run convergence flags
            rec["anomaly_flags"] = []
            if not all(res["fire_converged"]):
                rec["anomaly_flags"].append("minimization not converged in %d of %d steps" % (int((~np.asarray(res["fire_converged"])).sum()), len(res["fire_converged"])))
            if "numerical" in res["termination"]:
                rec["anomaly_flags"].append("numerical failure")
            db.save_run(run_id, rec, res, atoms, atoms_relaxed, atoms_final, save_trajectory=not args.no_traj)
            log(f"[done] {rec['name']} -> {run_id}: {via['classification']}; strength {metrics['strength_Nm']:.2f} N/m @ {metrics['strain_at_peak']:.3f}, "
                f"fail {metrics['failure_strain']:.3f}, mode {metrics['fracture_mode']}, {res['termination']}")
            saved[b] = run_id
        except Exception as e:
            rec.update(run_id=run_id, status="postprocessing error: " + repr(e), traceback=traceback.format_exc(), device=device, precision=str(dtype)[6:])
            db.dump_json(rec, os.path.join(db.RUNS_DIR, run_id, "record.json"))
            log(f"[error] {rec['name']}: {e}")
            saved[b] = run_id
        db.rebuild_index()
    runner = AQSRunner(eng, sysb, cfg, log=log, on_done=lambda b, res: postprocess(b, res) if b not in saved else None)
    results = runner.run()
    t_sim = time.time() - t_start
    for b, res in enumerate(results):
        if b not in saved:
            postprocess(b, res)
    log(f"batch finished in {time.time() - t_start:.0f}s (simulation {t_sim:.0f}s); evaluations {runner.stats}")
    db.rebuild_index()
    with open(os.path.join(db.DB_DIR, "logs", os.path.basename(args.spec).replace(".running", "").replace(".json", ".log")), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    os.makedirs(os.path.join(db.DB_DIR, "logs"), exist_ok=True)
    main()
