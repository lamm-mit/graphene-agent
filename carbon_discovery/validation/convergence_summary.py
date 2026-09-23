"""TESTS 13-15: summarise the convergence batches (campaign 'validation', stage 'convergence')."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from experiments import db


def main():
    recs = [r for r in db.all_records() if r.get("campaign") == "validation" and r.get("stage") == "convergence" and r.get("status") == "completed"]
    by = {r["name"]: r for r in recs}
    out = {}
    def row(r):
        m = r["metrics"]; return dict(strength_Nm=m["strength_Nm"], failure_strain=m["failure_strain"], first_damage_strain=m["first_damage_strain"], work=m["work_to_failure_J_m2"], modulus=m["modulus_2d_Nm"], n_atoms=r["n_atoms"], fire_total=r["stats"].get("fire_iterations", -1), wall=r.get("wall_time_s"))
    # TEST 13 minimizer tolerance
    fm = [(f, by.get(f"mesh64_fmax{f}")) for f in [0.05, 0.02, 0.01, 0.005]]
    fm = [(f, r) for f, r in fm if r]
    if len(fm) >= 2:
        rows = [row(r) for _, r in fm]; ref = rows[-1]
        errs = {k: max(abs(rr[k] - ref[k]) / max(abs(ref[k]), 1e-9) for rr in rows[1:]) for k in ["strength_Nm", "failure_strain", "first_damage_strain"]}
        prod = [rr for f, rr in zip([f for f, _ in fm], rows) if f == 0.02]
        e_prod = max(abs(prod[0][k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in ["strength_Nm", "first_damage_strain"]) if prod else np.nan
        out["minimizer"] = {"name": "minimiser convergence (fmax 0.05 -> 0.005 eV/A, 64 A nanomesh)", "expected": "metrics converge; production fmax=0.02 within 3% of fmax=0.005",
                            "observed": "; ".join(f"fmax={f}: sigma={rr['strength_Nm']:.2f} N/m, eps_fd={rr['first_damage_strain']:.4f}, eps_f={rr['failure_strain']:.4f}, FIRE its={rr['fire_total']}" for (f, _), rr in zip(fm, rows)),
                            "error": float(e_prod), "tolerance": 0.03, "pass": bool(e_prod < 0.03), "series": {"x": [f for f, _ in fm], "strength_Nm": [rr["strength_Nm"] for rr in rows], "failure_strain": [rr["failure_strain"] for rr in rows], "first_damage_strain": [rr["first_damage_strain"] for rr in rows]}}
    # TEST 14 strain step
    ds = [(d, by.get(n)) for d, n in [(0.01, "mesh64_ds0.01_ref1"), (0.005, "mesh64_ds0.005_ref1"), (0.0025, "mesh64_ds0.0025_ref1"), ("0.005 no refinement", "mesh64_ds0.005_ref0")]]
    ds = [(d, r) for d, r in ds if r]
    if len(ds) >= 2:
        rows = [row(r) for _, r in ds]; ref = [rr for (d, _), rr in zip(ds, rows) if d == 0.0025]
        ref = ref[0] if ref else rows[-1]
        prod = [rr for (d, _), rr in zip(ds, rows) if d == 0.005]
        e_prod = max(abs(prod[0][k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in ["strength_Nm", "first_damage_strain"]) if prod else np.nan
        out["strain_step"] = {"name": "strain-increment convergence (1%, 0.5%, 0.25% with bisection; 0.5% without)", "expected": "production increment 0.5% (+bisection) within 3% of 0.25% for strength and first-damage strain",
                              "observed": "; ".join(f"d_eps={d}: sigma={rr['strength_Nm']:.2f} N/m, eps_fd={rr['first_damage_strain']:.4f}, eps_f={rr['failure_strain']:.4f}, W={rr['work']:.2f}" for (d, _), rr in zip(ds, rows)),
                              "error": float(e_prod), "tolerance": 0.03, "pass": bool(e_prod < 0.03), "series": {"x": [d if not isinstance(d, str) else 0.005 for d, _ in ds], "strength_Nm": [rr["strength_Nm"] for rr in rows], "failure_strain": [rr["failure_strain"] for rr in rows], "first_damage_strain": [rr["first_damage_strain"] for rr in rows]}}
    # TEST 15 size
    sz = [(32 * k, by.get(f"mesh_rep{k}")) for k in [1, 2, 3]]; sz = [(L, r) for L, r in sz if r]
    ck = [(L, by.get(f"precrack12_cell{L}")) for L in [40, 60, 90]]; ck = [(L, r) for L, r in ck if r]
    if len(sz) >= 2:
        rows = [row(r) for _, r in sz]
        e = max(abs(rows[i]["strength_Nm"] - rows[-1]["strength_Nm"]) / rows[-1]["strength_Nm"] for i in range(len(rows) - 1))
        crows = [row(r) for _, r in ck]
        obs = "mesh repeats: " + "; ".join(f"L={L}: sigma={rr['strength_Nm']:.2f}, eps_fd={rr['first_damage_strain']:.4f}, eps_f={rr['failure_strain']:.4f}" for (L, _), rr in zip(sz, rows))
        if crows:
            obs += " | crack 12 A: " + "; ".join(f"cell {L}: sigma={rr['strength_Nm']:.2f}, eps_fd={rr['first_damage_strain']:.4f}" for (L, _), rr in zip(ck, crows))
        out["size"] = {"name": "system-size convergence (exact periodic repeats of a 32 A mesh pattern: 1x, 2x, 3x; fixed 12 A crack in 40/60/90 A cells)", "expected": "strength of periodic repeats size-independent within 5%; crack strength converges with cell size (image interaction)",
                       "observed": obs, "error": float(e), "tolerance": 0.05, "pass": bool(e < 0.05),
                       "series": {"x": [L for L, _ in sz], "strength_Nm": [rr["strength_Nm"] for rr in rows], "failure_strain": [rr["failure_strain"] for rr in rows], "first_damage_strain": [rr["first_damage_strain"] for rr in rows]},
                       "crack_series": {"x": [L for L, _ in ck], "strength_Nm": [rr["strength_Nm"] for rr in crows], "first_damage_strain": [rr["first_damage_strain"] for rr in crows]}}
    json.dump(out, open(os.path.join(ROOT, "validation", "results", "convergence_summary.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
