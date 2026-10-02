"""Stage 7: unseen holdout designs + quantitative predictions recorded BEFORE simulation.

Predictor (documented, mechanistic, fitted on Stages 1-6 records only):
  strength  : sigma = minSF x sigma_net(class), with sigma_net taken as the median net-section strength of
              the completed designs of the same ligament class (straight load-parallel ligaments, coarse
              round pores, fine round pores, random network, short bridges between perpendicular slits);
  modulus   : Y = Y_graphene x g(phi, class) from a linear fit of Y/Y0 vs porosity within the class;
  failure strain, work, localisation, mode : k-nearest completed designs in descriptor space
              (minSF, edge-atom fraction, anisotropy, porosity, ligament width) within the class (k=3, distance-weighted),
              strength-scaled work (W_pred = W_knn x sigma_pred/sigma_knn).
The classes are assigned from the generated (relaxed-free) geometry before any simulation."""
from __future__ import annotations
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
from experiments import db
from experiments.campaign_design import AQS, S, match
from experiments.stage_tools import spec, write_specs, write_predictions, classify_mode_class

Y0 = 250.0


def ligament_class(family, params, desc):
    if family == "slit_array":
        ang = params.get("angle_deg", 0.0) % 180
        return "straight" if ang < 30 or ang > 150 else ("bridges" if abs(ang - 90) < 30 else "oblique")
    if family == "nanomesh_single":
        aspect = params.get("aspect", 1.0); ang = params.get("angle_deg", 0.0)
        if aspect > 1.2:
            return "straight" if abs(ang - 90) < 30 else "fine_round"
        return "coarse_round" if desc["ligament_width_mean_A"] >= 18 else "fine_round"
    if family == "nanomesh_hier":
        return "fine_round"
    if family == "voronoi_network":
        return "random"
    if family == "strut_lattice":
        return "straight" if any(abs(a % 180) < 1 for a in params.get("angles_deg", [])) else "random"
    if family in ("precrack", "ring_around_hole", "vacancies", "graded_pores"):
        return "flawed_sheet"
    return "fine_round"


def fit_classes():
    recs = [r for r in db.all_records() if r.get("status") == "completed" and r.get("campaign") == "discovery" and r.get("stage") in ("stage1_baselines", "stage2_reconnaissance", "stage4_discriminating", "stage5_deep", "stage6_seeds")]
    rows = []
    for r in recs:
        d = r["descriptors"]; m = r["metrics"]
        cls = ligament_class(r["family"], r["params"], d)
        msf = d.get("min_solid_fraction_across_x") or np.nan
        rows.append(dict(name=r["name"], cls=cls, msf=msf, sigma=m["strength_Nm"], snet=m["strength_Nm"] / msf if msf else np.nan, Y=m["modulus_2d_Nm"], phi=r.get("porosity") or 0.0,
                         fuc=d.get("fraction_undercoordinated"), aniso=d.get("anisotropy_index"), lw=d.get("ligament_width_mean_A"), ef=m["failure_strain"], W=m["work_to_failure_J_m2"], L=m["damage_localization"], mode=m["fracture_mode"]))
    return rows


def predict(family, params, rows):
    a = D.generate(family, **params)
    d = compute_descriptors(a, a.info.get("design"))
    cls = ligament_class(family, params, d)
    msf = d["min_solid_fraction_across_x"]; phi = a.info["porosity"]
    same = [x for x in rows if x["cls"] == cls and np.isfinite(x["snet"])]
    if len(same) < 3:
        same = [x for x in rows if np.isfinite(x["snet"])]
    snet = float(np.median([x["snet"] for x in same]))
    sigma = msf * snet
    # modulus: linear fit Y vs phi within class (fallback all)
    xs = np.array([x["phi"] for x in same]); ys = np.array([x["Y"] for x in same])
    if len(same) >= 3 and np.ptp(xs) > 0.05:
        p = np.polyfit(xs, ys, 1); Y = float(np.polyval(p, phi))
    else:
        Y = float(np.median(ys))
    # kNN for failure strain / work / localisation / mode
    feats = lambda x: np.array([x["msf"], x["fuc"], x["aniso"], x["phi"], x["lw"] / 30.0])
    q = np.array([msf, d["fraction_undercoordinated"], d["anisotropy_index"], phi, d["ligament_width_mean_A"] / 30.0])
    cand = [x for x in same if all(np.isfinite(feats(x)))]
    dist = np.array([np.linalg.norm(feats(x) - q) for x in cand])
    idx = np.argsort(dist)[:3]; w = 1.0 / (dist[idx] + 0.05); w /= w.sum()
    ef = float(np.sum(w * np.array([cand[i]["ef"] for i in idx])))
    Wk = float(np.sum(w * np.array([cand[i]["W"] for i in idx]))); sk = float(np.sum(w * np.array([cand[i]["sigma"] for i in idx])))
    Wp = Wk * sigma / sk if sk > 0 else Wk
    L = float(np.sum(w * np.array([cand[i]["L"] for i in idx])))
    modes = [classify_mode_class(cand[i]["mode"]) for i in idx]
    mode = max(set(modes), key=modes.count)
    return dict(strength_Nm=round(sigma, 1), modulus_2d_Nm=round(Y, 0), failure_strain=round(ef, 3), work_to_failure_J_m2=round(Wp, 2), damage_localization=round(L, 2),
                fracture_mode_class=mode, basis=f"class={cls}, minSF={msf:.2f}, sigma_net(median of {len(same)})={snet:.1f} N/m; kNN: {[cand[i]['name'] for i in idx]}",
                crack_path_or_mechanism={"straight": "load-parallel strips fail together at the bridges/corners (single avalanche)",
                                         "coarse_round": "damage at pore poles, sequential row failure with load retention",
                                         "fine_round": "fine ligaments fail progressively at pore edges; load carried by remaining rows / veins",
                                         "random": "weakest thin ligament fails first, crack follows the thinnest path (localised)",
                                         "bridges": "short bridges between slit tips fail one by one, strips bend",
                                         "oblique": "ligaments rotate toward the load, shear-like stepwise failure",
                                         "flawed_sheet": "crack nucleates at the flaw tip / largest defect and runs across the sheet"}[cls]), a


def holdout_designs():
    L, LH = 100, 120
    H = []
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 40, "vein_w": 12, "vein_dirs": "x", "aspect": 2.0, "angle_deg": 90}, 0.20, "pore_d", 2, 11)
    H.append(("S7_H2_veinsX_W12_ellipseX", "nanomesh_hier", p, "unseen interaction: load-parallel veins (12 A) + fine pores elongated along the load"))
    p, f, n = match("slit_array", {"Lx": L, "Ly": L, "slit_w": 4.0, "period_x": 30, "period_y": 16, "angle_deg": 20}, 0.20, "slit_len", 4, 28)
    H.append(("S7_slit_20deg", "slit_array", p, "unseen orientation between 0 and 45 degrees"))
    p, f, n = match("nanomesh_single", {"Lx": LH, "Ly": LH, "period": 40, "shape": "square", "stagger": True}, 0.20, "pore_d", 5, 36)
    H.append(("S7_square_D40_staggered", "nanomesh_single", p, "unseen: staggered (brick-like) coarse square pores"))
    p, f, n = match("voronoi_network", {"Lx": L, "Ly": L, "n_cells": 35, "regularity": 0.8, "seed": 7}, 0.20, "ligament_w", 2, 20, increasing=False)
    H.append(("S7_voronoi_n35_reg0.8_s7", "voronoi_network", p, "unseen network (35 cells, regularity 0.8, new seed)"))
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 3, "period": 10, "domain": 30, "vein_w": 8, "level3_domain": 60, "level3_vein": 16}, 0.20, "pore_d", 2, 9)
    H.append(("S7_H3_D30_60_wideveins", "nanomesh_hier", p, "unseen three-level mesh with wide veins (8/16 A)"))
    H.append(("S7_hole30_rings3", "ring_around_hole", {"Lx": L, "Ly": L, "hole_d": 30, "n_rings": 3, "ring_pore_d": 5.0, "ring_gap": 7.0}, "unseen flaw halo: 30 A hole + 3 rings"))
    p, f, n = match("nanomesh_single", {"Lx": L, "Ly": L, "period": 24}, 0.30, "pore_d", 4, 22)
    H.append(("S7_mesh_p24_phi0.3", "nanomesh_single", p, "unseen period/porosity combination (24 A, phi 0.3)"))
    H.append(("S7_precrack_L50_cell150", "precrack", {"Lx": 150, "Ly": 150, "crack_len": 50, "crack_w": 3.0, "angle_deg": 90}, "unseen: 50 A crack (plateau test)"))
    p, f, n = match("graded_pores", {"Lx": L, "Ly": L, "period": 16, "pore_d_min": 5.0, "mode": "x"}, 0.20, "pore_d_max", 5, 14)
    H.append(("S7_graded_x_weak", "graded_pores", p, "unseen: weaker gradient (5 A minimum pore)"))
    p, f, n = match("slit_array", {"Lx": L, "Ly": L, "slit_w": 4.0, "period_x": 30, "period_y": 16, "angle_deg": 0, "orientation": "armchair"}, 0.20, "slit_len", 4, 28)
    H.append(("S7_slit_0deg_armchair", "slit_array", p, "unseen: parallel slits with armchair lattice along the load"))
    p, f, n = match("strut_lattice", {"Lx": L, "Ly": L, "spacing": 25, "angles_deg": [0, 90]}, 0.30, "strut_w", 3, 22, increasing=False)
    H.append(("S7_strut_090_phi0.3", "strut_lattice", p, "unseen: 0/90 strut lattice at porosity 0.30"))
    p, f, n = match("nanomesh_single", {"Lx": L, "Ly": L, "period": 16, "jitter": 1.5, "size_disorder": 0.1, "seed": 11}, 0.20, "pore_d", 3, 14)
    H.append(("S7_mesh_jitter1.5_sd0.1_s11", "nanomesh_single", p, "unseen combined disorder (jitter 1.5 A + 10% size dispersion, new seed)"))
    return H


if __name__ == "__main__":
    rows = fit_classes()
    print("fitted on", len(rows), "records; class medians of sigma_net:", {c: round(float(np.median([x['snet'] for x in rows if x['cls'] == c and np.isfinite(x['snet'])])), 1) for c in set(x['cls'] for x in rows)})
    st, preds = [], []
    for name, fam, params, why in holdout_designs():
        pr, a = predict(fam, params, rows)
        seed = params.get("seed", 0)
        st.append(S(name, fam, params, seed=seed, reason="Stage 7 holdout: " + why, hypothesis="prediction test of the net-section/class rule and kNN transfer", tags=["stage7", "holdout"], prediction=pr))
        preds.append({"name": name, "family": fam, "params": params, "predicted": pr, "rationale": why})
        print(f"  {name:32s} N={len(a):5d} phi={a.info['porosity']:.3f} pred sigma={pr['strength_Nm']} ef={pr['failure_strain']} W={pr['work_to_failure_J_m2']} L={pr['damage_localization']} mode={pr['fracture_mode_class']}")
    path, h = write_predictions("holdout_predictions", preds, notes="Predictions for the 12 unseen Stage 7 designs, written before their simulations were launched (see campaign_stage7_holdouts.py for the predictor).")
    batches = [spec("discovery", "stage7_holdouts", f"stage7_b{k}", st[3 * k:3 * k + 3], AQS) for k in range((len(st) + 2) // 3)]
    write_specs(os.path.join(ROOT, "experiments", "specs", "stage7"), batches)
    print("prediction file", path, "sha256", h[:16]); print("batches", len(batches))
