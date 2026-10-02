"""Stage 4 (discriminating matched pairs with recorded predictions), Stage 5 (deep mechanism studies:
hierarchy variables, interactions, scaling), Stage 6 (seed statistics).  Writes spec batches and the
Stage 4 prediction file (timestamped, hashed) BEFORE the simulations are launched."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
from experiments.campaign_design import AQS, S, match
from experiments.stage_tools import spec, write_specs, write_predictions

SIGMA0 = 39.0   # intrinsic (pristine zigzag) strength of the model, N/m
K = {"straight": 1.25, "necked": 1.8, "random": 3.0, "crack": None}


def minsf(family, params):
    a = D.generate(family, **params)
    d = compute_descriptors(a, a.info.get("design"))
    return d["min_solid_fraction_across_x"], a.info["porosity"], len(a)


def pred_HA(msf, kclass):
    return round(msf * SIGMA0 / K[kclass], 1)


def stage4():
    L, LH = 100, 120
    st, preds = [], []
    # --- pairs 1-2: square vs circle coarse mesh (H-A vs H-B)
    p_sq, f_sq, n = match("nanomesh_single", {"Lx": LH, "Ly": LH, "period": 40, "shape": "square", "stagger": False}, 0.20, "pore_d", 5, 36)
    side = p_sq["pore_d"]
    msf, phi, n = minsf("nanomesh_single", p_sq)
    st.append(S("S4_square_D40", "nanomesh_single", p_sq, reason="H-A vs H-B pair 1/2: coarse square pores (straight uniform ligaments), reference of the pair", hypothesis="H-A", tags=["stage4", "pairAB"], prediction={"strength_Nm": pred_HA(msf, "straight"), "fracture_mode_class": "abrupt", "basis": f"minSF={msf:.2f}, K=straight"}))
    p_ci = dict(p_sq); p_ci["shape"] = "circle"
    msf_c, phi_c, n = minsf("nanomesh_single", p_ci)
    st.append(S("S4_circle_D40_sameMinSF", "nanomesh_single", p_ci, reason=f"pair 1: circular pores with diameter = square side ({side:.1f} A) -> same minSF ({msf_c:.2f}) but lower porosity ({phi_c:.2f}); H-A predicts weaker than square (necked paths), H-B predicts >= square", hypothesis="H-A vs H-B", tags=["stage4", "pairAB"], prediction={"strength_Nm": pred_HA(msf_c, "necked"), "fracture_mode_class": "progressive", "basis": f"minSF={msf_c:.2f}, K=necked; H-B alternative ~{pred_HA(msf_c,'straight')}"}))
    p_c2, f_c2, n = match("nanomesh_single", {"Lx": LH, "Ly": LH, "period": 40, "shape": "circle", "stagger": False}, 0.20, "pore_d", 5, 36)
    msf_c2, phi_c2, n = minsf("nanomesh_single", p_c2)
    st.append(S("S4_circle_D40_samePhi", "nanomesh_single", p_c2, reason=f"pair 2: circular pores at the same porosity (0.20) -> minSF {msf_c2:.2f}", hypothesis="H-A vs H-B", tags=["stage4", "pairAB"], prediction={"strength_Nm": pred_HA(msf_c2, "necked"), "fracture_mode_class": "progressive", "basis": f"minSF={msf_c2:.2f}, K=necked"}))
    # --- pairs 3-4: ligament-width dispersion (H-C)
    p_sd = dict(p_sq); p_sd["size_disorder"] = 0.15; p_sd["seed"] = 1
    p_sd, f_sd, n = match("nanomesh_single", p_sd, 0.20, "pore_d", 5, 36)
    msf_sd, _, _ = minsf("nanomesh_single", p_sd)
    st.append(S("S4_square_D40_dispersed", "nanomesh_single", p_sd, seed=1, reason="pair 3 (H-C): square coarse mesh with +-15% pore-size dispersion (non-equivalent rows) at the same porosity", hypothesis="H-C", tags=["stage4", "pairC"], prediction={"strength_Nm": round(pred_HA(msf_sd, "straight") * 0.9, 1), "fracture_mode_class": "progressive", "basis": "weakest row ~10% below mean; H-C: multi-step; H-C': single avalanche"}))
    p_32 = {"Lx": LH, "Ly": LH, "period": 32, "size_disorder": 0.15, "seed": 1}
    p_32, f_32, n = match("nanomesh_single", p_32, 0.20, "pore_d", 6, 28)
    msf_32, _, _ = minsf("nanomesh_single", p_32)
    st.append(S("S4_p32_dispersed", "nanomesh_single", p_32, seed=1, reason="pair 4 (H-C): period-32 round-pore mesh with +-15% size dispersion", hypothesis="H-C", tags=["stage4", "pairC"], prediction={"strength_Nm": round(pred_HA(msf_32, "necked") * 0.95, 1), "fracture_mode_class": "progressive", "basis": "necked + non-equivalent"}))
    # --- H-E: flaw tolerance in larger cells
    for cl, cell in [(30, 150), (40, 150)]:
        st.append(S(f"S4_precrack_L{cl}_cell150", "precrack", {"Lx": cell, "Ly": cell, "crack_len": cl, "crack_w": 3.0, "angle_deg": 90}, reason=f"H-E: {cl} A crack in a 150 A cell (image interaction removed)", hypothesis="H-E", tags=["stage4", "flaw"], prediction={"strength_Nm": 18.0, "fracture_mode_class": "abrupt", "basis": "lattice-trapping plateau (H-E); Griffith alternative " + ("13.9" if cl == 30 else "12.0")}))
    st.append(S("S4_hole_d30", "ring_around_hole", {"Lx": L, "Ly": L, "hole_d": 30, "n_rings": 0}, reason="H-E: 30 A hole", hypothesis="H-E", tags=["stage4", "flaw"], prediction={"strength_Nm": 17.5, "fracture_mode_class": "abrupt", "basis": "plateau; net-section 0.7 x 39/1.6"}))
    # --- H-F: vein direction at matched porosity
    for vd in ["x", "y"]:
        p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 40, "vein_w": 8, "vein_dirs": vd}, 0.20, "pore_d", 3, 11)
        msf_v, _, _ = minsf("nanomesh_hier", p)
        st.append(S(f"S4_H2_D40_W8_veins_{vd}", "nanomesh_hier", p, reason=f"H-F: two-level mesh with veins only along {vd} ({'parallel' if vd=='x' else 'perpendicular'} to the load) at matched porosity; xy reference = S2_H2_D40_W8 (12.8 N/m)", hypothesis="H-F", tags=["stage4", "hierarchy"], prediction={"strength_Nm": 14.5 if vd == "x" else 11.5, "fracture_mode_class": "progressive", "basis": f"minSF={msf_v:.2f}; H-F: x-veins add straight load paths, y-veins add nothing (H-F' predicts xy > x)"}))
    batches = [spec("discovery", "stage4_discriminating", f"stage4_b{k}", st[3 * k:3 * k + 3], AQS) for k in range((len(st) + 2) // 3)]
    preds = [{"name": s_["name"], "family": s_["family"], "params": s_["params"], "predicted": s_["prediction"], "rationale": s_["reason"]} for s_ in st]
    return batches, preds


def stage5():
    L, LH = 100, 120
    st = []
    for W in [16, 20]:
        p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 40, "vein_w": W}, 0.20, "pore_d", 2, 11)
        st.append(S(f"S5_H2_D40_W{W}", "nanomesh_hier", p, reason=f"vein-width sweep (H-F/H-A rule of mixtures): W={W} A", hypothesis="H-F", tags=["stage5", "hierarchy"], notes=f"phi={f:.3f}"))
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 60, "vein_w": 12}, 0.20, "pore_d", 2, 11)
    st.append(S("S5_H2_D60_W12", "nanomesh_hier", p, reason="scale ratio x vein width interaction (D60, W12)", hypothesis="H-F", tags=["stage5", "hierarchy"], notes=f"phi={f:.3f}"))
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 8, "domain": 40, "vein_w": 8}, 0.20, "pore_d", 2, 7)
    st.append(S("S5_H2_p8_D40_W8", "nanomesh_hier", p, reason="finer fine level (period 8, scale ratio 5 at smaller absolute size)", hypothesis="H-F", tags=["stage5", "hierarchy"], notes=f"phi={f:.3f}"))
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 40, "vein_w": 8, "jitter": 2.0, "seed": 1}, 0.20, "pore_d", 3, 11)
    st.append(S("S5_H2_D40_W8_jitter2", "nanomesh_hier", p, seed=1, reason="hierarchy x disorder: fine level jittered 2 A", hypothesis="H-D/H-F", tags=["stage5", "hierarchy", "disorder"], notes=f"phi={f:.3f}"))
    p, f, n = match("nanomesh_hier", {"Lx": LH, "Ly": LH, "levels": 2, "period": 12, "domain": 40, "vein_w": 8, "aspect": 2.0, "angle_deg": 90}, 0.20, "pore_d", 3, 11)
    st.append(S("S5_H2_D40_W8_ellipseX", "nanomesh_hier", p, reason="hierarchy x anisotropy: fine pores elongated along the load (angle_deg=90 convention)", hypothesis="H-A/H-F", tags=["stage5", "hierarchy", "anisotropy"], notes=f"phi={f:.3f}"))
    for reg in [0.3, 0.8]:
        p, f, n = match("voronoi_network", {"Lx": L, "Ly": L, "n_cells": 25, "regularity": reg, "seed": 1}, 0.20, "ligament_w", 2, 20, increasing=False)
        st.append(S(f"S5_voronoi_n25_reg{reg}", "voronoi_network", p, seed=1, reason=f"Voronoi regularity sweep (H-D): regularity {reg}", hypothesis="H-D", tags=["stage5", "disorder"], notes=f"phi={f:.3f}"))
    for ph in [0.10, 0.30, 0.40]:
        p, f, n = match("slit_array", {"Lx": L, "Ly": L, "slit_w": 4.0, "period_x": 30, "period_y": 16, "angle_deg": 0}, ph, "slit_len", 2, 29)
        st.append(S(f"S5_slit0_phi{ph}", "slit_array", p, reason=f"porosity scaling of the load-parallel slit architecture (H-A: sigma = minSF x 31): phi={ph}", hypothesis="H-A", tags=["stage5", "porosity", "loadpath"], notes=f"phi={f:.3f}"))
    for ph in [0.10, 0.30]:
        p, f, n = match("nanomesh_single", {"Lx": LH, "Ly": LH, "period": 40, "shape": "square", "stagger": False}, ph, "pore_d", 3, 36)
        st.append(S(f"S5_square_D40_phi{ph}", "nanomesh_single", p, reason=f"porosity scaling of the coarse square mesh (H-A): phi={ph}", hypothesis="H-A", tags=["stage5", "porosity"], notes=f"phi={f:.3f}"))
    batches = [spec("discovery", "stage5_deep", f"stage5_b{k}", st[3 * k:3 * k + 3], AQS) for k in range((len(st) + 2) // 3)]
    return batches


def stage6():
    L = 100
    st = []
    for sd in [2, 3]:
        p, f, n = match("nanomesh_single", {"Lx": L, "Ly": L, "period": 16, "jitter": 2.0, "seed": sd}, 0.20, "pore_d", 3, 14)
        st.append(S(f"S6_mesh_jitter2.0_s{sd}", "nanomesh_single", p, seed=sd, reason="seed replicate of S2_mesh_jitter2.0", tags=["stage6", "disorder"], notes="mesh_jitter2.0"))
        p, f, n = match("nanomesh_single", {"Lx": L, "Ly": L, "period": 16, "size_disorder": 0.15, "seed": sd}, 0.20, "pore_d", 3, 14)
        st.append(S(f"S6_mesh_sizedis0.15_s{sd}", "nanomesh_single", p, seed=sd, reason="seed replicate of S2_mesh_sizedis0.15", tags=["stage6", "disorder"], notes="mesh_sizedis0.15"))
        for reg in [0.0, 0.6]:
            p, f, n = match("voronoi_network", {"Lx": L, "Ly": L, "n_cells": 25, "regularity": reg, "seed": sd}, 0.20, "ligament_w", 2, 20, increasing=False)
            st.append(S(f"S6_voronoi_n25_reg{reg}_s{sd}", "voronoi_network", p, seed=sd, reason=f"seed replicate of S2_voronoi_n25_reg{reg}", tags=["stage6", "disorder"], notes=f"voronoi_n25_reg{reg}"))
        st.append(S(f"S6_vac_random_2pct_s{sd}", "vacancies", {"Lx": L, "Ly": L, "fraction": 0.02, "organisation": "random"}, seed=sd, reason="seed replicate of S1_vac_random_2pct", tags=["stage6", "defects"], notes="vac_random_2pct"))
    # lattice-registry replicates of the ordered reference mesh (pore lattice shifted relative to the graphene lattice)
    for k, off in enumerate([(1.23, 0.0), (0.0, 2.13), (1.23, 2.13)]):
        p, f, n = match("nanomesh_single", {"Lx": L, "Ly": L, "period": 16, "offset": list(off)}, 0.20, "pore_d", 3, 14)
        st.append(S(f"S6_mesh_p16_offset{k+1}", "nanomesh_single", p, reason=f"lattice-registry replicate of the ordered p16 mesh: pore lattice shifted by {off} A", tags=["stage6", "registry"], notes="mesh_p16_registry"))
    batches = [spec("discovery", "stage6_seeds", f"stage6_b{k}", st[3 * k:3 * k + 3], AQS) for k in range((len(st) + 2) // 3)]
    return batches


if __name__ == "__main__":
    b4, preds4 = stage4()
    write_specs(os.path.join(ROOT, "experiments", "specs", "stage4"), b4)
    path, h = write_predictions("stage4_predictions", preds4, notes="Predictions for the Stage 4 discriminating experiments, written before launching them; H-A model sigma = minSF x 39/K with K(straight)=1.25, K(necked)=1.8.")
    print("stage4:", sum(len(b["structures"]) for b in b4), "structures; predictions", path, h[:12])
    b5 = stage5(); write_specs(os.path.join(ROOT, "experiments", "specs", "stage5"), b5); print("stage5:", sum(len(b["structures"]) for b in b5))
    b6 = stage6(); write_specs(os.path.join(ROOT, "experiments", "specs", "stage6"), b6); print("stage6:", sum(len(b["structures"]) for b in b6))
    for b in b4 + b5 + b6:
        for s_ in b["structures"]:
            a = D.generate(s_["family"], **s_["params"])
            print(f"  {s_['name']:30s} N={len(a):5d} phi={a.info['porosity']:.3f} pred={(s_.get('prediction') or {}).get('strength_Nm', '')}")
