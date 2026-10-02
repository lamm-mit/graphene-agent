"""Evaluation of the hierarchy follow-up (phase 3) against its pre-registered predictions.

Reads the completed runs of stage "hierarchy_followup" from this tree's database and the phase-1 references from
reference/phase1_runs, computes the test-specific metrics with the definitions fixed in the pre-registration
(analysis/assess_phase1.curve_metrics for the two-stage quantities), and writes

  results/hierarchy_followup_results.md        tables per test and the verdicts against the success criteria
  results/hierarchy_followup_results.json      the same numbers, machine-readable
  results/facts_phase3.json                    timestamps (prediction file, first and last run, evaluation), wall time, atoms
  experiments/holdouts/hierarchy_followup_evaluation.json   standard prediction evaluation (stage_tools.evaluate_predictions)
  figures/hierarchy/*.png|svg|pdf              stress-strain overlays per test, retention / W-ratio / residual charts
  figures/hierarchy/panels/                    event-selected fracture panels of the cracked runs (crack path), unless --no-panels

Can be run at any time while the queue is still working: missing runs are reported as pending.

    python analysis/hierarchy_followup_analysis.py [--no-panels]
"""
from __future__ import annotations
import sys, os, json, time, argparse, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import db
from experiments.stage_tools import evaluate_predictions
from experiments.campaign_paper_sweeps import alignment_index
from experiments.campaign_hierarchy_followup import STAGE, PRED_NAME
from atomistics.descriptors.columns import vein_bands, in_vein
from analysis.assess_phase1 import curve_metrics

RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures", "hierarchy")
PRED_FILE = os.path.join(ROOT, "experiments", "predictions", PRED_NAME + ".json")
PRED_FILE_T5 = os.path.join(ROOT, "experiments", "predictions", "hierarchy_T5_predictions.json")
plt.rcParams.update({"font.family": "Arial" if any("Arial" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "sans-serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def M(r, k):
    return (r.get("metrics") or {}).get(k, np.nan)


def load():
    own = {r["name"]: r for r in db.own_records() if r.get("stage") == STAGE and r.get("status") == "completed"}
    ref = {r["name"]: r for r in db.reference_records() if r.get("status") == "completed"}
    preds = {p["name"]: p for p in json.load(open(PRED_FILE))["predictions"]} if os.path.exists(PRED_FILE) else {}
    if os.path.exists(PRED_FILE_T5):
        preds.update({p["name"]: p for p in json.load(open(PRED_FILE_T5))["predictions"]})
    return own, ref, preds


# ---------------------------------------------------------------------------------------------------- crack path
def damage_events(rec):
    ev = rec.get("broken_bond_events") or []
    if not ev:
        return np.zeros(0), np.zeros((0, 2))
    return np.array([e[0] for e in ev], float), np.array([e[4][:2] for e in ev], float)


def crack_path(rec):
    """Where the damage went: extent of the broken-bond midpoints up to the peak and up to failure, first damage inside a
    load-parallel vein band (T1/T3 veins designs), damage inside seam rows and away from the crack plane (T2)."""
    strains, pos = damage_events(rec)
    d = rec.get("design") or {}; m = rec.get("metrics") or {}
    out = dict(n_events=int(len(strains)))
    if not len(strains):
        return out
    ep = m.get("strain_at_peak", np.nan)
    at_peak = strains <= ep + 1e-9
    out.update(damage_extent_y_at_peak=float(np.ptp(pos[at_peak, 1])) if at_peak.sum() > 1 else 0.0, damage_extent_x_at_peak=float(np.ptp(pos[at_peak, 0])) if at_peak.sum() > 1 else 0.0,
               damage_extent_y_final=float(np.ptp(pos[:, 1])) if len(pos) > 1 else 0.0, damage_extent_x_final=float(np.ptp(pos[:, 0])) if len(pos) > 1 else 0.0, n_events_at_peak=int(at_peak.sum()))
    bands = vein_bands(d)
    if bands:
        inv = np.array([in_vein(y, bands) for y in pos[:, 1]])
        out.update(n_events_in_veins_at_peak=int((inv & at_peak).sum()), n_events_in_veins_final=int(inv.sum()),
                   first_vein_damage_strain=float(strains[inv].min()) if inv.any() else None,
                   arrested_at_peak=bool(not (inv & at_peak).any()))
    if d.get("family") == "seam_pores" and d.get("cracked"):
        Lx, Ly = d["Lx"], d["Ly"]; xc, yc = d["crack_center"][0] * Lx, d["crack_center"][1] * Ly
        s = d["seam_spacing"]; n = d["n_rows"]
        if d["seam_dir"] == "x":
            rows = [((yc + j * s) % Ly) for j in range(n)]
            dy = np.array([min(abs(((y - r + Ly / 2) % Ly) - Ly / 2) for r in rows) for y in pos[:, 1]])
            dx = np.abs(((pos[:, 0] - xc + Lx / 2) % Lx) - Lx / 2)
            in_seam = (dy < d["pore_d"] / 2 + 1.5) & (dx > 4.0)
            out.update(events_in_seams_fraction=float(in_seam.mean()), events_in_seams_at_peak=int((in_seam & at_peak).sum()),
                       deflected=bool(in_seam.mean() > 0.25 or out["damage_extent_x_final"] > 20.0))
        else:
            out.update(events_in_seams_fraction=None, deflected=bool(out["damage_extent_x_final"] > 20.0))
    return out


# ---------------------------------------------------------------------------------------------------- tests
def fmt(v, f="{:.2f}"):
    try:
        return "pending" if v is None else ("—" if isinstance(v, float) and not np.isfinite(v) else f.format(v))
    except Exception:
        return str(v)


def rows_T1(own, ref, preds):
    rows = []
    for key in ("pristine", "ellipse", "veinsX", "slit"):
        for reg in (0, 1):
            unc = own.get(f"H1_{key}_r{reg}") if key != "pristine" else ref["S1_pristine_zz"]
            for cl in (20, 40):
                name = f"H1_{key}_L{cl}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {})
                row = dict(name=name, key=key, registry=reg, crack=cl, sigma=M(r, "strength_Nm") if r else None, sigma_uncracked=M(unc, "strength_Nm") if unc else None,
                           W=M(r, "work_to_failure_J_m2") if r else None, W_uncracked=M(unc, "work_to_failure_J_m2") if unc else None,
                           retention_rule=p.get("retention_rule"), retention_mechanism=p.get("retention_mechanism"), sigma_rule=p.get("strength_Nm_rule"), sigma_mech=p.get("strength_Nm_mechanism"),
                           first_damage_strain=M(r, "first_damage_strain") if r else None, mode=(M(r, "fracture_mode") if r else None))
                if r and unc:
                    row["retention"] = row["sigma"] / row["sigma_uncracked"]; row["W_ratio"] = row["W"] / row["W_uncracked"]
                    row["retention_over_rule"] = row["retention"] / row["retention_rule"] if row.get("retention_rule") else None
                    row.update(crack_path(r))
                rows.append(row)
    return rows


def verdict_T1(rows):
    v = {}
    for reg in (0, 1):
        for cl in (20, 40):
            vx = next(x for x in rows if x["key"] == "veinsX" and x["registry"] == reg and x["crack"] == cl)
            el = next(x for x in rows if x["key"] == "ellipse" and x["registry"] == reg and x["crack"] == cl)
            if vx.get("retention") is None or el.get("retention") is None:
                v[f"L{cl}_r{reg}"] = "pending"; continue
            ok_rule = vx["retention"] >= 1.15 * vx["retention_rule"]; ok_ctrl = vx["retention"] >= 1.10 * el["retention"]
            v[f"L{cl}_r{reg}"] = dict(retention=vx["retention"], rule=vx["retention_rule"], over_rule=vx["retention"] / vx["retention_rule"], ellipse_retention=el["retention"],
                                       arrested_at_peak=vx.get("arrested_at_peak"), passes=bool(ok_rule and ok_ctrl))
    done = [x for x in v.values() if x != "pending"]
    v["overall"] = "pending" if len(done) < 4 else ("PASS" if all(x["passes"] for x in done) else "FAIL")
    return v


def rows_T4(own, ref, preds):
    rows = []
    ctrl_names = ["H1_ellipse_r0", "H1_ellipse_r1"]
    ctrls = [own[n] for n in ctrl_names if n in own] or [ref["S2_ellipse_90deg"]]
    ctrl_E = float(np.mean([curve_metrics(c)["post_peak_energy_J_m2"] for c in ctrls])); ctrl_sig = float(np.mean([M(c, "strength_Nm") for c in ctrls]))
    for W in (16, 20):
        for reg in (0, 1):
            name = f"H4_veinsX_W{W}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {})
            row = dict(name=name, W_vein=W, registry=reg, sigma=M(r, "strength_Nm") if r else None, sigma_rule=p.get("strength_Nm_rule"), veins_alone_rule=p.get("strength_Nm_veins_alone"),
                       residual_rule=p.get("residual_fraction_rule"), control_post_peak_energy=ctrl_E, control_sigma=ctrl_sig, control=", ".join(c["name"] for c in ctrls), mode=(M(r, "fracture_mode") if r else None))
            if r:
                cm = curve_metrics(r); row.update({k: cm[k] for k in ("residual_after_first_avalanche", "second_peak_fraction", "post_peak_energy_J_m2", "post_avalanche_energy_J_m2", "post_avalanche_strain_range", "post_peak_strain_range", "first_avalanche_strain", "first_avalanche_bonds", "truncated")})
                row["W"] = M(r, "work_to_failure_J_m2"); row.update(crack_path(r))
                row["passes"] = bool(cm["residual_after_first_avalanche"] is not None and cm["residual_after_first_avalanche"] >= 0.6 and cm["post_peak_energy_J_m2"] >= 1.5 * ctrl_E)
            rows.append(row)
    # phase-1 comparison points
    for n in ("S4_H2_D40_W8_veins_x", "S5_H2_D40_W16", "S5_H2_D40_W20", "S7_H2_veinsX_W12_ellipseX"):
        r = ref[n]; cm = curve_metrics(r)
        rows.append(dict(name=n + " (phase 1)", W_vein=r["params"].get("vein_w"), registry=0, sigma=M(r, "strength_Nm"), residual_after_first_avalanche=cm["residual_after_first_avalanche"], second_peak_fraction=cm["second_peak_fraction"],
                         post_peak_energy_J_m2=cm["post_peak_energy_J_m2"], post_avalanche_strain_range=cm["post_avalanche_strain_range"], W=M(r, "work_to_failure_J_m2"), mode=M(r, "fracture_mode"), control_post_peak_energy=ctrl_E))
    return rows


def verdict_T4(rows):
    new = [x for x in rows if "(phase 1)" not in x["name"]]
    if any("passes" not in x for x in new):
        return dict(overall="pending", done=[x["name"] for x in new if "passes" in x])
    byW = {W: all(x["passes"] for x in new if x["W_vein"] == W) for W in (16, 20)}
    return dict(overall="PASS" if any(byW.values()) else "FAIL", per_width=byW)


def rows_T2(own, ref, preds):
    rows = []
    for key, sp in (("seamX", 20), ("seamX", 30), ("seamX", 40), ("weakseamX", 20), ("seamY", 20), ("random", 20)):
        for reg in (0, 1):
            name = f"H2_{key}_s{sp}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {}); c = own.get(f"H1_pristine_L20_r{reg}")
            row = dict(name=name, key=key, spacing=sp, registry=reg, sigma=M(r, "strength_Nm") if r else None, W=M(r, "work_to_failure_J_m2") if r else None,
                       sigma_ref=M(c, "strength_Nm") if c else None, W_ref=M(c, "work_to_failure_J_m2") if c else None, W_ratio_rule=p.get("W_ratio_rule"), W_ratio_hypothesis=p.get("W_ratio_hypothesis"),
                       sigma_rule=p.get("strength_Nm_rule"), deflection_expected=p.get("deflection_expected"), mode=(M(r, "fracture_mode") if r else None), first_damage_strain=M(r, "first_damage_strain") if r else None)
            if r and c:
                row["W_ratio"] = row["W"] / row["W_ref"]; row["sigma_ratio"] = row["sigma"] / row["sigma_ref"]; row.update(crack_path(r))
            rows.append(row)
    return rows


def verdict_T2(rows):
    seams = [x for x in rows if x["key"] in ("seamX", "weakseamX")]; ctrl = [x for x in rows if x["key"] in ("seamY", "random")]
    if any("W_ratio" not in x for x in rows):
        return dict(overall="pending", done=[x["name"] for x in rows if "W_ratio" in x])
    best = {}
    for key, sp in {(x["key"], x["spacing"]) for x in seams}:
        pair = [x for x in seams if x["key"] == key and x["spacing"] == sp]
        best[f"{key}_s{sp}"] = dict(W_ratios=[x["W_ratio"] for x in pair], deflected=[x.get("deflected") for x in pair], passes=bool(all(x["W_ratio"] >= 1.3 and x.get("deflected") for x in pair)))
    controls_ok = all(x["W_ratio"] < 1.1 for x in ctrl)
    return dict(overall="PASS" if any(v["passes"] for v in best.values()) and controls_ok else "FAIL", seams=best, controls_below_1p1=controls_ok, control_W_ratios={x["name"]: x["W_ratio"] for x in ctrl})


def rows_T3(own, ref, preds, R):
    fit = R["alignment"]["fit_one_level"]; scatter = R["alignment"]["one_level_resid_std"]; Wsc = R["alignment"]["one_level_W_std"]
    rows = []
    for reg in (0, 1):
        for key in ("2L", "1L_fine", "1L_coarse"):
            name = f"H3_{key}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {})
            row = dict(name=name, key=key, registry=reg, cracked=False, sigma=M(r, "strength_Nm") if r else None, W=M(r, "work_to_failure_J_m2") if r else None, sigma_rule=p.get("strength_Nm_rule"), sigma_trend=p.get("strength_Nm_trend"),
                       A=p.get("alignment_A"), mode=(M(r, "fracture_mode") if r else None))
            if r:
                A = alignment_index(r["descriptors"]); row["A_measured"] = A; row["premium_sigma"] = row["sigma"] - (fit[0] * A + fit[1]); row["scatter_sigma"] = scatter; row["scatter_W"] = Wsc
                cm = curve_metrics(r); row.update({k: cm[k] for k in ("residual_after_first_avalanche", "post_peak_energy_J_m2", "post_peak_strain_range")})
            rows.append(row)
        for key, cl in (("2L", 60), ("1L_fine", 60), ("1L_coarse", 60)):
            name = f"H3_{key}_L{cl}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {}); unc = own.get(f"H3_{key}_r{reg}")
            row = dict(name=name, key=key, registry=reg, cracked=True, crack=cl, sigma=M(r, "strength_Nm") if r else None, W=M(r, "work_to_failure_J_m2") if r else None, sigma_uncracked=M(unc, "strength_Nm") if unc else None,
                       retention_rule=p.get("retention_rule"), retention_mechanism=p.get("retention_mechanism"), sigma_rule=p.get("strength_Nm_rule"), mode=(M(r, "fracture_mode") if r else None))
            if r and unc:
                row["retention"] = row["sigma"] / row["sigma_uncracked"]; row["W_ratio"] = row["W"] / M(unc, "work_to_failure_J_m2"); row.update(crack_path(r))
            rows.append(row)
    # W premium: two-level minus the one-level fine design of the same registry (same cell, same pore shape) and minus the coarse one
    for reg in (0, 1):
        two = next(x for x in rows if x["name"] == f"H3_2L_r{reg}"); fine = next(x for x in rows if x["name"] == f"H3_1L_fine_r{reg}"); coarse = next(x for x in rows if x["name"] == f"H3_1L_coarse_r{reg}")
        if two.get("W") is not None and fine.get("W") is not None and coarse.get("W") is not None:
            two["W_premium_vs_fine"] = two["W"] - fine["W"]; two["W_premium_vs_coarse"] = two["W"] - coarse["W"]
            two["sigma_premium_vs_fine"] = two["sigma"] - fine["sigma"]; two["sigma_premium_vs_coarse"] = two["sigma"] - coarse["sigma"]   # same cell, same mass (the phase-1 trend does not transfer to 300 A)
        two_c = next(x for x in rows if x["name"] == f"H3_2L_L60_r{reg}"); fine_c = next(x for x in rows if x["name"] == f"H3_1L_fine_L60_r{reg}")
        if two_c.get("retention") is not None and fine_c.get("retention") is not None:
            two_c["retention_premium_vs_fine"] = two_c["retention"] - fine_c["retention"]; two_c["retention_over_rule"] = two_c["retention"] / two_c["retention_rule"]
    return rows


def verdict_T3(rows, R):
    Wsc = R["alignment"]["one_level_W_std"]; ssc = R["alignment"]["one_level_resid_std"]
    out = {}
    for reg in (0, 1):
        two = next(x for x in rows if x["name"] == f"H3_2L_r{reg}"); two_c = next(x for x in rows if x["name"] == f"H3_2L_L60_r{reg}")
        if "W_premium_vs_fine" not in two or "retention_premium_vs_fine" not in two_c:
            out[f"r{reg}"] = "pending"; continue
        out[f"r{reg}"] = dict(premium_sigma_vs_phase1_trend=two["premium_sigma"], sigma_premium_vs_fine=two["sigma_premium_vs_fine"], sigma_premium_vs_coarse=two["sigma_premium_vs_coarse"],
                              W_premium_vs_fine=two["W_premium_vs_fine"], W_premium_vs_coarse=two["W_premium_vs_coarse"], retention_premium_vs_fine=two_c["retention_premium_vs_fine"],
                              W_beyond_scatter=bool(min(two["W_premium_vs_fine"], two["W_premium_vs_coarse"]) > Wsc), retention_beyond_rule=bool(two_c["retention_over_rule"] >= 1.15 and two_c["retention_premium_vs_fine"] > 0.1),
                              strength_beyond_scatter_same_cell=bool(min(two["sigma_premium_vs_fine"], two["sigma_premium_vs_coarse"]) > ssc))
    done = [v for v in out.values() if v != "pending"]
    if len(done) < 1:
        out["overall"] = "pending"
    else:
        signs_W = [v["W_premium_vs_fine"] > Wsc for v in done]; signs_R = [v["retention_beyond_rule"] for v in done]
        out["overall"] = ("PASS (W)" if all(signs_W) else "PASS (crack retention)" if all(signs_R) else "FAIL") if len(done) == 2 else ("provisional PASS" if (signs_W[0] or signs_R[0]) else "provisional FAIL")
    return out


def rows_T5(own, ref, preds):
    """Composite hierarchies: uncracked (two registries) with two-stage metrics and rules; cracked with retention against the rule and the reference."""
    keys = ["VF_E", "VF_R", "VE_R", "VR_E", "VS_R", "GR_0", "GFx_0", "V3_E", "GS"]
    ref_unc = {"VF_E": "H3_2L_r0", "VF_R": "H5_VS_R_r0", "VE_R": "H5_VS_R_r0", "VR_E": "H3_2L_r0", "VS_R": "H3_1L_fine_r0", "GR_0": "H5_GS_r0", "GFx_0": "H5_GS_r0", "V3_E": "H3_2L_r0", "GS": "H3_1L_coarse_r0"}
    ref_cr = {"VF_E": "H3_2L_L60_r0", "VF_R": "H5_VS_R_L60_r0", "VE_R": "H5_VS_R_L60_r0", "VR_E": "H3_1L_fine_L60_r0", "VS_R": "H3_2L_L60_r0", "GR_0": "H5_GS_L40_r0", "GFx_0": "H5_GS_L40_r0", "V3_E": "H3_2L_L60_r0", "GS": None}
    rows = []
    for key in keys:
        for reg in (0, 1):
            name = f"H5_{key}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {}); c = own.get(ref_unc[key])
            row = dict(name=name, archetype=key, registry=reg, cracked=False, sigma=M(r, "strength_Nm") if r else None, sigma_rule=p.get("strength_Nm_rule"), sigma_rule_b=p.get("strength_Nm_rule_b"),
                       W=M(r, "work_to_failure_J_m2") if r else None, reference=ref_unc[key], sigma_ref=M(c, "strength_Nm") if c else None, mode=(M(r, "fracture_mode") if r else None))
            if r:
                cm = curve_metrics(r); row.update({k: cm[k] for k in ("residual_after_first_avalanche", "second_peak_fraction", "post_peak_energy_J_m2", "post_peak_strain_range", "post_avalanche_strain_range", "truncated")})
                if c:
                    row["ref_post_peak_energy"] = curve_metrics(c)["post_peak_energy_J_m2"]; row["post_peak_energy_ratio"] = cm["post_peak_energy_J_m2"] / max(row["ref_post_peak_energy"], 1e-9)
            rows.append(row)
        for reg in ((0, 1) if key in ("VF_E", "GR_0") else (0,)):
            cl = 40 if key.startswith("G") else 60
            name = f"H5_{key}_L{cl}_r{reg}"; r = own.get(name); p = preds.get(name, {}).get("predicted", {}); unc = own.get(f"H5_{key}_r{reg}"); c = own.get(ref_cr[key]) if ref_cr[key] else None
            cu = own.get(ref_cr[key].replace(f"_L{cl}", "")) if ref_cr[key] else None
            row = dict(name=name, archetype=key, registry=reg, cracked=True, crack=cl, sigma=M(r, "strength_Nm") if r else None, sigma_uncracked=M(unc, "strength_Nm") if unc else None, sigma_rule=p.get("strength_Nm_rule"),
                       retention_rule=p.get("retention_rule"), retention_mechanism=p.get("retention_mechanism"), reference=ref_cr[key], W=M(r, "work_to_failure_J_m2") if r else None, mode=(M(r, "fracture_mode") if r else None))
            if r and unc:
                row["retention"] = row["sigma"] / row["sigma_uncracked"]; row["retention_over_rule"] = row["retention"] / row["retention_rule"] if row.get("retention_rule") else None
                row["W_ratio"] = row["W"] / M(unc, "work_to_failure_J_m2"); row.update(crack_path(r))
                if c and cu:
                    row["ref_retention"] = M(c, "strength_Nm") / M(cu, "strength_Nm")
            rows.append(row)
    return rows


def verdict_T5(rows):
    out = {}
    def get(n): return next((x for x in rows if x["name"] == n), None)
    for key in ("VF_E", "VF_R"):
        u = [get(f"H5_{key}_r{r}") for r in (0, 1)]; c = get(f"H5_{key}_L60_r0")
        if any(x is None or x.get("residual_after_first_avalanche") is None and x.get("sigma") is None for x in u) or c is None or c.get("retention") is None or any("post_peak_energy_ratio" not in x for x in u):
            out[key] = "pending"; continue
        out[key] = dict(residuals=[x["residual_after_first_avalanche"] for x in u], energy_ratios=[x["post_peak_energy_ratio"] for x in u], retention=c["retention"], over_rule=c.get("retention_over_rule"), ref_retention=c.get("ref_retention"),
                        passes=bool(all((x["residual_after_first_avalanche"] or 0) >= 0.6 and x["post_peak_energy_ratio"] >= 1.5 for x in u) and (c.get("retention_over_rule") or 0) >= 1.15 and c["retention"] >= (c.get("ref_retention") or 9)))
    for key in ("VE_R", "VR_E"):
        c = get(f"H5_{key}_L60_r0")
        out[key] = "pending" if c is None or c.get("retention") is None or c.get("ref_retention") is None else dict(retention=c["retention"], ref_retention=c["ref_retention"], no_advantage=bool(c["retention"] <= c["ref_retention"] + 0.02))
    for key in ("GR_0", "GFx_0"):
        u = get(f"H5_{key}_r0"); c = get(f"H5_{key}_L40_r0")
        if u is None or u.get("residual_after_first_avalanche") is None and u.get("sigma") is None or c is None or c.get("retention") is None:
            out[key] = "pending"; continue
        out[key] = dict(residual=u["residual_after_first_avalanche"], strain_range=u["post_peak_strain_range"], retention=c["retention"], over_rule=c.get("retention_over_rule"), ref_retention=c.get("ref_retention"),
                        passes=bool((u["residual_after_first_avalanche"] or 0) >= 0.6 and u["post_peak_strain_range"] >= 0.03 and (c.get("retention_over_rule") or 0) >= 1.15 and c["retention"] > (c.get("ref_retention") or 9)))
    u = get("H5_V3_E_r0"); c = get("H5_V3_E_L60_r0")
    out["V3_E"] = "pending" if u is None or u.get("sigma") is None or c is None or c.get("retention") is None or c.get("ref_retention") is None else dict(sigma=u["sigma"], sigma_ref=u.get("sigma_ref"), retention=c["retention"], ref_retention=c["ref_retention"], passes=bool(c["retention"] >= 1.10 * c["ref_retention"]))
    done = [v for v in out.values() if v != "pending"]
    out["overall"] = "pending" if not done else ("PASS" if any(isinstance(v, dict) and v.get("passes") for v in done) else "FAIL" if len(done) == len(out) - 1 else "provisional FAIL")
    return out


# ---------------------------------------------------------------------------------------------------- figures
def fig_curves(own, ref, groups, name, title):
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 2.5), squeeze=False)
    for ax, (gt, names) in zip(axes[0], groups.items()):
        for n in names:
            r = own.get(n) or ref.get(n)
            if not r:
                continue
            c = db.stress_strain(r); ax.plot(c["eps_x"], c["sigma_xx_Nm"], lw=1.0, label=n + (" (phase 1)" if n in ref and n not in own else ""))
        ax.set_title(gt, fontsize=8); ax.set_xlabel("engineering strain"); ax.grid(alpha=0.3); ax.legend(fontsize=5.5)
    axes[0][0].set_ylabel("2D stress (N/m)")
    fig.suptitle(title, fontsize=8)
    save(fig, name)


def fig_bars(rows, xkey, ykeys, name, title, ylabel):
    fig, ax = plt.subplots(figsize=(0.5 * len(rows) + 1.5, 2.6))
    x = np.arange(len(rows)); w = 0.8 / len(ykeys)
    for k, (yk, lab) in enumerate(ykeys):
        ax.bar(x + k * w, [(r.get(yk) if r.get(yk) is not None else np.nan) for r in rows], w, label=lab)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([r[xkey] for r in rows], rotation=70, fontsize=6); ax.set_ylabel(ylabel); ax.grid(alpha=0.3, axis="y"); ax.legend(); ax.set_title(title, fontsize=8)
    save(fig, name)


def panels(own, no_panels):
    if no_panels:
        return []
    from analysis import fracture_viz
    out_dir = os.path.join(FIG, "panels"); os.makedirs(out_dir, exist_ok=True); done = []
    for n, r in sorted(own.items()):
        if (r.get("design") or {}).get("cracked") and db.trajectory_path(r) and not os.path.exists(os.path.join(out_dir, f"progression_{n}_energy_rel.png")):
            try:
                fracture_viz.progression_panel(r["run_id"], out_dir); done.append(n)
            except Exception as e:
                print("panel failed", n, e)
    return done


# ---------------------------------------------------------------------------------------------------- report
def table(rows, cols):
    L = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(fmt(r.get(k), f) if not isinstance(r.get(k), str) else r.get(k) for _, (k, f) in [(c, v) for c, v in cols]) + " |")
    return "\n".join(L)


def main(no_panels=False):
    own, ref, preds = load(); R = json.load(open(os.path.join(ROOT, "reference", "phase1_reference_numbers.json")))
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    t1 = rows_T1(own, ref, preds); v1 = verdict_T1(t1)
    t4 = rows_T4(own, ref, preds); v4 = verdict_T4(t4)
    t2 = rows_T2(own, ref, preds); v2 = verdict_T2(t2)
    t3 = rows_T3(own, ref, preds, R); v3 = verdict_T3(t3, R)
    t5 = rows_T5(own, ref, preds); v5 = verdict_T5(t5)
    ev = evaluate_predictions(PRED_FILE, STAGE) if os.path.exists(PRED_FILE) else None
    if ev:
        json.dump(ev, open(os.path.join(ROOT, "experiments", "holdouts", "hierarchy_followup_evaluation.json"), "w"), indent=1, default=db._json_default)
    if os.path.exists(PRED_FILE_T5):
        json.dump(evaluate_predictions(PRED_FILE_T5, STAGE), open(os.path.join(ROOT, "experiments", "holdouts", "hierarchy_T5_evaluation.json"), "w"), indent=1, default=db._json_default)
    # facts (timeline of the pre-registration)
    runs = sorted(own.values(), key=lambda r: r["run_id"])
    def ts(rid):
        return datetime.datetime.strptime(rid.split("_")[1] + rid.split("_")[2], "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
    facts = dict(n=len(runs), prediction_written=json.load(open(PRED_FILE))["written"] if os.path.exists(PRED_FILE) else None, prediction_sha256=json.load(open(PRED_FILE)).get("sha256_of_content") if os.path.exists(PRED_FILE) else None,
                 first=ts(runs[0]["run_id"]) if runs else None, last=ts(runs[-1]["run_id"]) if runs else None, evaluated=time.strftime("%Y-%m-%d %H:%M"),
                 wall_h=float(sum((r.get("wall_time_s") or 0) for r in runs) / 3600), atoms=int(sum(r.get("n_atoms", 0) for r in runs)), n_expected=len(preds),
                 prediction_written_T5=json.load(open(PRED_FILE_T5))["written"] if os.path.exists(PRED_FILE_T5) else None, prediction_sha256_T5=json.load(open(PRED_FILE_T5)).get("sha256_of_content") if os.path.exists(PRED_FILE_T5) else None)
    json.dump(facts, open(os.path.join(RES, "facts_phase3.json"), "w"), indent=1)
    # figures
    fig_curves(own, ref, {"pristine + crack": ["H1_pristine_L20_r0", "H1_pristine_L40_r0", "S1_precrack_L20"], "one-level ellipse mesh": ["H1_ellipse_r0", "H1_ellipse_L20_r0", "H1_ellipse_L40_r0"],
                          "two-level veins mesh": ["H1_veinsX_r0", "H1_veinsX_L20_r0", "H1_veinsX_L40_r0"], "slit array": ["H1_slit_r0", "H1_slit_L20_r0", "H1_slit_L40_r0"]}, "T1_curves", "T1 crack arrest: registry 0 curves")
    fig_curves(own, ref, {"W16": ["H4_veinsX_W16_r0", "H4_veinsX_W16_r1", "S5_H2_D40_W16"], "W20": ["H4_veinsX_W20_r0", "H4_veinsX_W20_r1", "S5_H2_D40_W20"], "controls": ["H1_ellipse_r0", "H1_slit_r0", "S4_H2_D40_W8_veins_x"]}, "T4_curves", "T4 two-stage failure")
    fig_curves(own, ref, {"seams along the load": ["H2_seamX_s20_r0", "H2_seamX_s30_r0", "H2_seamX_s40_r0", "H2_weakseamX_s20_r0", "H1_pristine_L20_r0"], "controls": ["H2_seamY_s20_r0", "H2_random_s20_r0", "H1_pristine_L20_r0"]}, "T2_curves", "T2 Cook-Gordon seams (registry 0)")
    fig_curves(own, ref, {"uncracked": ["H3_2L_r0", "H3_1L_fine_r0", "H3_1L_coarse_r0"], "cracked": ["H3_2L_L60_r0", "H3_1L_fine_L60_r0", "H3_1L_coarse_L60_r0"]}, "T3_curves", "T3 scale separation (300 A)")
    fig_bars([x for x in t1 if x["key"] != "pristine"], "name", [("retention", "observed"), ("retention_rule", "net-section rule"), ("retention_mechanism", "mechanism")], "T1_retention", "T1 notch retention", "sigma_cracked / sigma_uncracked")
    fig_bars([x for x in t4 if "(phase 1)" not in x["name"]] + [x for x in t4 if "(phase 1)" in x["name"]], "name", [("residual_after_first_avalanche", "residual after 1st avalanche"), ("post_peak_energy_J_m2", "post-peak energy (J/m2)")], "T4_residual", "T4 two-stage metrics", "")
    fig_bars(t2, "name", [("W_ratio", "W ratio observed"), ("W_ratio_rule", "rule"), ("W_ratio_hypothesis", "hypothesis")], "T2_Wratio", "T2 work-to-failure ratio to the cracked pristine sheet", "W / W_ref")
    fig_curves(own, ref, {"fibrous veins": ["H5_VF_E_r0", "H5_VF_R_r0", "H3_2L_r0", "H5_VS_R_r0"], "porous veins": ["H5_VE_R_r0", "H5_VR_E_r0", "H3_2L_r0"], "grids": ["H5_GR_0_r0", "H5_GFx_0_r0", "H5_GS_r0", "H3_1L_coarse_r0"], "three levels": ["H5_V3_E_r0", "H3_2L_r0"]}, "T5_curves", "T5 composite hierarchies (registry 0) with their references")
    fig_curves(own, ref, {"fibrous veins, cracked": ["H5_VF_E_L60_r0", "H5_VF_R_L60_r0", "H3_2L_L60_r0", "H5_VS_R_L60_r0"], "grids, cracked": ["H5_GR_0_L40_r0", "H5_GFx_0_L40_r0", "H5_GS_L40_r0"], "three levels, cracked": ["H5_V3_E_L60_r0", "H3_2L_L60_r0"]}, "T5_curves_cracked", "T5 cracked composites with their references")
    fig_bars([x for x in t5 if not x["cracked"]], "name", [("sigma", "strength (N/m)"), ("sigma_rule", "rule"), ("sigma_rule_b", "rule b")], "T5_strength", "T5 strength against the column class rules", "N/m")
    fig_bars([x for x in t5 if x["cracked"]], "name", [("retention", "observed"), ("retention_rule", "net-section rule"), ("retention_mechanism", "mechanism"), ("ref_retention", "reference design")], "T5_retention", "T5 notch retention", "sigma_cracked / sigma_uncracked")
    pan = panels(own, no_panels)
    # report
    L = [f"# Hierarchy follow-up: results against the pre-registered predictions\n", f"Evaluated {facts['evaluated']}; {facts['n']} of {facts['n_expected']} runs completed; predictions written {facts['prediction_written']} (SHA-256 {str(facts['prediction_sha256'])[:16]}…); "
         f"runs {facts['first']} → {facts['last']}; {facts['wall_h']:.1f} process-hours.\n"]
    L.append("## T1 crack arrest by veins\n"); L.append(f"Verdict: **{v1['overall']}** — " + json.dumps({k: v for k, v in v1.items() if k != 'overall'}, default=str) + "\n")
    L.append(table(t1, [("design", ("name", "{}")), ("σ cracked", ("sigma", "{:.1f}")), ("σ uncracked", ("sigma_uncracked", "{:.1f}")), ("retention", ("retention", "{:.3f}")), ("rule", ("retention_rule", "{:.3f}")), ("mechanism", ("retention_mechanism", "{:.3f}")),
                        ("obs/rule", ("retention_over_rule", "{:.2f}")), ("W ratio", ("W_ratio", "{:.2f}")), ("first damage ε", ("first_damage_strain", "{:.3f}")), ("arrested at peak", ("arrested_at_peak", "{}")), ("first vein damage ε", ("first_vein_damage_strain", "{:.3f}")), ("mode", ("mode", "{}"))]))
    L.append("\n## T4 designed two-stage failure\n"); L.append(f"Verdict: **{v4['overall']}** — {json.dumps({k: v for k, v in v4.items() if k != 'overall'}, default=str)}\n")
    L.append(table(t4, [("design", ("name", "{}")), ("σ", ("sigma", "{:.1f}")), ("σ rule", ("sigma_rule", "{:.1f}")), ("veins alone (rule)", ("veins_alone_rule", "{:.1f}")), ("residual after 1st avalanche", ("residual_after_first_avalanche", "{:.2f}")), ("rule", ("residual_rule", "{:.2f}")),
                        ("second peak", ("second_peak_fraction", "{:.2f}")), ("post-peak energy", ("post_peak_energy_J_m2", "{:.3f}")), ("control", ("control_post_peak_energy", "{:.3f}")), ("post-avalanche strain range", ("post_avalanche_strain_range", "{:.3f}")), ("W", ("W", "{:.2f}")), ("truncated", ("truncated", "{}")), ("mode", ("mode", "{}")), ("passes", ("passes", "{}"))]))
    L.append("\n## T2 Cook–Gordon seams\n"); L.append(f"Verdict: **{v2['overall']}** — {json.dumps({k: v for k, v in v2.items() if k != 'overall'}, default=str)}\n")
    L.append(table(t2, [("design", ("name", "{}")), ("σ", ("sigma", "{:.1f}")), ("σ ref", ("sigma_ref", "{:.1f}")), ("σ rule", ("sigma_rule", "{:.1f}")), ("σ ratio", ("sigma_ratio", "{:.2f}")), ("W", ("W", "{:.2f}")), ("W ref", ("W_ref", "{:.2f}")), ("W ratio", ("W_ratio", "{:.2f}")), ("rule", ("W_ratio_rule", "{:.2f}")), ("hypothesis", ("W_ratio_hypothesis", "{:.2f}")),
                        ("deflected", ("deflected", "{}")), ("events in seams", ("events_in_seams_fraction", "{:.2f}")), ("damage extent x (Å)", ("damage_extent_x_final", "{:.1f}")), ("mode", ("mode", "{}"))]))
    L.append("\n## T3 scale separation (300 Å)\n"); L.append(f"Verdict: **{v3['overall']}** — {json.dumps({k: v for k, v in v3.items() if k != 'overall'}, default=str)}\n")
    L.append(table(t3, [("design", ("name", "{}")), ("σ", ("sigma", "{:.1f}")), ("σ rule", ("sigma_rule", "{:.1f}")), ("σ trend(A)", ("sigma_trend", "{:.1f}")), ("A", ("A_measured", "{:+.2f}")), ("σ − phase-1 trend", ("premium_sigma", "{:+.1f}")), ("σ − fine", ("sigma_premium_vs_fine", "{:+.1f}")), ("σ − coarse", ("sigma_premium_vs_coarse", "{:+.1f}")), ("W", ("W", "{:.2f}")), ("W − fine", ("W_premium_vs_fine", "{:+.2f}")), ("W − coarse", ("W_premium_vs_coarse", "{:+.2f}")),
                        ("retention", ("retention", "{:.3f}")), ("rule", ("retention_rule", "{:.3f}")), ("obs/rule", ("retention_over_rule", "{:.2f}")), ("retention − fine", ("retention_premium_vs_fine", "{:+.3f}")), ("arrested at peak", ("arrested_at_peak", "{}")), ("mode", ("mode", "{}"))]))
    L.append("\n## T5 composite hierarchies (300 Å)\n"); L.append(f"Verdict: **{v5['overall']}** — {json.dumps({k: v for k, v in v5.items() if k != 'overall'}, default=str)}\n")
    L.append(table([x for x in t5 if not x["cracked"]], [("design", ("name", "{}")), ("σ", ("sigma", "{:.1f}")), ("σ rule", ("sigma_rule", "{:.1f}")), ("σ rule b", ("sigma_rule_b", "{:.1f}")), ("reference", ("reference", "{}")), ("σ ref", ("sigma_ref", "{:.1f}")), ("W", ("W", "{:.2f}")),
                        ("residual after 1st avalanche", ("residual_after_first_avalanche", "{:.2f}")), ("second peak", ("second_peak_fraction", "{:.2f}")), ("post-peak energy", ("post_peak_energy_J_m2", "{:.3f}")), ("/ reference", ("post_peak_energy_ratio", "{:.2f}")), ("post-peak strain range", ("post_peak_strain_range", "{:.3f}")), ("mode", ("mode", "{}"))]))
    L.append("")
    L.append(table([x for x in t5 if x["cracked"]], [("design", ("name", "{}")), ("σ cracked", ("sigma", "{:.1f}")), ("σ uncracked", ("sigma_uncracked", "{:.1f}")), ("retention", ("retention", "{:.3f}")), ("rule", ("retention_rule", "{:.3f}")), ("mechanism", ("retention_mechanism", "{:.3f}")), ("obs/rule", ("retention_over_rule", "{:.2f}")),
                        ("reference", ("reference", "{}")), ("reference retention", ("ref_retention", "{:.3f}")), ("W ratio", ("W_ratio", "{:.2f}")), ("arrested at peak", ("arrested_at_peak", "{}")), ("mode", ("mode", "{}"))]))
    if ev:
        s = ev["summary"]; L.append("\n## Standard prediction evaluation (all designs with a completed run)\n")
        for k, v in s.items():
            if isinstance(v, dict):
                L.append(f"- {k}: median |rel. error| {v['median_abs_rel_error']:.2f}, mean |abs. error| {v['mean_abs_error']:.2f} (n = {v['n']})")
        L.append(f"- fracture-mode class accuracy: {fmt(s.get('fracture_mode_class_accuracy'))}")
    if pan:
        L.append(f"\nFracture panels written for {len(pan)} cracked runs in figures/hierarchy/panels/.")
    open(os.path.join(RES, "hierarchy_followup_results.md"), "w").write("\n".join(L))
    json.dump(dict(facts=facts, T1=dict(rows=t1, verdict=v1), T4=dict(rows=t4, verdict=v4), T2=dict(rows=t2, verdict=v2), T3=dict(rows=t3, verdict=v3), T5=dict(rows=t5, verdict=v5)), open(os.path.join(RES, "hierarchy_followup_results.json"), "w"), indent=1, default=db._json_default)
    print("\n".join(L))
    return facts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-panels", action="store_true"); a = ap.parse_args()
    main(no_panels=a.no_panels)
