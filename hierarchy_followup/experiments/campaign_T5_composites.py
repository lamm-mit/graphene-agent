"""T5, composite hierarchies (phase 3, added 2026-09-23 on request after T1 had started): coarse architectures whose
building blocks are themselves a DIFFERENT fine architecture.  Same campaign/stage as T1-T4 ("hierarchy" /
"hierarchy_followup"), own prediction file and own queue (experiments/specs/hierarchy_T5, run after T3).

Archetypes (300 A cell, porosity 0.20 +- 0.005 matched by bisection on the fine pore size or the bar width):
  VF_E   fibrous veins (bundles of load-parallel slit strips, 30 A every 100 A) + domains of pores elongated along the load
  VF_R   fibrous veins + domains of round pores
  VE_R   veins made of the aligned elongated-pore mesh + domains of round pores      (two different porous levels)
  VR_E   veins made of a round-pore mesh + domains of elongated pores               (porous, weaker veins: are solid veins needed?)
  VS_R   solid veins + domains of round pores                                        (solid-vein reference of VF_R / VE_R; T4 at scale)
  GR_0   mesh of meshes: square grid (bars 65 A every 100 A) whose bars are a round-pore mesh, coarse pores open
  GFx_0  fibre-bar frame: square grid whose bars along the load are slit bundles, bars across the load solid, coarse pores open
  V3_E   three levels: solid veins 30 A every 100 A, solid sub-veins 8 A every 33 A inside the domains, elongated pores between
  GS     solid square grid with 65 A bars (porosity 0.12, NOT mass-matched): flaw-tolerance control of the grid designs
Runs: every archetype uncracked in two registries and cracked in registry 0 (60 A crack centred mid-domain for the vein
designs, 40 A crack centred on a load-parallel bar for the grid designs); cracked registry 1 for VF_E and GR_0; GS
uncracked + cracked (29 runs).

Rules (inputs from reference/phase1_reference_numbers.json): column class rule sigma = sum over regions of the crack /
minimum column of SF_region x sigma_net(region): solid or fibrous veins, sub-veins and solid bars -> straight class;
any pore-filled region -> fine_round class ("rule"); alternative "rule_b" with the aligned elongated-pore mesh at its
own phase-1 net-section strength (S2_ellipse_90deg).  Cracked: net-section scaling from the uncracked rule.
Hypotheses and success criteria are in the notes of the prediction file.

    python experiments/campaign_T5_composites.py --dry-run
    python experiments/campaign_T5_composites.py
"""
from __future__ import annotations
import sys, os, json, argparse, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from atomistics.structures import design_space as D
from experiments.campaign_design import S, spec
from experiments.stage_tools import write_predictions, write_specs
from experiments.campaign_hierarchy_followup import analyse, refs, REG, AQS_H, L3, PHI, TOL, CAMPAIGN, STAGE

SPEC_DIR = os.path.join(ROOT, "experiments", "specs", "hierarchy_T5")
PRED_NAME = "hierarchy_T5_predictions"
FIBRE = dict(slit_len=26.0, slit_w=3.0, period_y=12.0)         # strips ~8 A wide after pruning (slits widen to ~4.2 A), 3.5 A bridges between slit tips
FIBRE_BAR = dict(slit_len=26.0, slit_w=3.0, period_y=20.0)     # strips ~16 A wide (frame bars)
ROUND_VEIN = dict(period=12.0, pore_d=5.0)


def porosity(params):
    a = D.generate("composite", **params); return a.info["porosity"], len(a)


def match_nested(params, target, path, lo, hi, increasing=True, tol=TOL, it=20):
    """Bisection on the nested parameter params[path[0]][path[1]] (or params[path[0]] if len(path) == 1)."""
    p = copy.deepcopy(params)
    def setv(v):
        if len(path) == 1: p[path[0]] = v
        else: p[path[0]][path[1]] = v
    for _ in range(it):
        mid = 0.5 * (lo + hi); setv(round(mid, 3))
        phi, n = porosity(p)
        if abs(phi - target) < tol:
            break
        if (phi < target) == increasing:
            lo = mid
        else:
            hi = mid
    phi, n = porosity(p)
    return p, phi, n


def base(coarse, fill_solid, fill_domain, reg, fill_solid_y=None, solid_params=None, domain_params=None, bar_w=65.0):
    dx, dy = REG[reg]
    p = dict(Lx=L3, Ly=L3, coarse=coarse, domain=100.0, vein_w=30.0, bar_w=bar_w, fill_solid=fill_solid, fill_domain=fill_domain, offset=[dx, dy], vein_offset=dy,
             solid_params=dict(solid_params or {}), domain_params=dict(domain_params or {}))
    if fill_solid_y is not None:
        p["fill_solid_y"] = fill_solid_y; p["solid_y_params"] = {}
    return p


def build(dry_run=False):
    R = refs(); C = R["classes"]; s_straight = C["straight"]["median"]; s_fine = C["fine_round"]["median"]
    e = R["references"]["S2_ellipse_90deg"]; s_ellipse = e["strength_Nm"] / e["minSF_corrected"]
    rows = []
    designs = {}
    for reg in (0, 1):
        # vein designs: match the domain pore size
        p, phi, n = match_nested(base("veins_x", "slits_x", "ellipse_x", reg, solid_params=FIBRE, domain_params=dict(period=12.0, pore_d=4.0)), PHI, ("domain_params", "pore_d"), 2.5, 5.9)
        designs[("VF_E", reg)] = p
        p, phi, n = match_nested(base("veins_x", "slits_x", "round", reg, solid_params=FIBRE, domain_params=dict(period=12.0, pore_d=5.0)), PHI, ("domain_params", "pore_d"), 2.5, 9.0)
        designs[("VF_R", reg)] = p
        p, phi, n = match_nested(base("veins_x", "ellipse_x", "round", reg, solid_params=dict(period=12.0, pore_d=4.6), domain_params=dict(period=12.0, pore_d=5.0)), PHI, ("domain_params", "pore_d"), 2.5, 9.0)
        designs[("VE_R", reg)] = p
        p, phi, n = match_nested(base("veins_x", "round", "ellipse_x", reg, solid_params=ROUND_VEIN, domain_params=dict(period=12.0, pore_d=4.0)), PHI, ("domain_params", "pore_d"), 2.5, 5.9)
        designs[("VR_E", reg)] = p
        p, phi, n = match_nested(base("veins_x", "none", "round", reg, domain_params=dict(period=12.0, pore_d=6.0)), PHI, ("domain_params", "pore_d"), 3.0, 10.0)
        designs[("VS_R", reg)] = p
        p, phi, n = match_nested(base("veins_x", "none", "subveins_ellipse", reg, domain_params=dict(period=12.0, pore_d=4.0, sub_domain=33.3, sub_w=8.0)), PHI, ("domain_params", "pore_d"), 2.5, 5.9)
        designs[("V3_E", reg)] = p
        # grid designs: mesh of meshes matches the pore size in the bars; the fibre frame matches the bar width (fibre strips fixed)
        p, phi, n = match_nested(base("square", "round", "empty", reg, solid_params=dict(period=12.0, pore_d=4.0), bar_w=65.0), PHI, ("solid_params", "pore_d"), 2.0, 8.0)
        designs[("GR_0", reg)] = p
        p, phi, n = match_nested(base("square", "slits_x", "empty", reg, fill_solid_y="none", solid_params=FIBRE_BAR, bar_w=75.0), PHI, ("bar_w",), 60.0, 95.0, increasing=False)
        designs[("GFx_0", reg)] = p
        designs[("GS", reg)] = base("square", "none", "empty", reg, bar_w=65.0)      # not mass-matched: flaw-tolerance control (same bars as GR_0)
    desc = {"VF_E": "fibrous veins (slit bundles, 30 A every 100 A) + domains of pores elongated along the load", "VF_R": "fibrous veins + domains of round pores",
            "VE_R": "veins made of the aligned elongated-pore mesh + domains of round pores", "VR_E": "veins made of a round-pore mesh + domains of elongated pores",
            "VS_R": "solid veins + domains of round pores", "GR_0": "mesh of meshes: square grid whose 65 A bars are a round-pore mesh, coarse pores open",
            "GFx_0": "fibre-bar frame: square grid whose bars along the load are slit bundles (~16 A strips), bars across the load solid, coarse pores open",
            "V3_E": "three levels: solid veins 30 A every 100 A, solid sub-veins 8 A every 33 A, elongated pores between", "GS": "solid square grid with 65 A bars (porosity 0.12, not mass-matched; control of GR_0 / GFx_0)"}
    parent = {"VF_E": "H3_2L_r0", "VF_R": "H5_VS_R_r0", "VE_R": "H5_VS_R_r0", "VR_E": "H3_2L_r0", "VS_R": "H3_2L_r0", "GR_0": "H5_GS_r0", "GFx_0": "H5_GS_r0", "V3_E": "H3_2L_r0", "GS": "H3_1L_coarse_r0"}
    order = 500
    def region_rule(col, key, rule_b=False):
        vein_cls = {"VF_E": s_straight, "VF_R": s_straight, "VE_R": (s_ellipse if rule_b else s_fine), "VR_E": s_fine, "VS_R": s_straight, "GR_0": s_fine, "GFx_0": s_straight, "V3_E": s_straight, "GS": s_straight}[key]
        dom_cls = {"VF_E": (s_ellipse if rule_b else s_fine), "VF_R": s_fine, "VE_R": s_fine, "VR_E": (s_ellipse if rule_b else s_fine), "VS_R": s_fine, "GR_0": 0.0, "GFx_0": 0.0, "V3_E": (s_ellipse if rule_b else s_fine), "GS": 0.0}[key]
        return col["SF_vein"] * vein_cls + col["SF_fine"] * dom_cls
    hyp = {"VF_E": "graceful veins: a slit bundle cannot be severed at once, each strip fails separately; strength <= the solid-vein reference (H3_2L) by the rule, residual after the first avalanche >= 0.6 and post-peak energy >= 1.5 x H3_2L; cracked: retention >= 1.15 x rule and >= the retention of H3_2L_L60",
           "VF_R": "graceful veins with a weaker (round-pore) domain filler that fails first: two-stage failure with residual >= 0.6 and post-peak energy >= 1.5 x VS_R",
           "VE_R": "two different porous levels: the aligned-pore vein material (~22 N/m net section) replaces solid graphene; hypothesis: strength = rule_b, no crack-arrest bonus (retention = rule)",
           "VR_E": "porous, weaker veins: strength = rule (below H3_2L); hypothesis: no flaw-tolerance advantage over the one-level fine mesh (retention <= H3_1L_fine_L60)",
           "VS_R": "solid veins with a round-pore filler (T4 geometry at scale, vein fraction 0.3): reference of VF_R / VE_R; two-stage hypothesis as T4 (residual >= 0.6, post-peak energy >= 1.5 x H3_1L_fine)",
           "GR_0": "mesh of meshes: strength = rule (fine-round bars, well below the solid-bar grid), but damage tolerance: residual >= 0.6, post-peak strain range >= 0.03; cracked (40 A in a bar): retention >= 1.15 x rule and > GS_L40 (pores blunt the tip)",
           "GFx_0": "fibre-bar frame: strength = rule (strips), progressive failure strip by strip: residual >= 0.6; cracked: retention >= 1.15 x rule and > GS_L40",
           "V3_E": "three levels: strength within the one-level scatter of H3_2L; cracked (60 A mid-domain, severing two sub-veins): retention >= 1.10 x H3_2L_L60 (sub-veins as additional arrest lines)",
           "GS": "solid-bar grid control (same bars as GR_0/GFx_0): strength = rule; cracked (40 A in a bar): net-section rule, sharp tip -> retention below the rule as for the pristine sheet"}
    keys = ["VF_E", "VF_R", "VE_R", "VR_E", "VS_R", "GR_0", "GFx_0", "V3_E", "GS"]
    C_ = type("C", (), {})(); C_.rows = rows; C_.cache = {}
    def add(name, key, params, reg, cracked, reason, pred, rationale, tags):
        g = analyse("composite", params)
        col = g["col"]
        row = dict(test="T5", order=len(rows), name=name, family="composite", params=params, seed=0, reason=reason, tags=tags, parent=parent[key] if not cracked else f"H5_{key}_r{reg}",
                   notes=None, prediction=pred(g), rationale=rationale, geometry={k: g[k] for k in ("n_atoms", "porosity", "porosity_base", "minSF", "A", "col", "spanning", "n_fragments")},
                   crack_actual=g["crack_actual"], tip_to_vein_A=g.get("tip_to_vein_A"), spec_dir=SPEC_DIR, archetype=key, registry=reg)
        rows.append(row); C_.cache[name] = row; return row
    for key in keys:
        for reg in (0, 1):
            p = designs[(key, reg)]
            def pred_unc(g, key=key):
                col = g["col"]; r = region_rule(col, key); rb = region_rule(col, key, rule_b=True)
                return dict(strength_Nm_rule=round(r, 1), strength_Nm_rule_b=round(rb, 1), fracture_mode_class={"GS": "abrupt", "VS_R": "stepwise", "V3_E": "stepwise"}.get(key, "progressive"),
                            residual_fraction_hypothesis=0.6 if key in ("VF_E", "VF_R", "VS_R", "GR_0", "GFx_0") else None,
                            work_to_failure_J_m2=round(0.5 * r * 0.13 + 0.3 * r * 0.05, 2), modulus_2d_Nm=round(col["SF_vein"] * 250 + col["SF_fine"] * 120, 0))
            add(f"H5_{key}_r{reg}", key, p, reg, False, f"T5 composite hierarchy, registry {reg}: {desc[key]}", pred_unc,
                f"column class rule with straight {s_straight:.1f} / fine_round {s_fine:.1f} N/m (rule_b: aligned elongated-pore regions at {s_ellipse:.1f} N/m); {hyp[key]}",
                ["hierarchy", "T5", key, f"registry{reg}", "uncracked"] + (["control"] if key == "GS" else []))
        # cracked: registry 0 for all, registry 1 for VF_E and GR_0
        for reg in ((0, 1) if key in ("VF_E", "GR_0") else (0,)):
            p = designs[(key, reg)]; dx, dy = REG[reg]
            if key.startswith("G"):
                cl = 40.0; center = [round(0.5 + dx / L3, 6), round((100.0 + dy) / L3, 6)]; where = "centred on a load-parallel bar (tips 12.5 A from the bar edges)"
            else:
                cl = 60.0; center = [round(0.5 + dx / L3, 6), round((50.0 + dy) / L3, 6)]; where = "centred mid-domain (tips 5 A from the veins)"
            pc = dict(copy.deepcopy(p), crack_len=cl, crack_w=3.0, crack_angle_deg=90.0, crack_center=center)
            unc = C_.cache[f"H5_{key}_r{reg}"]; sf_u = unc["geometry"]["col"]["SF"]; s_u = unc["prediction"]["strength_Nm_rule"]
            mech = 1.15 if key in ("VF_E", "VF_R", "GR_0", "GFx_0") else (1.10 if key == "V3_E" else 1.0)
            def pred_cr(g, sf_u=sf_u, s_u=s_u, mech=mech, key=key):
                sf = g["col"]["SF"]; ret = sf / sf_u
                return dict(strength_Nm_rule=round(s_u * ret, 1), strength_Nm_mechanism=round(s_u * ret * mech, 1), retention_rule=round(ret, 3), retention_mechanism=round(ret * mech, 3),
                            fracture_mode_class="abrupt" if key == "GS" else "progressive")
            add(f"H5_{key}_L{int(cl)}_r{reg}", key, pc, reg, True, f"T5 cracked, registry {reg}: {desc[key]} + {int(cl)} A crack perpendicular to the load {where}", pred_cr,
                f"net-section scaling from the uncracked rule value (same registry); mechanism factor {mech}: {hyp[key]}", ["hierarchy", "T5", key, f"registry{reg}", "cracked", f"L{int(cl)}"])
    # ---- table
    print(f"{'name':22s} {'N':>6s} {'phi0':>5s} {'phi':>5s} {'SFcol':>6s} {'vein':>5s} {'fine':>5s} {'A':>6s} {'Lcr':>5s} {'tips':>12s} {'rule':>6s} {'rule_b/mech':>11s} span")
    for r in rows:
        g = r["geometry"]; ca = r["crack_actual"] or {}; p = r["prediction"]
        print(f"{r['name']:22s} {g['n_atoms']:6d} {g['porosity_base']:5.3f} {g['porosity']:5.3f} {g['col']['SF']:6.3f} {g['col']['SF_vein']:5.3f} {g['col']['SF_fine']:5.3f} {g['A']:+6.2f} {ca.get('crack_len_actual', 0):5.1f} {str(r.get('tip_to_vein_A') or ''):>12s} {p['strength_Nm_rule']:6.1f} {p.get('strength_Nm_rule_b', p.get('strength_Nm_mechanism', float('nan'))):11.1f} {g['spanning']}")
    bad = [r["name"] for r in rows if not r["geometry"]["spanning"] or r["geometry"]["n_fragments"] != 1]
    print(len(rows), "designs;", "all spanning" if not bad else f"NOT SPANNING: {bad}")
    if bad:
        raise SystemExit("fix the designs above")
    if dry_run:
        return rows
    preds = [dict(name=r["name"], test="T5", archetype=r["archetype"], family="composite", params=r["params"], seed=0, parent=r["parent"], porosity=round(r["geometry"]["porosity"], 4), porosity_base=round(r["geometry"]["porosity_base"], 4),
                  n_atoms=r["geometry"]["n_atoms"], minSF=round(r["geometry"]["minSF"], 4), alignment_A=round(r["geometry"]["A"], 4), crack_column=r["geometry"]["col"], crack_actual=r["crack_actual"], tip_to_vein_A=r.get("tip_to_vein_A"),
                  predicted=r["prediction"], rationale=r["rationale"], tags=r["tags"]) for r in rows]
    notes = ("Pre-registered predictions of T5 (composite hierarchies), written before its specs were written; T5 runs after T3 in its own queue. Success criteria: "
             "H5a fibrous veins (VF_E, VF_R): residual after the first avalanche >= 0.6 at both registries and post-peak energy >= 1.5 x the solid-vein reference (H3_2L for VF_E, VS_R for VF_R); cracked retention >= 1.15 x rule and >= the reference's cracked retention. "
             "H5b porous veins (VE_R, VR_E): rule only; pass = no flaw-tolerance advantage (retention <= reference). "
             "H5c grid designs (GR_0, GFx_0): residual >= 0.6 and post-peak strain range >= 0.03; cracked retention >= 1.15 x rule and > GS_L40 retention. "
             "H5d three levels (V3_E): strength within the one-level scatter of H3_2L (2.0 N/m); cracked retention >= 1.10 x H3_2L_L60 retention. "
             f"Rule inputs: straight {s_straight:.2f}, fine_round {s_fine:.2f}, aligned elongated-pore mesh {s_ellipse:.2f} N/m per unit section. Loading as T1-T4 (AQS_H).")
    path, sha = write_predictions(PRED_NAME, preds, notes)
    batches = [spec(CAMPAIGN, STAGE, f"{500 + k:03d}_T5_{r['name']}", [S(r["name"], "composite", r["params"], seed=0, reason=r["reason"], hypothesis=r["rationale"], tags=r["tags"], parent=r["parent"], prediction=r["prediction"])], aqs=AQS_H) for k, r in enumerate(rows)]
    write_specs(SPEC_DIR, batches)
    print(f"predictions: {path} (sha256 {sha[:16]}...)  specs: {len(batches)} in {os.path.relpath(SPEC_DIR, ROOT)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    build(dry_run=a.dry_run)
