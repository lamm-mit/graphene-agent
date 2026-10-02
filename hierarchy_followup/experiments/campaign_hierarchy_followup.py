"""Hierarchy follow-up campaign (phase 3): campaign "hierarchy", stage "hierarchy_followup".

Implements HIERARCHY_FOLLOWUP_MEMO.md (T1 crack arrest by veins, T4 designed two-stage failure, T2 Cook-Gordon seams,
T3 scale separation) with the deviations listed in README.md.  For every design the structure is generated and
analysed before any simulation (porosity before and after the crack, realised crack length and tip positions, solid
fraction of the crack column split into veins and fine level, minimum section, alignment index) and quantitative
predictions from the stated rules are written to a timestamped, hashed file BEFORE the specs are written:

  experiments/predictions/hierarchy_followup_predictions.json     (pre-registration)
  experiments/specs/hierarchy/*.json                              T1, T4, T2 (one design per spec; queue order = file order)
  experiments/specs/hierarchy_T3/*.json                           T3 (300 A cells, run last)

Rules (all inputs from reference/phase1_reference_numbers.json, written by analysis/assess_phase1.py):
  net-section scaling   sigma_cracked = sigma_uncracked(reference) x SF_col(cracked) / SF_col(uncracked); the column is the
                        crack column (minimum solid fraction over the raster columns crossing the crack), so
                        retention_rule = SF ratio.  Work to failure is scaled with the same ratio (W follows sigma, r = 0.82).
  column class rule     sigma = f_vein x sigma_net(straight) + f_fine x sigma_net(fine_round) with the solid fractions of the
                        minimum column split into vein bands and fine level (T3, T4; also recorded for T1).
  alignment trend       sigma = a A + b of the one-level meshes of phase 1 (T3: the hypothesis under test is a premium
                        beyond the one-level scatter).
  mechanism hypotheses  T1: veins retention >= 1.15 x rule (crack arrested at the first vein at peak load); slit array = rule;
                        pristine sheet follows the phase-1 notch series (0.48 at 20 A).
                        T2: crack deflects into the first seam, strength within 10 % of the cracked pristine reference, W ratio >= 1.3.
                        T3: two-level design shows a W or crack-retention premium beyond the one-level scatter although its
                        strength does not.
                        T4: residual after the first avalanche >= 0.6 and post-peak energy >= 1.5 x the one-level control.

    python experiments/campaign_hierarchy_followup.py --dry-run     # table only, nothing written
    python experiments/campaign_hierarchy_followup.py               # predictions (hashed) THEN specs
"""
from __future__ import annotations
import sys, os, json, math, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
from atomistics.descriptors.columns import column_analysis, vein_bands, in_vein
from experiments.campaign_design import AQS, S, spec, match
from experiments.campaign_paper_sweeps import alignment_index
from experiments.stage_tools import write_predictions, write_specs

CAMPAIGN, STAGE = "hierarchy", "hierarchy_followup"
SPEC_DIR = os.path.join(ROOT, "experiments", "specs", "hierarchy")
SPEC_DIR_T3 = os.path.join(ROOT, "experiments", "specs", "hierarchy_T3")
PRED_NAME = "hierarchy_followup_predictions"
REF_FILE = os.path.join(ROOT, "reference", "phase1_reference_numbers.json")
# loading protocol: every physics constant of phase 1 (strain steps, bisection, FIRE fmax, transverse relaxation) is unchanged;
# only the two termination bounds are extended so that a second load-carrying stage cannot be truncated.
AQS_H = dict(AQS, max_strain=0.35, post_peak_strain_limit=0.20)
LC = 120.0                     # cell of T1, T2, T4 (A)
L3 = 300.0                     # cell of T3
PHI = 0.20
REG = {0: (0.0, 0.0), 1: (1.23, 0.0)}    # lattice-registry replicates: rigid shift of the whole pattern (pores, veins, slits, crack) by half a lattice constant
                                         # along x.  NOTE (1.23, 2.13) is a primitive translation of the honeycomb lattice and would reproduce registry 0 exactly.
TOL = 0.005                    # porosity matching tolerance (phase 1: 0.008)


def refs():
    R = json.load(open(REF_FILE))
    return R


def analyse(family, params):
    a = D.generate(family, **params)
    dsg = a.info["design"]; d = compute_descriptors(a, dsg)
    ca = a.info.get("crack_actual")
    x0 = dsg["crack_center"][0] * dsg["Lx"] if ca else None
    col = column_analysis(a, x0)
    from simulation.topology import spanning_x
    from atomistics.descriptors.descriptors import bond_graph
    bi, bj, bS, _ = bond_graph(a)
    out = dict(n_atoms=len(a), porosity=float(a.info["porosity"]), porosity_base=float(dsg.get("porosity_base", a.info["porosity"])), minSF=d["min_solid_fraction_across_x"],
               A=alignment_index(d), col=col, design=dsg, crack_actual=ca, descriptors_keys=len(d), spanning=bool(spanning_x(bi, bj, bS, len(a))), n_fragments=int(d["n_fragments"]))
    if ca:
        bands = vein_bands(dsg)
        if bands:
            # distance from each crack tip to the edge of the next vein AHEAD of it (lower tip: downwards, upper tip: upwards);
            # negative: the tip lies inside that vein
            def tip_gap(y, direction):
                best = None
                for yc, hw, sy in bands:
                    dd = (y - yc) if direction < 0 else (yc - y)
                    dd = dd % sy                      # vein centre ahead of the tip, periodic
                    g = dd - hw
                    best = g if best is None or g < best else best
                return best
            out["tip_to_vein_A"] = [round(tip_gap(ca["tip_lo"][1], -1), 2), round(tip_gap(ca["tip_hi"][1], +1), 2)]
    return out


class Campaign:
    def __init__(self, R):
        self.R = R; self.rows = []; self.cache = {}
        C = R["classes"]; self.s_straight = C["straight"]["median"]; self.s_fine = C["fine_round"]["median"]; self.s_coarse = C["coarse_round"]["median"]
        self.fit1 = R["alignment"]["fit_one_level"]; self.scatter = R["alignment"]["one_level_resid_std"]; self.W_scatter = R["alignment"]["one_level_W_std"]
        self.ref = R["references"]

    # ---- helpers
    def add(self, test, order, name, family, params, reason, tags, prediction, rationale, parent=None, seed=0, notes=None, extra=None, spec_dir=None):
        g = analyse(family, params)
        row = dict(test=test, order=order, name=name, family=family, params=params, seed=seed, reason=reason, tags=tags, parent=parent, notes=notes,
                   prediction=prediction, rationale=rationale, geometry={k: g[k] for k in ("n_atoms", "porosity", "porosity_base", "minSF", "A", "col", "spanning", "n_fragments")},
                   crack_actual=g["crack_actual"], tip_to_vein_A=g.get("tip_to_vein_A"), spec_dir=spec_dir or SPEC_DIR)
        if extra:
            row.update(extra)
        self.rows.append(row); self.cache[name] = row
        return row

    def matched(self, family, base, key, lo, hi, target=PHI):
        p, phi, n = match(family, base, target, key, lo, hi, tol=TOL)
        return p, phi, n

    def shifted_center(self, base_xy, reg, Lx=LC, Ly=LC):
        dx, dy = REG[reg]
        return [round(base_xy[0] + dx / Lx, 6), round(base_xy[1] + dy / Ly, 6)]

    # ---- T1: crack arrest by veins ------------------------------------------------------------------------------
    def T1(self):
        R = self.ref; s0 = R["S1_pristine_zz"]["strength_Nm"]
        notch = self.R["pristine_notch"]
        ret20 = notch["S1_precrack_L20"]["retention"]; ret40 = 0.5 * (notch["S4_precrack_L40_cell150"]["retention"] + notch["S1_precrack_L30"]["retention"])
        order = 0
        # (a) pristine sheet with cracks (reference notch sensitivity in the 120 A cell), two registries (crack shifted by the registry vector)
        for reg in (0, 1):
            for cl, ret in ((20, ret20), (40, ret40)):
                params = dict(Lx=LC, Ly=LC, orientation="zigzag", crack_len=float(cl), crack_w=3.0, crack_angle_deg=90.0, crack_center=self.shifted_center((0.5, 0.5), reg))
                g = analyse("pristine", params)
                sf = g["col"]["SF"]
                pred = dict(strength_Nm_rule=round(s0 * sf, 1), strength_Nm_mechanism=round(s0 * ret, 1), retention_rule=round(sf, 3), retention_mechanism=round(ret, 3),
                            modulus_2d_Nm=round(R["S1_precrack_L20"]["modulus_2d_Nm"] * (1 - 0.15 * (cl - 20) / 20), 0), work_to_failure_J_m2=round(R["S1_precrack_L20"]["work_to_failure_J_m2"] * (ret / ret20), 2),
                            failure_strain=round(R["S1_precrack_L20"]["failure_strain"] * (ret / ret20), 3), fracture_mode_class="abrupt")
                self.add("T1", order, f"H1_pristine_L{cl}_r{reg}", "pristine", params,
                         reason=f"T1 reference: pristine sheet with a {cl} A crack perpendicular to the load in the 120 A cell, registry {reg}",
                         tags=["hierarchy", "T1", "flaw", "pristine", f"registry{reg}", "cracked"], parent="S1_pristine_zz", prediction=pred,
                         rationale=f"sharp crack in the pristine sheet: net section {sf:.2f} but the phase-1 notch series gives retention {ret:.2f} (lattice-trapping plateau); the cell is 120 A instead of 100/150 A")
                order += 1
        # (b) three porous architectures at phi 0.20 in the 120 A cell, uncracked (two registries), then cracked (20 and 40 A, two registries)
        arch = {}
        for reg in (0, 1):
            dx, dy = REG[reg]
            p, phi, n = self.matched("nanomesh_single", dict(Lx=LC, Ly=LC, period=16, aspect=2.0, angle_deg=90, offset=[dx, dy]), "pore_d", 3.0, 9.0)
            arch[("ellipse", reg)] = ("nanomesh_single", p, R["S2_ellipse_90deg"], "one-level mesh of elongated pores along the load (S2_ellipse_90deg parameters, 120 A cell)")
            p, phi, n = self.matched("nanomesh_hier", dict(Lx=LC, Ly=LC, levels=2, period=12, domain=40, vein_w=12, vein_dirs="x", aspect=2.0, angle_deg=90, vein_offset=dy, offset=[dx, dy]), "pore_d", 3.0, 5.9)
            arch[("veinsX", reg)] = ("nanomesh_hier", p, R["S7_H2_veinsX_W12_ellipseX"], "two-level mesh, 12 A veins along the load, elongated fine pores (S7_H2_veinsX_W12_ellipseX parameters)")
            p, phi, n = self.matched("slit_array", dict(Lx=LC, Ly=LC, slit_w=4.0, period_x=30, period_y=16, angle_deg=0, offset=[dx, dy]), "slit_len", 8.0, 29.0)
            arch[("slit", reg)] = ("slit_array", p, R["S2_slit_0deg"], "load-parallel slit array (S2_slit_0deg parameters, 120 A cell)")
        cls = {"ellipse": self.s_straight, "veinsX": None, "slit": self.s_straight}
        for key in ("ellipse", "veinsX", "slit"):
            for reg in (0, 1):
                fam, p, ref, desc = arch[(key, reg)]
                name = f"H1_{key}_r{reg}"
                g = analyse(fam, p); col = g["col"]
                class_rule = (col["SF_vein"] * self.s_straight + col["SF_fine"] * self.s_fine) if key == "veinsX" else col["SF"] * cls[key]
                pred = dict(strength_Nm_rule=round(ref["strength_Nm"], 1), strength_Nm_class_rule=round(class_rule, 1), modulus_2d_Nm=round(ref["modulus_2d_Nm"], 0),
                            failure_strain=round(ref["failure_strain"], 3), work_to_failure_J_m2=round(ref["work_to_failure_J_m2"], 2), damage_localization=round(ref["damage_localization"], 2),
                            fracture_mode_class={"ellipse": "progressive", "veinsX": "stepwise", "slit": "stepwise"}[key])
                self.add("T1", order, name, fam, p, reason=f"T1 uncracked reference, registry {reg}: {desc}", tags=["hierarchy", "T1", key, f"registry{reg}", "uncracked"],
                         parent=[n_ for n_, v in R.items() if v is ref][0], prediction=pred,
                         rationale=f"same design as the phase-1 record (cell {ref['Lx']:.0f} A -> 120 A, registry {reg}); rule = the phase-1 value; class rule = column rule with sigma_net straight {self.s_straight:.1f} / fine {self.s_fine:.1f} N/m")
                order += 1
        for key in ("ellipse", "veinsX", "slit"):
            for reg in (0, 1):
                fam, p, ref, desc = arch[(key, reg)]
                unc = self.cache[f"H1_{key}_r{reg}"]; sf_u = unc["geometry"]["col"]["SF"]
                for cl in (20, 40):
                    dx, dy = REG[reg]
                    if key == "veinsX":
                        # veins at y = 0, 40, 80 (+ vein_offset): the 20 A crack is centred mid-domain (tips 4 A from the vein edges), the 40 A crack is centred
                        # on the vein at y = 40 (it severs that vein; its tips end 14 A from the edges of the next veins)
                        yc = 0.5 * 40 if cl == 20 else 40.0
                        center = [round(0.5 + dx / LC, 6), round((yc + dy) / LC, 6)]
                    elif key == "slit":
                        # slit rows at y = k * py (py = Ly / round(Ly / period_y)); the crack is centred on the strip nearest the cell centre so that it
                        # severs whole strips (one for 20 A, three for 40 A) and nothing else
                        py = LC / round(LC / 16); yc = (round(0.5 * LC / py - 0.5) + 0.5) * py
                        center = [round(0.5 + dx / LC, 6), round((yc + dy) / LC, 6)]
                    else:
                        center = self.shifted_center((0.5, 0.5), reg)
                    pc = dict(p, crack_len=float(cl), crack_w=3.0, crack_angle_deg=90.0, crack_center=center)
                    g = analyse(fam, pc); col = g["col"]; sf_c = col["SF"]
                    ret_rule = sf_c / sf_u
                    s_rule = ref["strength_Nm"] * ret_rule
                    if key == "veinsX":
                        mech = 1.15; why = ("crack arrest: the crack tips face load-parallel veins (Y ~ 250 N/m) that are stiffer than the fine-pored domain (~100 N/m); retention 15 % above the net-section rule with the crack arrested at the first vein at peak load"
                                            + ("; the 40 A crack severs the vein it is centred on, the next veins are 14 A ahead of its tips" if cl == 40 else "; tips 4 A from the vein edges"))
                        class_rule = col["SF_vein"] * self.s_straight + col["SF_fine"] * self.s_fine
                    elif key == "slit":
                        mech = 1.0; why = "strips are independent: the crack removes strips and nothing else, the net-section rule is exact"
                        class_rule = sf_c * self.s_straight
                    else:
                        mech = 1.0; why = "no barrier ahead of the tips (fine ligaments everywhere): net-section rule; far less notch-sensitive than the pristine sheet because the pores already blunt the tip"
                        class_rule = sf_c * self.s_straight
                    pred = dict(strength_Nm_rule=round(s_rule, 1), strength_Nm_mechanism=round(s_rule * mech, 1), strength_Nm_class_rule=round(class_rule, 1),
                                retention_rule=round(ret_rule, 3), retention_mechanism=round(ret_rule * mech, 3), modulus_2d_Nm=round(ref["modulus_2d_Nm"] * (1 - 0.06 * cl / 20), 0),
                                work_to_failure_J_m2=round(ref["work_to_failure_J_m2"] * ret_rule * mech, 2), failure_strain=round(ref["failure_strain"] * ret_rule, 3),
                                fracture_mode_class={"ellipse": "progressive", "veinsX": "stepwise", "slit": "abrupt"}[key])
                    self.add("T1", order, f"H1_{key}_L{cl}_r{reg}", fam, pc, reason=f"T1 cracked, registry {reg}: {desc} + {cl} A crack perpendicular to the load" + (" centred mid-domain (tips point at the veins)" if key == "veinsX" else ""),
                             tags=["hierarchy", "T1", key, f"registry{reg}", "cracked", f"L{cl}"], parent=f"H1_{key}_r{reg}", prediction=pred,
                             rationale=f"net-section scaling from the uncracked design of the same registry: SF_col {sf_c:.3f} / {sf_u:.3f} = {ret_rule:.3f}; {why}",
                             extra=dict(SF_uncracked=sf_u))
                    order += 1

    # ---- T4: designed two-stage failure --------------------------------------------------------------------------
    def T4(self):
        R = self.ref; order = 100
        ctrl = R["S2_ellipse_90deg"]; ctrl_E = ctrl["curve_post_peak_energy_J_m2"]
        for W in (16, 20):
            for reg in (0, 1):
                dx, dy = REG[reg]
                p, phi, n = self.matched("nanomesh_hier", dict(Lx=LC, Ly=LC, levels=2, period=12, domain=40, vein_w=float(W), vein_dirs="x", aspect=1.0, angle_deg=0.0, vein_offset=dy, offset=[dx, dy]), "pore_d", 4.0, 10.5)
                g = analyse("nanomesh_hier", p); col = g["col"]
                veins_alone = col["SF_vein"] * self.s_straight; peak = veins_alone + col["SF_fine"] * self.s_fine
                pred = dict(strength_Nm_rule=round(peak, 1), strength_Nm_veins_alone=round(veins_alone, 1), residual_fraction_rule=round(veins_alone / peak, 3), residual_fraction_hypothesis=0.6,
                            post_peak_energy_J_m2_control=round(ctrl_E, 3), post_peak_energy_J_m2_hypothesis=round(1.5 * ctrl_E, 3),
                            modulus_2d_Nm=round(col["SF_vein"] * 250 + (1 - col["SF_vein"]) * 120, 0),
                            work_to_failure_J_m2=round(0.5 * peak * 0.13 + veins_alone * 0.07, 2), failure_strain=0.20, fracture_mode_class="stepwise")
                self.add("T4", order, f"H4_veinsX_W{W}_r{reg}", "nanomesh_hier", p,
                         reason=f"T4 designed two-stage failure, registry {reg}: {W} A veins along the load (vein fraction {W/40:.2f}), round fine pores (period 12) at porosity 0.20",
                         tags=["hierarchy", "T4", "H2", "veins_x", f"W{W}", f"registry{reg}", "uncracked"], parent="S4_H2_D40_W8_veins_x", prediction=pred,
                         rationale=(f"column rule: veins alone carry SF_vein {col['SF_vein']:.3f} x {self.s_straight:.1f} = {veins_alone:.1f} N/m, the fine level adds SF_fine {col['SF_fine']:.3f} x {self.s_fine:.1f}; "
                                    f"peak {peak:.1f} N/m, residual fraction {veins_alone/peak:.2f}. Hypothesis: the round-pore fine level fails first (~0.12 strain) and the veins keep carrying "
                                    f">= 0.6 of the peak with post-peak energy >= 1.5 x the one-level control ({ctrl_E:.2f} J/m2, S2_ellipse_90deg); phase-1 square-vein W16/W20 (veins also across the load) "
                                    f"did not show it (residual {R['S5_H2_D40_W16']['curve_residual_after_first_avalanche']:.2f} then fracture within 0.025 strain; single avalanche)"))
                order += 1

    # ---- T2: Cook-Gordon seams ----------------------------------------------------------------------------------
    def T2(self):
        R = self.ref; order = 200
        s_ref = R["S1_precrack_L20"]["strength_Nm"] * 1.016   # 100 -> 120 A cell: +1.6 % (interpolated from the 30 A pair 18.72 -> 19.45 N/m at 100 -> 150 A)
        W_ref = R["S1_precrack_L20"]["work_to_failure_J_m2"] * 1.016
        designs = [("seamX", dict(seam_dir="x", pitch=8.0), 20), ("seamX", dict(seam_dir="x", pitch=8.0), 30), ("seamX", dict(seam_dir="x", pitch=8.0), 40),
                   ("weakseamX", dict(seam_dir="x", pitch=6.5), 20), ("seamY", dict(seam_dir="y", pitch=8.0), 20), ("random", dict(seam_dir="x", pitch=8.0, placement="random"), 20)]
        for key, kw, sp in designs:
            for reg in (0, 1):
                dx, dy = REG[reg]
                anchor = [round(0.5 + dx / LC, 6), round(0.5 + dy / LC, 6)]
                params = dict(Lx=LC, Ly=LC, pore_d=4.0, seam_spacing=float(sp), seam_anchor=anchor, skip_near=[anchor[0] * LC, anchor[1] * LC, 8.0, 10.0],
                              crack_len=20.0, crack_w=3.0, crack_angle_deg=90.0, crack_center=anchor, **kw)
                seed = 1 + reg if key == "random" else 0
                g = analyse("seam_pores", params)
                phi_seam = g["porosity_base"]
                s_rule = s_ref * (1 - phi_seam); W_rule = W_ref * (1 - phi_seam)
                if key in ("seamX", "weakseamX"):
                    gap = sp - 10.0          # tip-to-seam distance for a 20 A crack with a row through the crack centre
                    lig = kw["pitch"] - 4.0
                    row_strength = lig / kw["pitch"] * self.s_fine
                    mech_W = 1.3; mech_s = 0.9 * s_ref
                    why = (f"Cook-Gordon: rows of pores along the load are weak in tension across the row (row net section {lig/kw['pitch']:.2f}, ~{row_strength:.0f} N/m against ~{self.s_straight:.0f} N/m "
                           f"for the sheet); the stress parallel to the crack ahead of the tip (~1/5 of the opening stress) opens the row {gap:.0f} A ahead of the tips before the crack arrives, "
                           f"deflecting and blunting it: W ratio >= 1.3 with the strength within 10 % of the cracked pristine reference")
                    pred = dict(strength_Nm_rule=round(s_rule, 1), strength_Nm_mechanism=round(mech_s, 1), work_to_failure_J_m2=round(W_rule, 2), W_ratio_rule=round(1 - phi_seam, 3),
                                W_ratio_hypothesis=mech_W, deflection_expected=True, tip_to_seam_A=gap, row_net_section=round(lig / kw["pitch"], 3), fracture_mode_class="stepwise",
                                modulus_2d_Nm=round(R["S1_precrack_L20"]["modulus_2d_Nm"] * (1 - 1.5 * phi_seam), 0), failure_strain=round(R["S1_precrack_L20"]["failure_strain"] * 1.3, 3))
                    tags = ["hierarchy", "T2", "seams", f"s{sp}", f"registry{reg}", "cracked"] + (["weak_seam"] if key == "weakseamX" else [])
                    reason = f"T2 seams along the load every {sp} A (pores {params['pore_d']} A at pitch {kw['pitch']} A, one row through the crack centre, tips {gap:.0f} A from the next rows) + 20 A crack perpendicular to the load, registry {reg}"
                else:
                    why = {"seamY": "control: the same pores in columns across the load (parallel to the crack, 10 A on either side of it): no weak plane ahead of the tips, only a loss of section; rule only",
                           "random": "control: the same number of pores at random positions (minimum separation 8 A, crack flanks kept free): no organised weak plane; rule only"}[key]
                    alt = g["minSF"] * self.s_fine if key == "seamY" else None    # columns of 4 A pores across the load: net section minSF with fine ligaments, if that column fails first
                    pred = dict(strength_Nm_rule=round(s_rule, 1), strength_Nm_mechanism=round(s_rule, 1), strength_Nm_alt_rule=(round(alt, 1) if alt else None), work_to_failure_J_m2=round(W_rule, 2), W_ratio_rule=round(1 - phi_seam, 3),
                                W_ratio_hypothesis=round(1 - phi_seam, 3), deflection_expected=False, fracture_mode_class="abrupt",
                                modulus_2d_Nm=round(R["S1_precrack_L20"]["modulus_2d_Nm"] * (1 - 1.5 * phi_seam), 0), failure_strain=round(R["S1_precrack_L20"]["failure_strain"], 3))
                    tags = ["hierarchy", "T2", "control", key, f"registry{reg}", "cracked"]
                    reason = f"T2 control ({key}): {g['design']['n_pores']} pores of {params['pore_d']} A, {'columns across the load every ' + str(sp) + ' A' if key == 'seamY' else 'random placement'} + 20 A crack, registry {reg}"
                self.add("T2", order, f"H2_{key}_s{sp}_r{reg}", "seam_pores", params, reason=reason, tags=tags, parent="H1_pristine_L20_r%d" % reg, prediction=pred, seed=seed,
                         rationale=f"rule: strength and W of the cracked pristine reference ({s_ref:.1f} N/m, {W_ref:.2f} J/m2 after the 100 -> 120 A cell correction) reduced by the seam porosity {phi_seam:.3f}; {why}",
                         extra=dict(seam_porosity=phi_seam))
                order += 1

    # ---- T3: scale separation -----------------------------------------------------------------------------------
    def T3(self):
        R = self.ref; order = 300
        for reg in (0, 1):
            dx, dy = REG[reg]
            p2, phi, n = self.matched("nanomesh_hier", dict(Lx=L3, Ly=L3, levels=2, period=12, domain=100, vein_w=30, vein_dirs="x", aspect=2.0, angle_deg=90, vein_offset=dy, offset=[dx, dy]), "pore_d", 3.0, 5.9)
            pf, phi, n = self.matched("nanomesh_single", dict(Lx=L3, Ly=L3, period=12, aspect=2.0, angle_deg=90, offset=[dx, dy]), "pore_d", 3.0, 5.9)
            pc, phi, n = self.matched("nanomesh_single", dict(Lx=L3, Ly=L3, period=100, shape="square", stagger=False, offset=[dx, dy]), "pore_d", 20.0, 80.0)
            trio = [("2L", "nanomesh_hier", p2, "two-level mesh: 30 A veins along the load every 100 A (level ratio 100/12 = 8, eight pore rows per domain), elongated fine pores", "stepwise"),
                    ("1L_fine", "nanomesh_single", pf, "one-level fine mesh (period 12, elongated pores along the load)", "progressive"),
                    ("1L_coarse", "nanomesh_single", pc, "one-level coarse square mesh (period 100, ~55 A ligaments)", "abrupt")]
            for key, fam, p, desc, mode in trio:
                g = analyse(fam, p); col = g["col"]
                class_rule = col["SF_vein"] * self.s_straight + col["SF_fine"] * (self.s_fine if key != "1L_coarse" else self.s_straight)
                trend = self.fit1[0] * g["A"] + self.fit1[1]
                pred = dict(strength_Nm_rule=round(class_rule, 1), strength_Nm_trend=round(trend, 1), alignment_A=round(g["A"], 3), one_level_scatter_Nm=round(self.scatter, 2), one_level_W_scatter=round(self.W_scatter, 2),
                            work_to_failure_J_m2=round({"2L": R["S7_H2_veinsX_W12_ellipseX"]["work_to_failure_J_m2"], "1L_fine": R["S2_ellipse_90deg"]["work_to_failure_J_m2"] * 0.75, "1L_coarse": R["S4_square_D40"]["work_to_failure_J_m2"]}[key], 2),
                            modulus_2d_Nm=round({"2L": 178, "1L_fine": 150, "1L_coarse": 151}[key], 0), fracture_mode_class=mode)
                self.add("T3", order, f"H3_{key}_r{reg}", fam, p, reason=f"T3 scale separation (300 A cell), registry {reg}: {desc}", tags=["hierarchy", "T3", key, f"registry{reg}", "uncracked"],
                         parent={"2L": "S7_H2_veinsX_W12_ellipseX", "1L_fine": "S2_ellipse_90deg", "1L_coarse": "S4_square_D40"}[key], prediction=pred,
                         rationale=(f"column class rule (veins {col['SF_vein']:.3f} x {self.s_straight:.1f} + fine {col['SF_fine']:.3f} x {self.s_fine if key != '1L_coarse' else self.s_straight:.1f}) = {class_rule:.1f} N/m; "
                                    f"one-level trend of phase 1 at A = {g['A']:.2f}: {trend:.1f} N/m (scatter {self.scatter:.1f}). Hypothesis under test: the two-level design shows a W or crack-retention premium "
                                    f"beyond the one-level scatter ({self.W_scatter:.2f} J/m2, {self.scatter:.1f} N/m) even though its strength does not"),
                         spec_dir=SPEC_DIR_T3)
                order += 1
            # cracked pair(s): 60 A crack mid-domain in the two-level design and in the fine one-level control; 60 A crack at the cell centre of the coarse mesh (memo)
            for key, fam, p, cl, center, desc in [("2L", "nanomesh_hier", p2, 60.0, [0.5, round((50.0 + dy) / L3, 6)], "two-level mesh + 60 A crack centred mid-domain (tips 5 A from the veins)"),
                                                  ("1L_fine", "nanomesh_single", pf, 60.0, self.shifted_center((0.5, 0.5), reg, L3, L3), "one-level fine mesh + 60 A crack (the cracked control without a barrier)"),
                                                  ("1L_coarse", "nanomesh_single", pc, 60.0, self.shifted_center((0.5, 0.5), reg, L3, L3), "one-level coarse mesh + 60 A crack at the cell centre (severs one of the three load-bearing bars)")]:
                if reg == 1 and key == "1L_coarse":
                    continue
                pcr = dict(p, crack_len=cl, crack_w=3.0, crack_angle_deg=90.0, crack_center=center)
                unc = self.cache[f"H3_{key}_r{reg}"]; sf_u = unc["geometry"]["col"]["SF"]
                g = analyse(fam, pcr); col = g["col"]; ret_rule = col["SF"] / sf_u
                if key == "1L_coarse":
                    ret_rule = 2.0 / 3.0      # the 60 A crack severs one of the three 50 A bars along the load completely; the raster column rule cannot see a severed bar
                s_u = unc["prediction"]["strength_Nm_rule"]
                mech = 1.15 if key == "2L" else 1.0
                pred = dict(strength_Nm_rule=round(s_u * ret_rule, 1), strength_Nm_mechanism=round(s_u * ret_rule * mech, 1), retention_rule=round(ret_rule, 3), retention_mechanism=round(ret_rule * mech, 3),
                            work_to_failure_J_m2=round(unc["prediction"]["work_to_failure_J_m2"] * ret_rule * mech, 2), fracture_mode_class="stepwise" if key == "2L" else ("abrupt" if key == "1L_coarse" else "progressive"))
                self.add("T3", order, f"H3_{key}_L{int(cl)}_r{reg}", fam, pcr, reason=f"T3 cracked (300 A cell), registry {reg}: {desc}", tags=["hierarchy", "T3", key, f"registry{reg}", "cracked", f"L{int(cl)}"],
                         parent=f"H3_{key}_r{reg}", prediction=pred,
                         rationale=(f"net-section scaling from the uncracked rule value: SF_col {col['SF']:.3f} / {sf_u:.3f} = {col['SF']/sf_u:.3f}" if key != "1L_coarse" else "one of three load-bearing bars severed: retention 2/3")
                                   + ("; crack-arrest hypothesis: retention 15 % above the rule, crack arrested at the first vein" if key == "2L" else ""),
                         extra=dict(SF_uncracked=sf_u), spec_dir=SPEC_DIR_T3)
                order += 1


def build(dry_run=False):
    R = refs(); C = Campaign(R)
    C.T1(); C.T4(); C.T2(); C.T3()
    rows = C.rows
    # ---- table
    print(f"{'name':28s} {'fam':15s} {'N':>6s} {'phi0':>5s} {'phi':>5s} {'SFcol':>6s} {'vein':>5s} {'fine':>5s} {'minSF':>6s} {'A':>6s} {'Lcr':>5s} {'tips->vein':>11s} {'rule':>6s} {'mech':>6s}")
    for r in rows:
        g = r["geometry"]; ca = r["crack_actual"] or {}
        print(f"{r['name']:28s} {r['family']:15s} {g['n_atoms']:6d} {g['porosity_base']:5.3f} {g['porosity']:5.3f} {g['col']['SF']:6.3f} {g['col']['SF_vein']:5.3f} {g['col']['SF_fine']:5.3f} {g['minSF']:6.3f} {g['A']:+6.2f} "
              f"{ca.get('crack_len_actual', 0):5.1f} {str(r.get('tip_to_vein_A') or ''):>11s} {r['prediction'].get('strength_Nm_rule', float('nan')):6.1f} {r['prediction'].get('strength_Nm_mechanism', r['prediction'].get('strength_Nm_trend', float('nan'))):6.1f}")
    by_test = {}
    for r in rows:
        by_test.setdefault(r["test"], []).append(r)
    bad = [r["name"] for r in rows if not r["geometry"]["spanning"] or r["geometry"]["n_fragments"] != 1]
    print({t: len(v) for t, v in by_test.items()}, "designs;", len(rows), "total;", "all spanning and single-fragment" if not bad else f"NOT SPANNING / FRAGMENTED: {bad}")
    if bad:
        raise SystemExit("fix the designs above before pre-registering")
    if dry_run:
        return rows
    # ---- predictions (written before the specs)
    preds = []
    for r in rows:
        preds.append(dict(name=r["name"], test=r["test"], family=r["family"], params=r["params"], seed=r["seed"], parent=r["parent"], porosity=round(r["geometry"]["porosity"], 4),
                          porosity_base=round(r["geometry"]["porosity_base"], 4), n_atoms=r["geometry"]["n_atoms"], minSF=round(r["geometry"]["minSF"], 4), alignment_A=round(r["geometry"]["A"], 4),
                          crack_column=r["geometry"]["col"], crack_actual=r["crack_actual"], tip_to_vein_A=r.get("tip_to_vein_A"), predicted=r["prediction"], rationale=r["rationale"], tags=r["tags"]))
    notes = ("Pre-registered predictions of the hierarchy follow-up (phase 3), written before any spec was written or launched. Tests and success criteria: "
             "T1 crack arrest -- veins-mesh retention (sigma_cracked/sigma_uncracked, same registry) exceeds the net-section rule by >= 15 % and the one-level control by >= 10 % at both crack lengths and both registries, "
             "with the crack arrested at the first vein at peak load (no broken bonds inside the vein bands up to the peak). "
             "T4 two-stage failure -- residual after the first avalanche (first single-step drop > 15 % of the peak) >= 0.6 at both registries and post-peak energy >= 1.5 x the one-level control (S2_ellipse_90deg, 0.21 J/m2; the same-cell control H1_ellipse_r0/r1 once available). "
             "T2 Cook-Gordon seams -- W ratio to the cracked pristine reference (H1_pristine_L20 of the same registry) >= 1.3 with visible deflection into a seam at both registries of at least one spacing; random-pore and across-load controls < 1.1. "
             "T3 scale separation -- W or crack-retention premium of the two-level design beyond the one-level scatter (0.31 J/m2, 2.0 N/m), same sign at both registries. "
             f"Rule inputs: reference/phase1_reference_numbers.json (class medians straight {C.s_straight:.2f}, fine_round {C.s_fine:.2f} N/m; one-level trend sigma = {C.fit1[0]:.2f} A + {C.fit1[1]:.2f}). Loading: phase-1 AQS constants with max_strain 0.35 and post_peak_strain_limit 0.20.")
    path, sha = write_predictions(PRED_NAME, preds, notes)
    # ---- specs: one design per spec, file order = queue order (T1, T4, T2 in specs/hierarchy; T3 in specs/hierarchy_T3)
    out = {}
    for r in rows:
        s_ = S(r["name"], r["family"], r["params"], seed=r["seed"], reason=r["reason"], hypothesis=r["rationale"], tags=r["tags"], parent=r["parent"], prediction=r["prediction"], notes=r["notes"])
        out.setdefault(r["spec_dir"], []).append(spec(CAMPAIGN, STAGE, f"{r['order']:03d}_{r['test']}_{r['name']}", [s_], aqs=AQS_H))
    for d, batches in out.items():
        write_specs(d, batches)
    print(f"predictions: {path} (sha256 {sha[:16]}...)  specs: " + ", ".join(f"{len(b)} in {os.path.relpath(d, ROOT)}" for d, b in out.items()))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    build(dry_run=a.dry_run)
