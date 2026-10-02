"""Paper sweeps (campaign "paper", stage "paper_sweeps"): additional simulations that turn three campaign
findings into controlled, pre-registered tests.

  P1  slit-array angle sweep at porosity 0.20: angles 5, 10, 15, 25, 30, 35, 60, 75 deg (0, 20, 45, 90 exist)
  P2  tip-overlap controls at 20 deg: porosity 0.10 (slits too short to overlap) and 0.30 (larger overlap)
  P3  alignment sweep of elongated fine pores (aspect 2) at porosity 0.20:
        single-level mesh          angle 30, 60 deg (0 and 90 exist)
        two-level, veins along x   angle 0, 30, 60 deg (90 exists: S7_H2_veinsX_W12_ellipseX)
        two-level, square veins    angle 0, 30, 60 deg (90 exists: S5_H2_D40_W8_ellipseX)
      (generator convention: angle_deg = 90 puts the long axis of the ellipse along the load)

For every design the geometry is analysed before simulation (tip overlap O, vertical clearance t, minimum
load-bearing solid fraction, load-path alignment index A) and quantitative predictions from two competing
rules are written to a timestamped, hashed file BEFORE the specs are launched:
  rule  : net-section rule, sigma = minSF x sigma_net(straight ligaments)  (the Stage 7 predictor)
  mech  : the same rule with the en-echelon knock-down when tips overlap (O > 0) and the slits are inclined.
"""
from __future__ import annotations
import sys, os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from atomistics.structures import design_space as D
from atomistics.descriptors.descriptors import compute_descriptors
from experiments import db
from experiments.campaign_design import AQS, S, spec, match
from experiments.stage_tools import write_predictions, write_specs

SPEC_DIR = os.path.join(ROOT, "experiments", "specs", "paper")
CAMPAIGN, STAGE = "paper", "paper_sweeps"


def alignment_index(desc):
    """Signed load-path alignment A = a cos(2 phi): a = anisotropy of the structure tensor of the solid phase,
    phi = ligament orientation relative to the load axis.  +1 all ligaments along the load, -1 all across, 0 isotropic."""
    return float(desc["anisotropy_index"] * math.cos(2.0 * math.radians(desc["ligament_orientation_deg"])))


def slit_geometry(params, design):
    """Tip overlap along the load O = L cos(theta) - s (s = stagger = half the slit period along x) and the
    vertical clearance between the tips of adjacent rows t = p_y - L sin(theta)."""
    Lx, Ly = design["Lx"], design["Ly"]
    nx = max(1, int(round(Lx / params["period_x"]))); ny = max(1, int(round(Ly / params["period_y"])))
    px, py = Lx / nx, Ly / ny; s = 0.5 * px
    th = math.radians(params["angle_deg"]); L = params["slit_len"]
    return dict(O_A=L * math.cos(th) - s, t_A=py - L * math.sin(th), px_A=px, py_A=py, L_A=L)


def sigma_net_straight():
    """Median net-section strength of the completed load-parallel slit arrays (Stage 2/5 records)."""
    vals = []
    for r in db.all_records():
        if r.get("family") == "slit_array" and r.get("campaign") == "discovery" and abs(r["params"].get("angle_deg", 0)) < 1e-6 and r["params"].get("orientation", "zigzag") == "zigzag":
            msf = r["descriptors"]["min_solid_fraction_across_x"]
            vals.append(r["metrics"]["strength_Nm"] / msf)
    return float(np.median(vals)), len(vals)


def build():
    snet, n_ref = sigma_net_straight()
    designs, preds = [], []

    def add(batch, name, family, params, key, lo, hi, target, reason, tags, series):
        p, phi, n = match(family, params, target, key, lo, hi)
        a = D.generate(family, **p); d = compute_descriptors(a, a.info.get("design"))
        row = dict(batch=batch, name=name, family=family, params=p, phi=phi, n_atoms=n, msf=d["min_solid_fraction_across_x"],
                   A=alignment_index(d), reason=reason, tags=tags, series=series)
        if family == "slit_array":
            row.update(slit_geometry(p, a.info["design"]))
        designs.append(row); return row

    # ---- P1: angle sweep at phi = 0.20
    for ang in (5, 10, 15, 25, 30, 35, 60, 75):
        add("paper_P1_slit_angles", f"P1_slit_{ang:02d}deg", "slit_array",
            dict(Lx=100, Ly=100, slit_w=4.0, period_x=30, period_y=16, angle_deg=ang), "slit_len", 8.0, 45.0, 0.20,
            "angle sweep of the slit array at matched porosity (alignment cliff)", ["paper", "slit_angle"], "angle_sweep")
    # ---- P2: overlap controls at 20 deg
    for phi in (0.10, 0.30):
        add("paper_P2_slit_overlap", f"P2_slit_20deg_phi{phi:.2f}", "slit_array",
            dict(Lx=100, Ly=100, slit_w=4.0, period_x=30, period_y=16, angle_deg=20), "slit_len", 5.0, 60.0, phi,
            "tip-overlap control at 20 deg: porosity sets the slit length and hence whether tips of adjacent rows overlap", ["paper", "slit_overlap"], "overlap_control")
    # ---- P3: alignment sweep of elongated fine pores
    for ang in (30, 60):
        add("paper_P3_alignment", f"P3_H1_ellipse_{ang:02d}deg", "nanomesh_single",
            dict(Lx=100, Ly=100, period=16, aspect=2.0, angle_deg=ang), "pore_d", 3.0, 9.0, 0.20,
            "single-level mesh, elongated pores rotated (alignment sweep)", ["paper", "alignment", "H1"], "H1_ellipse")
    for ang in (0, 30, 60):
        add("paper_P3_alignment", f"P3_H2_veinsX_W12_ellipse_{ang:02d}deg", "nanomesh_hier",
            dict(Lx=120, Ly=120, levels=2, period=12, domain=40, vein_w=12, vein_dirs="x", aspect=2.0, angle_deg=ang), "pore_d", 3.0, 8.0, 0.20,
            "two-level mesh with veins along the load, elongated fine pores rotated (alignment sweep)", ["paper", "alignment", "H2", "veins_x"], "H2_veinsX_ellipse")
    for ang in (0, 30, 60):
        add("paper_P3_alignment", f"P3_H2_square_W8_ellipse_{ang:02d}deg", "nanomesh_hier",
            dict(Lx=120, Ly=120, levels=2, period=12, domain=40, vein_w=8, vein_dirs="xy", aspect=2.0, angle_deg=ang), "pore_d", 3.0, 8.0, 0.20,
            "two-level mesh with square veins, elongated fine pores rotated (alignment sweep)", ["paper", "alignment", "H2", "veins_xy"], "H2_square_ellipse")

    # ---- predictions (written before launch)
    # anchors for the alignment series (existing records)
    R = {r["name"]: r for r in db.all_records()}
    def sig(n): return R[n]["metrics"]["strength_Nm"]
    def A_of(n): return alignment_index(R[n]["descriptors"])
    anchors = {"H1_ellipse": [("S2_ellipse_0deg", A_of("S2_ellipse_0deg"), sig("S2_ellipse_0deg")), ("S2_ellipse_90deg", A_of("S2_ellipse_90deg"), sig("S2_ellipse_90deg"))],
               "H2_veinsX_ellipse": [("S4_H2_D40_W8_veins_x", A_of("S4_H2_D40_W8_veins_x"), sig("S4_H2_D40_W8_veins_x")), ("S7_H2_veinsX_W12_ellipseX", A_of("S7_H2_veinsX_W12_ellipseX"), sig("S7_H2_veinsX_W12_ellipseX"))],
               "H2_square_ellipse": [("S2_H2_D40_W8", A_of("S2_H2_D40_W8"), sig("S2_H2_D40_W8")), ("S5_H2_D40_W8_ellipseX", A_of("S5_H2_D40_W8_ellipseX"), sig("S5_H2_D40_W8_ellipseX"))]}
    for row in designs:
        if row["family"] == "slit_array":
            rule = row["msf"] * snet
            th = row["params"]["angle_deg"]; O = row["O_A"]
            if O > 0 and th >= 5:
                mech = rule * max(0.38, 1.0 - 0.62 * min(1.0, th / 20.0))
                why = f"tips of adjacent rows overlap (O = {O:.1f} A) and the slits are inclined: en-echelon linking, knock-down growing with the angle up to the 20-deg level"
            else:
                mech = 0.9 * rule
                why = f"no tip overlap (O = {O:.1f} A): isolated inclined slits, net-section rule with a mild stress-concentration knock-down"
            row["pred_rule"] = round(rule, 1); row["pred_mech"] = round(mech, 1); row["why"] = why
        else:
            (n0, A0, s0), (n1, A1, s1) = anchors[row["series"]]
            f = (row["A"] - A0) / (A1 - A0) if abs(A1 - A0) > 1e-9 else 0.5
            mech = s0 + f * (s1 - s0); rule = row["msf"] * snet * 0.55   # fine round-pore ligaments: ~55% of straight ligament strength
            row["pred_rule"] = round(rule, 1); row["pred_mech"] = round(mech, 1)
            row["why"] = f"linear in the alignment index A between the series anchors {n0} (A={A0:.2f}, {s0:.1f} N/m) and {n1} (A={A1:.2f}, {s1:.1f} N/m); hypothesis: no hierarchy premium at equal A"
        preds.append(dict(name=row["name"], family=row["family"], params=row["params"], porosity=round(row["phi"], 4), minSF=round(row["msf"], 4), alignment_A=round(row["A"], 4),
                          slit_tip_overlap_A=round(row.get("O_A", float("nan")), 2), slit_row_clearance_A=round(row.get("t_A", float("nan")), 2),
                          predicted={"strength_Nm_rule": row["pred_rule"], "strength_Nm_mechanism": row["pred_mech"]}, rationale=row["why"], series=row["series"]))
    notes = (f"Pre-registered predictions for the paper sweeps, written before the specs were launched. sigma_net(straight) = {snet:.2f} N/m "
             f"(median of {n_ref} load-parallel zigzag slit arrays). Two rules are recorded: the net-section rule of the Stage 7 predictor and the "
             f"mechanism rule with the en-echelon knock-down when slit tips of adjacent rows overlap (O > 0) and the slits are inclined. "
             f"Alignment series: linear interpolation in the alignment index A between the existing anchors of each series; hypothesis under test: no hierarchy premium at equal A.")
    path, sha = write_predictions("paper_sweeps_predictions", preds, notes)
    # ---- specs (one batch per group; run_batch handles a batch sequentially, the queue runs batches concurrently)
    batches = {}
    for row in designs:
        batches.setdefault(row["batch"], []).append(S(row["name"], row["family"], row["params"], seed=row["params"].get("seed", 0), reason=row["reason"], tags=row["tags"],
                                                     prediction={"strength_Nm_rule": row["pred_rule"], "strength_Nm_mechanism": row["pred_mech"]}, notes=row["why"]))
    # split every batch into single-structure batches so that 4 workers can share the work evenly
    out = []
    for b, structs in batches.items():
        for k, s_ in enumerate(structs):
            out.append(spec(CAMPAIGN, STAGE, f"{b}_{k:02d}", [s_], aqs=AQS))
    write_specs(SPEC_DIR, out)
    print(f"predictions: {path} (sha256 {sha[:16]}...)  specs: {len(out)} in {SPEC_DIR}")
    print(f"{'name':38s} {'phi':>5s} {'N':>5s} {'minSF':>6s} {'A':>6s} {'O(A)':>6s} {'t(A)':>6s} {'rule':>6s} {'mech':>6s}")
    for row in designs:
        print(f"{row['name']:38s} {row['phi']:5.3f} {row['n_atoms']:5d} {row['msf']:6.3f} {row['A']:6.3f} {row.get('O_A', float('nan')):6.1f} {row.get('t_A', float('nan')):6.1f} {row['pred_rule']:6.1f} {row['pred_mech']:6.1f}")
    return designs


if __name__ == "__main__":
    build()
