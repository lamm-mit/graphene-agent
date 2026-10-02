"""Tests of the phase-3 additions: phase-1 designs regenerate unchanged, cracks and seams behave as documented."""
import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pytest
from atomistics.structures import design_space as D
from experiments import db


def test_phase1_designs_regenerate_identically():
    recs = [r for r in db.reference_records() if r.get("status") == "completed" and r["family"] != "repeat"]
    assert len(recs) >= 120
    for r in recs:
        params = dict(r["params"])
        if r.get("seed") is not None and r["family"] != "pristine":
            params["seed"] = r["seed"]
        a = D.generate(r["family"], **params)
        assert len(a) == r["n_atoms"], r["name"]
        assert abs(a.info["porosity"] - r["porosity"]) < 1e-9, r["name"]


def test_precrack_family_untouched():
    a = D.generate("precrack", Lx=100, Ly=100, crack_len=20, crack_w=3.0, angle_deg=90)
    assert len(a) == 3744 and "cracked" not in a.info["design"]


def test_add_crack_geometry_and_porosity():
    base = D.generate("nanomesh_hier", Lx=120, Ly=120, levels=2, period=12, domain=40, vein_w=12, vein_dirs="x", aspect=2.0, angle_deg=90, pore_d=4.812)
    a = D.generate("nanomesh_hier", Lx=120, Ly=120, levels=2, period=12, domain=40, vein_w=12, vein_dirs="x", aspect=2.0, angle_deg=90, pore_d=4.812, crack_len=20, crack_center=(0.5, 20 / 120))
    d = a.info["design"]; ca = a.info["crack_actual"]
    assert d["cracked"] and abs(d["porosity_base"] - base.info["porosity"]) < 1e-12
    assert a.info["n_initial"] == base.info["n_initial"] and abs(a.info["porosity"] - (1 - len(a) / base.info["n_initial"])) < 1e-12
    assert 18 < ca["crack_len_actual"] < 24 and abs(ca["tip_lo"][0] - 60.27) < 1.0
    assert 9 < ca["tip_lo"][1] < 12 and 29 < ca["tip_hi"][1] < 32      # tips 4 A (nominal) from the vein edges at y = 6 and 34
    assert len(a) < len(base)


def test_seam_pores_rows_and_controls():
    a = D.generate("seam_pores", Lx=120, Ly=120, pore_d=4, pitch=8, seam_spacing=20, seam_dir="x", skip_near=(60, 60, 8, 10))
    d = a.info["design"]
    assert d["n_rows"] == 6 and d["n_pores"] == 6 * 15 - 2 and abs(d["seam_spacing"] - 20.0) < 0.3
    assert 0.06 < a.info["porosity"] < 0.10
    b = D.generate("seam_pores", Lx=120, Ly=120, pore_d=4, pitch=8, seam_spacing=20, seam_dir="x", placement="random", seed=1, skip_near=(60, 60, 8, 10))
    assert b.info["design"]["n_pores"] == d["n_pores"]
    c = D.generate("seam_pores", Lx=120, Ly=120, pore_d=4, pitch=8, seam_spacing=20, seam_dir="y", skip_near=(60, 60, 8, 10))
    assert c.info["design"]["n_rows"] == 6
    # rows anchored at the crack centre, pores flank the crack line
    from simulation.topology import spanning_x
    from atomistics.descriptors.descriptors import bond_graph
    for s in (a, b, c):
        i, j, S, dd = bond_graph(s)
        assert spanning_x(i, j, S, len(s))


def test_slit_offset_registry():
    a = D.generate("slit_array", Lx=100, Ly=100, slit_w=4.0, period_x=30, period_y=16, angle_deg=0, slit_len=26.5)
    b = D.generate("slit_array", Lx=100, Ly=100, slit_w=4.0, period_x=30, period_y=16, angle_deg=0, slit_len=26.5, offset=(1.23, 2.13))
    assert len(a) == 3000 and abs(len(b) - 3000) < 60 and a.info["design"]["offset"] == [0.0, 0.0]


def test_reference_curves_and_metrics():
    from analysis.assess_phase1 import curve_metrics
    recs = {r["name"]: r for r in db.reference_records()}
    m = curve_metrics(recs["S2_slit_0deg"])
    assert abs(m["peak_Nm"] - 25.4) < 0.1 and m["residual_after_first_avalanche"] is not None and m["residual_after_first_avalanche"] < 0.3
    m = curve_metrics(recs["S4_square_D40"])
    assert m["residual_after_first_avalanche"] < 0.1


def _is_translated_copy(a, b, shift, tol=0.05):
    """True if structure b equals structure a shifted by `shift` (periodic), atom for atom."""
    from scipy.spatial import cKDTree
    if len(a) != len(b):
        return False
    L = np.array([a.cell[0, 0], a.cell[1, 1]])
    pa = a.positions[:, :2] % L; pb = (b.positions[:, :2] - np.asarray(shift)) % L
    tree = cKDTree(np.concatenate([pa + [i * L[0], j * L[1]] for i in (-1, 0, 1) for j in (-1, 0, 1)]))
    d, _ = tree.query(pb)
    return bool(d.max() < tol)


def test_registry_offsets_are_not_lattice_translations():
    """(1.23, 2.13) is a primitive translation of the honeycomb lattice: the shifted pattern removes the same atoms (phase-1 S6 offset3 = offset1)."""
    base = dict(Lx=100, Ly=100, period=16, pore_d=7.0)
    a = D.generate("nanomesh_single", **base); b = D.generate("nanomesh_single", offset=(1.23, 2.13), **base); c = D.generate("nanomesh_single", offset=(1.23, 0.0), **base)
    assert _is_translated_copy(a, b, (1.23, 2.13))          # identical structure up to the translation
    assert not _is_translated_copy(a, c, (1.23, 0.0))       # a genuine registry change
