"""Comparison panels for the two most informative Stage 7 holdout results:
(1) slit arrays parallel to the load (Stage 2) versus rotated by 20 degrees (holdout; en-echelon coalescence),
(2) nested mesh with load-elongated fine pores and square veins (Stage 5) versus the holdout that additionally
    aligns the veins with the load (synergy under-predicted by the data-driven model)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from experiments.db import all_records
from analysis.comparison_panels import comparison

by = {r["name"]: r for r in all_records() if r.get("metrics")}
comparison(by["S2_slit_0deg"]["run_id"], by["S7_slit_20deg"]["run_id"], "slit_orientation_echelon",
           "slit array parallel to the load (Stage 2, 25.4 N/m)", "slit array rotated by 20 deg (holdout, 8.9 N/m)",
           caption="Holdout: rotating a parallel slit array by 20 degrees reduces the strength from 25.4 to 8.9 N/m (predicted 23.9 N/m): the slit tips of neighbouring rows now overlap along the load and the ligaments between them fail by en-echelon crack coalescence, a mechanism absent from the training set. Rows: relaxed, first damage, peak load, post-peak, final; colour: energy change per atom; bottom: stress-strain curves with the selected events marked.")
comparison(by["S5_H2_D40_W8_ellipseX"]["run_id"], by["S7_H2_veinsX_W12_ellipseX"]["run_id"], "aligned_hierarchy_synergy",
           "H2: load-elongated fine pores, square veins (Stage 5, 17.5 N/m)", "H2: load-elongated fine pores + load-parallel veins (holdout, 19.4 N/m)",
           caption="Holdout: combining the two orientation effects (load-elongated fine pores and veins along the load only) gives the strongest nested design of the campaign (19.4 N/m, predicted 13.3 N/m by the data-driven predictor): the two alignments act on different length scales and their benefits add. Rows: relaxed, first damage, peak load, post-peak, final; colour: energy change per atom; bottom: stress-strain curves with the selected events marked.")
print("stage7 panels written")
