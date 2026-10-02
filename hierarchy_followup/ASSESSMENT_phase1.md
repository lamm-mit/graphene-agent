# Assessment of the phase-1/2 results before the hierarchy follow-up

Generated 2026-09-23 14:01:49 by `analysis/assess_phase1.py` from the 132 phase-1 records copied to `reference/phase1_runs/` (record.json and stress_strain.csv of every run; the trajectories stay in the phase-1 tree). Every number below is recomputed; where the memo or the paper states the same number it is quoted for comparison.

## 1. What phase 1 and 2 established about hierarchy

Ordered meshes at porosity 0.17–0.23 (the paper's Fig. 4 selection, one-level = `nanomesh_single`, nested = `nanomesh_hier`):

| quantity | one-level | nested | memo / paper |
|---|---|---|---|
| designs | 19 | 21 | 18 / 21 (memo) |
| strength median (N/m) | 13.6 | 13.0 | 13.5 / 13.0 |
| work to failure median (J/m²) | 1.27 | 1.34 | 1.27 / 1.34 |
| hierarchy premium vs one-level trend in A (N/m) | scatter ± 1.9 | -1.6 ± 1.4 (n = 21) | -1.6 ± 1.4 (paper, n = 21) |
| one-level trend σ = a·A + b | a = 16.8, b = 13.2 (n = 19) | | slope 16.8 (paper) |

The memo's central statement is reproduced: at equal alignment index the nested meshes sit about 1.6 N/m *below* the one-level trend, with a scatter of the one-level designs of the same size; strength and work to failure medians are equal within that scatter.

## 2. Ligament-class net-section strengths (the rules' inputs)

Median of σ/minSF over the discovery designs with porosity > 0.03 (paper Fig. 2; used by every rule prediction of the follow-up):

| class | median σ/minSF, corrected minSF (rule input) | with the stored minSF | n | paper |
|---|---|---|---|---|
| straight | 28.9 | 30.0 | 10 | 30.3 (n = 9) |
| coarse_round | 29.4 | 29.8 | 10 | 29.8 (n = 10) |
| fine_round | 16.5 | 16.9 | 38 | 16.9 (n = 46) |
| random | 19.5 | 19.9 | 12 | 19.9 (n = 12) |
| bridges | 22.9 | 25.3 | 1 | — (n = —) |
| oblique | 21.8 | 22.7 | 1 | — (n = —) |
| flawed_sheet | 19.2 | 19.3 | 9 | — (n = —) |

"Corrected minSF" excludes the seam of the occupancy raster (Section 9); the stored values reproduce the paper's medians.

## 3. Reference designs for the follow-up (memo Section 3)

| record | cell (Å) | φ | σ (N/m) | Y (N/m) | ε_f | W (J/m²) | minSF stored | minSF corrected | A | mode | residual after 1st avalanche | post-peak energy (J/m²) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `S1_pristine_zz` | 39 | 0.000 | 38.8 | 250 | 0.271 | 6.73 | 1.004 | 1.004 | -0.03 | abrupt brittle | 0.01 | 0.02 |
| `S1_pristine_ac` | 38 | 0.000 | 30.7 | 287 | 0.300 | 7.05 | 1.003 | 1.003 | +0.02 | abrupt brittle | — | 3.41 |
| `S1_precrack_L10` | 101 | 0.004 | 24.1 | 245 | 0.119 | 1.55 | 0.908 | 0.908 | -0.10 | abrupt brittle | 0.04 | 0.06 |
| `S1_precrack_L20` | 101 | 0.007 | 18.5 | 235 | 0.091 | 0.89 | 0.811 | 0.811 | -0.24 | abrupt brittle | 0.06 | 0.05 |
| `S1_precrack_L30` | 101 | 0.012 | 18.7 | 224 | 0.104 | 1.05 | 0.689 | 0.689 | -0.40 | abrupt brittle | 0.06 | 0.05 |
| `S4_precrack_L30_cell150` | 150 | 0.005 | 19.4 | 239 | 0.096 | 0.99 | 0.795 | 0.795 | -0.35 | abrupt brittle | 0.07 | 0.05 |
| `S4_precrack_L40_cell150` | 150 | 0.007 | 17.9 | 234 | 0.091 | 0.87 | 0.738 | 0.738 | -0.41 | abrupt brittle | 0.07 | 0.05 |
| `S7_precrack_L50_cell150` | 150 | 0.008 | 16.1 | 225 | 0.111 | 0.81 | 0.664 | 0.667 | +0.00 | stepwise | 0.26 | 0.11 |
| `S1_hole_d20` | 101 | 0.033 | 19.1 | 230 | 0.098 | 0.99 | 0.811 | 0.811 | -0.00 | abrupt brittle | 0.04 | 0.05 |
| `S4_hole_d30` | 101 | 0.073 | 18.5 | 208 | 0.108 | 1.08 | 0.688 | 0.689 | +0.01 | abrupt brittle | 0.04 | 0.05 |
| `S2_hole20_rings2` | 101 | 0.103 | 16.2 | 178 | 0.124 | 1.02 | 0.663 | 0.658 | +0.03 | stepwise | 0.26 | 0.07 |
| `S2_hole20_background` | 101 | 0.104 | 13.2 | 171 | 0.120 | 0.92 | 0.684 | 0.689 | +0.03 | progressive | 0.26 | 0.06 |
| `S7_hole30_rings3` | 101 | 0.215 | 8.9 | 123 | 0.183 | 0.68 | 0.499 | 0.536 | -0.06 | stepwise | 0.37 | 0.19 |
| `S2_ellipse_90deg` | 102 | 0.194 | 19.0 | 171 | 0.191 | 1.94 | 0.783 | 0.786 | +0.35 | progressive | 0.45 | 0.21 |
| `S2_ellipse_0deg` | 100 | 0.203 | 8.6 | 88 | 0.236 | 1.07 | 0.596 | 0.612 | -0.24 | progressive | 0.52 | 0.37 |
| `S7_H2_veinsX_W12_ellipseX` | 122 | 0.206 | 19.4 | 178 | 0.221 | 1.98 | 0.802 | 0.821 | +0.58 | stepwise | 0.72 | 0.36 |
| `S5_H2_D40_W8_ellipseX` | 122 | 0.208 | 17.5 | 174 | 0.259 | 2.02 | 0.757 | 0.767 | +0.37 | progressive | 0.42 | 0.60 |
| `S2_slit_0deg` | 102 | 0.205 | 25.4 | 200 | 0.219 | 3.25 | 0.800 | 0.796 | +0.73 | stepwise | 0.20 | 0.07 |
| `S2_slit_90deg` | 100 | 0.187 | 4.0 | 43 | 0.281 | 0.62 | 0.157 | 0.173 | -0.62 | progressive | 0.66 | 0.25 |
| `S4_square_D40` | 121 | 0.202 | 17.0 | 151 | 0.170 | 1.76 | 0.552 | 0.562 | +0.03 | abrupt brittle | 0.03 | 0.04 |
| `S7_square_D40_staggered` | 121 | 0.202 | 17.7 | 150 | 0.180 | 1.94 | 0.690 | 0.696 | +0.04 | abrupt brittle | 0.05 | 0.05 |
| `S2_H1_p16` | 121 | 0.207 | 11.7 | 123 | 0.133 | 0.89 | 0.681 | 0.729 | +0.08 | progressive | 0.79 | 0.10 |
| `S2_H1_p12` | 121 | 0.199 | 13.3 | 124 | 0.221 | 1.45 | 0.724 | 0.775 | +0.08 | progressive | 0.40 | 0.31 |
| `S2_H1_p32` | 121 | 0.203 | 16.4 | 146 | 0.264 | 1.94 | 0.739 | 0.767 | +0.08 | stepwise | 0.24 | 0.29 |
| `S2_H2_D40_W8` | 121 | 0.202 | 12.8 | 132 | 0.206 | 1.33 | 0.769 | 0.771 | +0.13 | progressive | 0.68 | 0.41 |
| `S4_H2_D40_W8_veins_x` | 121 | 0.198 | 16.0 | 137 | 0.189 | 1.63 | 0.783 | 0.796 | +0.19 | progressive | 0.51 | 0.14 |
| `S4_H2_D40_W8_veins_y` | 121 | 0.198 | 12.0 | 126 | 0.181 | 1.03 | 0.718 | 0.733 | +0.01 | progressive | 0.20 | 0.22 |
| `S5_H2_D40_W16` | 121 | 0.195 | 14.8 | 139 | 0.163 | 1.36 | 0.612 | 0.641 | +0.10 | progressive | 0.79 | 0.18 |
| `S5_H2_D40_W20` | 121 | 0.197 | 15.6 | 150 | 0.153 | 1.34 | 0.590 | 0.608 | +0.04 | abrupt brittle | 0.24 | 0.06 |
| `S2_H2_D40_W12` | 121 | 0.197 | 13.7 | 145 | 0.213 | 1.40 | 0.739 | 0.750 | +0.13 | progressive | 0.45 | 0.38 |
| `S5_H2_D60_W12` | 121 | 0.195 | 13.0 | 137 | 0.149 | 1.17 | 0.764 | 0.779 | +0.10 | abrupt brittle | 0.33 | 0.06 |

Checks against the memo's table: pristine 38.8 ✓, precrack L10/L20/L30 24.1/18.5/18.7 ✓, L30/L40/L50 in 150 Å cells 19.4/17.9/16.1 ✓, `S2_ellipse_90deg` 19.0 (W 1.94, A = +0.35) ✓, `S7_H2_veinsX_W12_ellipseX` 19.4 (W 1.98, A = +0.59) ✓, `S2_slit_0deg` 25.4 (W 3.25) ✓, `S4_square_D40` 17.0 ✓, `S2_H1_p16`/`S2_H1_p12` 11.7/13.3 ✓, `S2_H2_D40_W8` 12.8 ✓. The memo's record names `S4_precrack_L40` and `S7_precrack_L50` are `S4_precrack_L40_cell150` and `S7_precrack_L50_cell150` in the database.

## 4. Flaw tolerance of the pristine sheet (the T1 yardstick)

| crack (Å) | cell (Å) | σ_cracked/σ_pristine | net section |
|---|---|---|---|
| 10 | 101 | 0.62 | 0.91 |
| 20 | 101 | 0.48 | 0.81 |
| 30 | 101 | 0.48 | 0.69 |
| 30 | 150 | 0.50 | 0.79 |
| 40 | 150 | 0.46 | 0.74 |
| 50 | 150 | 0.42 | 0.67 |

The pristine sheet is strongly notch-sensitive (a 20 Å crack halves the strength although it removes a fifth of the section), and the plateau of the 20–50 Å cracks at 16–19 N/m is the lattice-trapping regime the phase-1 report described. A porous mesh has no such sharp-tip amplification available (its ligaments are already at the pore scale), which is why the memo expects every porous design to be less notch-sensitive than the pristine sheet. The 100 → 150 Å cell change moves the 30 Å result by 4 % (18.7 → 19.4 N/m); the follow-up therefore keeps every T1 comparison inside one cell size (120 Å) and treats differences below 1 N/m as noise.

## 5. Two-stage failure in the existing data (the T4 baseline)

Curve metrics with the definitions fixed in `curve_metrics` (first avalanche = first single-step stress drop > 15 % of the peak):

| level | designs | with an avalanche | residual after it, median | residual ≥ 0.6 | second peak ≥ 0.6 | post-peak energy median (J/m²) | post-peak energy max | post-peak strain range median | tails truncated by the 0.12 limit |
|---|---|---|---|---|---|---|---|---|---|
| one | 19 | 19 | 0.40 | 5 | 2 of 8 with any second peak | 0.12 | 0.62 | 0.030 | 2 |
| nested | 21 | 21 | 0.58 | 10 | 1 of 11 with any second peak | 0.31 | 0.60 | 0.060 | 6 |

Reading the post-peak stress sequences of the nested designs shows what "two-stage" means in this model so far: after the first avalanche the stress falls in a staircase (e.g. `S7_H2_veinsX_W12_ellipseX`: 1.00 → 0.72 → 0.61 → 0.47 → 0.25 of the peak over four strain steps of 0.5 %), i.e. the veins keep carrying load for a few steps but never rise to a second maximum. The wide-vein square-vein designs of phase 1 already tested vein fractions 0.4 and 0.5 (`S5_H2_D40_W16`: 14.8 N/m, residual 0.79 then fractured within 0.025 strain; `S5_H2_D40_W20`: 15.6 N/m, single avalanche, residual 0.24): the veins did not survive the failure of the fine level. A high residual right after the first avalanche is therefore not by itself a signature of hierarchy (one-level meshes reach 0.7–0.8 too, when one row of ligaments fails first); the discriminating quantities are the energy and the strain range carried after the avalanche. T4 is pre-registered on both.

Consequences for the follow-up design: (i) the loading protocol keeps every physics constant of phase 1 but extends the post-peak termination bound from 0.12 to 0.20 strain beyond the peak (three phase-1 nested tails were cut by it), so that a second stage cannot be truncated; (ii) the fine level of the T4 designs uses round pores: aligned elliptical pores fail at 0.15–0.19 strain, close to the 0.20 of straight veins, whereas round-pore fine levels fail at 0.12–0.13, which gives the two stages the largest strain separation; elliptical pores at the fine porosity that T4 needs inside the domains (0.33–0.40) would also merge into slits (tip ligaments < 2 Å).

## 6. Already answered, not repeated

- Isotropy: `S4_square_D40` 17.0 N/m (symmetric) against the best nested design under 0°/90° loading (veins-x/veins-y 16.0/12.0); elongated one-level pores 19.0/8.6; slits 25.4/4.0.
- Flaw halos (pores around a hole) cost strength: `S2_hole20_rings2` 16.2 vs bare hole 19.1 N/m; the halo control with the same porosity in the far field 13.2. T1 puts solid across the crack path instead of pores around it.
- No geometric descriptor predicts avalanche size or post-peak retention across the 79 matched-porosity designs (paper); T4 tests a designed two-stage failure instead.

## 7. Registry replicates: one of the three phase-1 offsets is a null replicate

Phase 1 replicated the ordered p16 mesh by shifting the pore lattice by (1.23, 0), (0, 2.13) and (1.23, 2.13) Å (`S6_mesh_p16_offset1/2/3`). The last vector is (a/2, √3a/2) with a = 2.46 Å, i.e. a primitive translation of the honeycomb lattice: the shifted pattern removes the same atoms as offset 1 up to a lattice translation, and the database shows it (`S6_mesh_p16_offset1` 13.13 N/m, `S6_mesh_p16_offset3` 13.13 N/m: identical to two decimals, whereas offset 2 gives 13.26). Only two distinct registries were therefore run; the registry scatter quoted in the memo ("up to 2 N/m") comes from other pairs. The follow-up uses a rigid shift of the whole pattern (pores, veins, slits and crack) by (1.23, 0) Å, which is not a lattice translation (checked in `tests/test_followup.py`).

## 8. Budget check (measured 2026-09-23 in the new environment, torch 2.14, MPS float32)

One energy + force + virial evaluation: 13.8 ms for the 4,330-atom T1 veins mesh (3.2 µs/atom), 15.6 ms for the 5,046-atom T2 seam sheet, 74 ms for the 27,171-atom T3 two-level mesh (2.7 µs/atom, 3.3 GB of GPU memory). The memo measured 69 / 285 ms for the same cells with torch 2.7, so the process-hour estimates of the memo (25 + 8 + 12 + 60–100) are upper bounds by a factor of about four.

## 9. A seam artefact in the phase-1 minimum-section descriptor

`min_solid_fraction_across_x` takes the minimum over the columns of a 0.5 Å occupancy raster that is `ceil(Lx/0.5)·0.5` long, i.e. up to one column longer than the periodic cell. Its last column lies mostly outside the cell (for the 120.55 Å cells: 0.45 of its 0.5 Å) and is painted only by the discs of neighbouring atoms, so it reads emptier than any real column. For every 120 Å design of phase 1 the stored minimum is that seam column, not a load-bearing section: `S7_H2_veinsX_W12_ellipseX` stored 0.802 against 0.876 for its weakest interior column; the ellipse mesh in the 120 Å cell of this campaign 0.721 against 0.855. In 100 Å cells (Lx = 100.9 Å, seam sliver 0.1 Å) the effect is small.

| cell | designs | mean stored minSF | mean corrected minSF | mean difference |
|---|---|---|---|---|
| 120 Å | 17 | 0.671 | 0.686 | +0.015 |
| other | 72 | 0.677 | 0.686 | +0.010 |

Consequences: (i) the follow-up computes every section (crack column and minimum column) with the seam excluded (`atomistics/descriptors/columns.py`) and uses the class medians recomputed with corrected sections as rule inputs (Section 2); (ii) the paper's Fig. 2 predictor (minSF × class median) and the Stage-7 holdout predictions used the stored values; because the same contaminated values enter both the medians and the predictions, the bias largely cancels within a cell size but not between 100 Å and 120 Å cells. This should be re-examined in the phase-1 tree (not touched here). The pre-registered rules of this campaign were corrected for it before any porous design had been launched (see the README status log).
