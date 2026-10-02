# Phase-II loading-extension results

Updated 2026-09-26T06:53:06.817374+00:00. **Interim results; final scientific review pending.**

34 of 34 extensions, 2 of 2 controls, and 4 of 4 resolution checks currently have paired results.

A = archived original; B = extended rerun truncated at its original stopping rule; C = extended endpoint. B-to-C measures the added observation window along the same computed trajectory. Its physical interpretation remains subject to numerical convergence. All integrals use the original stress/engineering-strain convention.

| Source phase | Available / planned | Driver failure detected | Failure criterion unmet | Residual quality flags | Reproduction flags |
|---|---:|---:|---:|---:|---:|
| I | 20 / 20 | 20 | 0 | 19 | 14 |
| II | 14 / 14 | 13 | 1 | 14 | 11 |

All six pilot extensions and both controls have been reviewed. Production is released under `protocol/production_gate.json` for the original-setting protocol-sensitivity comparison and four planned resolution checks. This decision does not certify the trajectories as numerically converged or authorize manuscript integration. The separate Q4/Q7/Q8 branch shows unresolved increment sensitivity despite passing saved-state residual checks.

## Strength and integrated response: separate comparisons

Extending the same recorded trajectory changes its maximum stress only when a later higher peak occurs. Its integral can increase even when the maximum remains unchanged. A-to-B changes arise before the added loading window and must not be attributed to extension. Percentages below describe computed outcomes with the numerical flags retained; they are not uncertainty bounds or validated material constants.

| Completed extension | Peak A-to-B (%) | Peak B-to-C (%) | Integral A-to-B (%) | Integral B-to-C (%) |
|---|---:|---:|---:|---:|
| S2_slit_45deg | -34.18 | +0.00 | -0.72 | +86.69 |
| S7_slit_20deg | +0.00 | +67.77 | -1.71 | +300.84 |
| S7_slit_0deg_armchair | +0.00 | +0.00 | -8.61 | +23.44 |
| S5_H2_D40_W8_ellipseX | +0.31 | +0.00 | +6.16 | +41.44 |
| P1_slit_35deg | +11.71 | +2.47 | +5.73 | +43.27 |
| P3_H2_square_W8_ellipse_30deg | -0.27 | +0.00 | -1.06 | +19.59 |
| P1_slit_05deg | -0.18 | +0.00 | -8.37 | +0.00 |
| P1_slit_10deg | -0.10 | +0.00 | +8.87 | +7.62 |
| P1_slit_15deg | -2.97 | +0.00 | -10.02 | +0.00 |
| P1_slit_25deg | +2.39 | +0.00 | +9.22 | +0.00 |
| P1_slit_30deg | +25.91 | +0.00 | +71.74 | +97.98 |
| P1_slit_60deg | -14.03 | +0.00 | -7.86 | +62.99 |
| P1_slit_75deg | +0.07 | +0.00 | -4.50 | +39.49 |
| P2_slit_20deg_phi0.10 | -0.01 | +0.00 | +0.22 | +6.90 |
| P2_slit_20deg_phi0.30 | -31.43 | +0.00 | -27.77 | +95.57 |
| P3_H1_ellipse_30deg | -0.28 | +0.00 | -1.69 | +5.66 |
| P3_H2_square_W8_ellipse_00deg | -0.05 | +0.00 | +3.29 | +7.35 |
| P3_H2_veinsX_W12_ellipse_30deg | -1.38 | +0.00 | -13.78 | +0.00 |
| S2_H2_D40_W4 | +0.10 | +0.00 | -6.11 | +4.22 |
| S2_mesh_jitter2.0 | +0.12 | +0.00 | -2.83 | +10.00 |
| S2_mesh_jitter3.0 | +0.13 | +0.00 | -22.92 | +0.00 |
| S2_mesh_p16_armchair | +0.11 | +0.00 | -11.47 | +0.00 |
| S2_mesh_phi0.4 | -0.06 | +0.00 | +1.79 | +22.86 |
| S2_mesh_sizedis0.3 | +0.19 | +0.00 | +4.00 | +5.99 |
| S2_slit_90deg | +0.26 | +0.00 | -12.67 | +0.00 |
| S2_strut_090 | +0.01 | +0.00 | -0.68 | +28.19 |
| S2_voronoi_n25_reg0.0 | -0.09 | +0.00 | +0.60 | +21.09 |
| S2_voronoi_n25_reg0.6 | +0.21 | +0.00 | +0.54 | +11.99 |
| S5_H2_D40_W8_jitter2 | +2.27 | +0.00 | +0.87 | +3.86 |
| S5_H2_p8_D40_W8 | +0.01 | +0.00 | -8.61 | +0.00 |
| S6_voronoi_n25_reg0.0_s2 | +0.20 | +0.00 | -1.05 | +6.44 |
| S7_mesh_p24_phi0.3 | -1.37 | +0.00 | +3.78 | +12.16 |
| S7_strut_090_phi0.3 | -0.52 | +0.00 | -3.82 | +0.00 |
| S7_voronoi_n35_reg0.8_s7 | +0.25 | +0.00 | -4.10 | +15.03 |

For P1_slit_05deg, P1_slit_15deg, P1_slit_25deg, P3_H2_veinsX_W12_ellipse_30deg, S2_mesh_jitter3.0, S2_mesh_p16_armchair, S2_slit_90deg, S5_H2_p8_D40_W8, S7_strut_090_phi0.3, B and C coincide: the rerun meets the driver failure criterion before an administrative cutoff. Its zero B-to-C change means no added loading interval occurred; it does not impute a missing result or imply reproduction of A. Archive-to-rerun differences are reported separately.

### Completed executions with an unmet failure criterion

These runs remain censored after extension. Their ending strain and recorded integral must not be substituted for detected failure strain or full-failure work. The nominal strain cap is evaluated after an accepted increment, so the recorded endpoint can slightly exceed the cap. Numerical quality flags remain separate from censoring.

| Case | C ending strain | Stop reason | C load / running maximum (%) | Spanning at C | Residual quality flag |
|---|---:|---|---:|---|---|
| P1_slit_30deg | 1.00375 | max strain reached | 17.34 | True | True |

### Geometry and boundary-condition interpretation

A separate saved-configuration audit of 11 completed slit extensions found every inspected frame exactly planar. The original setup uses one carbon monolayer, periodicity in x and y, and a nonperiodic out-of-plane direction. The initially flat athermal trajectory has no imposed out-of-plane perturbation; the rectangular cell does not relax macroscopic shear. These responses do not test stability against out-of-plane buckling or an alternative free-edge or shear-relaxed setup. The primary extensions retain those original conditions. The audit covers its dated inventory, not subsequent uninspected cases. See `results/geometry_audit_20260925T083357Z/README.md` and `audit.json`.

### Completed example: S2_slit_45deg

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.32375 | 0.32125 | 0.66125 |
| Maximum recorded stress (N/m) | 18.5430 | 12.2046 | 12.2046 |
| Recorded integral (J/m²) | 1.3103 | 1.3009 | 2.4287 |

The same-rerun B-to-C integral change is +86.69%. C terminated by **stress collapsed**; spanning connectivity at C was **True**, and the residual load was 8.66% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 25 saved frames above the force tolerance and 11 above the transverse-stress tolerance. Reproduction flags: maximum_stress_relative exceeds 2% diagnostic trigger. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.2780 eV/Å and transverse stress -0.0264 N/m; endpoint residual checks do not pass.

### Completed example: S7_slit_20deg

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.19500 | 0.19500 | 0.82000 |
| Maximum recorded stress (N/m) | 8.9375 | 8.9376 | 14.9947 |
| Recorded integral (J/m²) | 1.2305 | 1.2094 | 4.8479 |

The same-rerun B-to-C integral change is +300.84%. C terminated by **stress collapsed**; spanning connectivity at C was **True**, and the residual load was 9.62% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 91 saved frames above the force tolerance and 11 above the transverse-stress tolerance. Reproduction flags: none at screening thresholds. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.4122 eV/Å and transverse stress -0.0853 N/m; endpoint residual checks do not pass.

### Completed example: S7_slit_0deg_armchair

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.25375 | 0.25375 | 0.47875 |
| Maximum recorded stress (N/m) | 20.2183 | 20.2192 | 20.2192 |
| Recorded integral (J/m²) | 2.9634 | 2.7082 | 3.3430 |

The same-rerun B-to-C integral change is +23.44%. C terminated by **not spanning (fractured)**; spanning connectivity at C was **False**, and the residual load was 9.39% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 62 saved frames above the force tolerance and 4 above the transverse-stress tolerance. Reproduction flags: integral_relative exceeds 2% diagnostic trigger. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.0521 eV/Å and transverse stress -0.0716 N/m; endpoint residual checks do not pass.

### Completed example: S5_H2_D40_W8_ellipseX

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.25875 | 0.25875 | 0.60875 |
| Maximum recorded stress (N/m) | 17.4602 | 17.5135 | 17.5135 |
| Recorded integral (J/m²) | 2.0206 | 2.1451 | 3.0341 |

The same-rerun B-to-C integral change is +41.44%. C terminated by **not spanning (fractured)**; spanning connectivity at C was **False**, and the residual load was 11.49% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 94 saved frames above the force tolerance and 3 above the transverse-stress tolerance. Reproduction flags: integral_relative exceeds 2% diagnostic trigger. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.0355 eV/Å and transverse stress -0.0253 N/m; endpoint residual checks do not pass.

### Completed example: P1_slit_35deg

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.32125 | 0.32125 | 0.44625 |
| Maximum recorded stress (N/m) | 17.0767 | 19.0772 | 19.5482 |
| Recorded integral (J/m²) | 1.6201 | 1.7128 | 2.4539 |

The same-rerun B-to-C integral change is +43.27%. C terminated by **stress collapsed**; spanning connectivity at C was **True**, and the residual load was 9.88% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 42 saved frames above the force tolerance and 26 above the transverse-stress tolerance. Reproduction flags: maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; first_damage_strain_difference exceeds one parent increment. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.0693 eV/Å and transverse stress -0.0383 N/m; endpoint residual checks do not pass.

### Completed example: P3_H2_square_W8_ellipse_30deg

| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.25625 | 0.25625 | 0.39625 |
| Maximum recorded stress (N/m) | 11.2279 | 11.1971 | 11.1971 |
| Recorded integral (J/m²) | 1.2735 | 1.2600 | 1.5068 |

The same-rerun B-to-C integral change is +19.59%. C terminated by **not spanning (fractured)**; spanning connectivity at C was **False**, and the residual load was 10.15% of the running maximum. A stress-collapse stop does not imply complete geometrical separation.

This trajectory has 50 saved frames above the force tolerance and 7 above the transverse-stress tolerance. Reproduction flags: none at screening thresholds. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain.

The final recorded frame itself has maximum force 0.9711 eV/Å and transverse stress -0.1680 N/m; endpoint residual checks do not pass.

Numerical tolerance exceedances and reproduction differences are retained in the paired table. A completed execution is not synonymous with observed failure or numerical validation. This analysis writes only to the isolated study folder; integration of the new simulation results remains deferred.

The original 40-mesh population is fixed. The original endpoint-integral/alignment correlation is 0.4715. 8 of its eight planned replacements are available. The updated endpoint correlation is 0.4884; it is not a completed-failure work correlation when censoring remains.

## Controls and numerical checks

| Case | Execution | Driver failure | Frames above force / transverse tolerance | Reproduction diagnostics |
|---|---|---|---|---|
| ctrl__S1_pristine_zz | completed | True | 0 / 2 | Within screening thresholds |
| ctrl__S2_slit_0deg | completed | True | 0 / 1 | integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment; first_damage_strain_difference exceeds one parent increment |
| res__S2_slit_45deg | completed | True | 112 / 18 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger |
| res__S5_H2_D40_W8_ellipseX | completed | True | 60 / 5 | integral_relative exceeds 2% diagnostic trigger |
| res__S7_slit_0deg_armchair | completed | True | 39 / 0 | Within screening thresholds |
| res__S7_slit_20deg | completed | True | 75 / 15 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment |
| quality__S1_pristine_zz_transverse15 | completed | True | 0 / 0 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment; first_damage_strain_difference exceeds one parent increment |
| quality__S1_pristine_zz_verified_Q4 | completed | True | 0 / 0 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment; first_damage_strain_difference exceeds one parent increment |
| quality__S7_slit_20deg_verified_Q4 | completed | True | 0 / 0 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment |
| quality__S1_pristine_zz_halfstep_Q7 | completed | True | 0 / 0 | integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment; first_damage_strain_difference exceeds one parent increment |
| quality__S1_pristine_zz_quarterstep_Q8 | completed | True | 0 / 0 | maximum_stress_relative exceeds 2% diagnostic trigger; integral_relative exceeds 2% diagnostic trigger; stop_strain_difference exceeds one parent increment; first_damage_strain_difference exceeds one parent increment |

Control **S1_pristine_zz**: original ending strain 0.27125, rerun 0.27125; maximum recorded stress 38.7946 → 38.7862 N/m; integral 6.7257 → 6.7193 J/m². The B-to-C added interval is 0.00000 strain. When B and C coincide, the archive-to-rerun difference is a reproduction difference, not an effect of longer loading.

Control **S2_slit_0deg**: original ending strain 0.21875, rerun 0.19750; maximum recorded stress 25.4021 → 24.9746 N/m; integral 3.2531 → 2.9993 J/m². The B-to-C added interval is 0.00000 strain. When B and C coincide, the archive-to-rerun difference is a reproduction difference, not an effect of longer loading.

Q1 increased only the pristine control's transverse-iteration budget from 3 to 15. Its ending strain changed from 0.27125 to 0.25375, and its recorded integral changed by -10.01%. All saved force and transverse-stress residuals passed their nominal tolerances in Q1. This is numerical-protocol sensitivity, not a longer-loading effect.

Independent CPU/float64 static reevaluation confirms large residual forces in selected accepted pilot frames, including frames whose inherited FIRE flag is true. The driver records that flag from an earlier minimization, before subsequent transverse relaxation; final forces must be checked directly. No potential or primary execution settings have been altered.

Q2 tests further fixed-cell relaxation of three saved frames and audits saved forces in all eight archived pilot-parent trajectories. These diagnostics do not resume loading trajectories and are not spliced into primary runs. Detailed results reside in `diagnostics/`. The production decision is recorded separately in `protocol/production_gate.json`.

### Archived pilot-parent force audit

This is a static CPU/float64 reevaluation of saved coordinates, without relaxation. The saved positions have float32 precision, so near-tolerance counts need that qualification. This selected eight-case audit does not estimate prevalence across all 96 discovery runs.

| Source phase | Parent design | Frames above force tolerance / saved frames | Maximum force residual (eV/Å) |
|---|---|---:|---:|
| I | S1_pristine_zz | 0 / 29 | 0.0194 |
| I | S2_slit_45deg | 13 / 52 | 1.7732 |
| I | S7_slit_20deg | 14 / 36 | 1.2888 |
| I | S7_slit_0deg_armchair | 0 / 40 | 0.0198 |
| I | S5_H2_D40_W8_ellipseX | 25 / 43 | 6.4769 |
| II | P1_slit_35deg | 24 / 60 | 3.5669 |
| II | P3_H2_square_W8_ellipse_30deg | 23 / 42 | 3.4791 |
| I | S2_slit_0deg | 4 / 27 | 0.2996 |

Q2 fixed-cell relaxation probes: **completed**. These results diagnose saved states; they cannot establish a repaired loading path or failure endpoint.

| Saved frame | Force before → after (eV/Å) | Stress before → after (N/m) | Force criterion met |
|---|---:|---:|---|
| ext__S2_slit_45deg / 37 | 1.0145 → 0.0808 | 11.815 → 8.439 | False |
| ext__S7_slit_20deg / 20 | 2.4524 → 3.7604 | 6.211 → 6.006 | False |
| ext__S7_slit_0deg_armchair / 27 | 2.5456 → 0.0168 | 7.053 → 6.936 | True |

Q3 conservative FIRE probes: **completed**. These continue the two unresolved Q2 configurations at fixed cell, with smaller time/displacement limits. They are not loading continuations.
- ext__S2_slit_45deg: final force 0.0176 eV/Å; transverse stress 0.4455 N/m; force tolerance met.
- ext__S7_slit_20deg: final force 0.0594 eV/Å; transverse stress 0.0594 N/m; force tolerance unmet.

Q5 endpoint relaxation probe: **completed**. It checks the recorded 45° endpoint at fixed cell, without modifying or extending its loading trajectory.
- Final force 0.3436 eV/Å; transverse stress -0.2185 N/m; stress remains below the frozen running-peak threshold: True; spanning: True.

Q6 CPU/float64 L-BFGS-B endpoint diagnostic: **execution_error**. It retains the fixed cell and original recorded endpoint geometry as input.
This diagnostic attempt failed during setup: AttributeError("'NoneType' object has no attribute 'ref_cell'"). Its files are retained; no material outcome is inferred.

Q6b CPU/float64 L-BFGS-B endpoint diagnostic: **completed**. It retains the fixed cell and original recorded endpoint geometry as input.
Final force 0.03684 eV/Å; transverse stress -0.33449 N/m; axial stress 0.07733 N/m. Force tolerance met: False; transverse tolerance met: False. Optimizer success alone is not used as a convergence certificate.

Q4 is a separately frozen numerical-protocol branch (`code_verified/`). It requires both final force and transverse-stress tolerances before accepting a loading step, retains rejected candidate states, and terminates unresolved trajectories as numerical limitations. Passing these residual checks alone does not establish strain-increment convergence.

B-to-C entries are left unavailable when a trajectory terminates before its virtual original stopping rule. No missing extension is represented by a zero effect.

The Q4 pristine control passed all saved-frame residual checks, including independent CPU/float64 reevaluation. Its ending strain was 0.24875, maximum recorded stress 36.507 N/m and integral 5.8302 J/m² (-13.23% relative to the new original-setting control). A separate Q4 20° slit diagnostic was released after this review. This does not release primary production or establish a unique numerically converged failure strain.

### Completed Q4 20-degree slit diagnostic

| Quantity | Archived original A | Strict rerun at original rule B | Strict extended endpoint C |
|---|---:|---:|---:|
| Ending strain | 0.19500 | 0.32000 | 0.33500 |
| Maximum recorded stress (N/m) | 8.9375 | 14.3315 | 14.3315 |
| Recorded integral (J/m²) | 1.2305 | 2.4338 | 2.4579 |

The Q4 stopping criterion was **stress collapsed**; spanning remains **True**. The original rule on this changed numerical trajectory stops by **max strain reached**, with renewed peaks preventing the earlier archived post-peak stop. B-to-C changes the integral by +0.99%. The larger archive-to-C change also includes numerical-protocol and reproduction effects. Passing the saved-state residual checks does not establish strain-increment convergence or complete geometric separation.

The completed original-setting 20-degree extension stops at 0.82000, versus 0.33500 for Q4. Its B-to-C recorded integral changes by +300.84%, versus +0.99% for Q4. The original-setting trajectory fails residual checks; Q4 passes saved-state checks but lacks demonstrated increment convergence. Both stop by stress collapse while still spanning. These contrasting numerical trajectories show that the inferred loading-window effect depends on the relaxation procedure; neither endpoint is selected as a uniquely converged material-failure value.

### Strain-increment diagnostics on the Q4 numerical branch

Q7 halves all three strain increments relative to Q4 while retaining its potential, solver, tolerances and loading rules. Q8, when listed below, halves those increments again. Each run starts from the original initial geometry. These are additional to the four original-setting resolution checks scheduled through the primary production queue. Each comparison uses the stated next-coarser reference; comparisons with the archived Phase-I result remain separately available in the main paired table.

quality__S1_pristine_zz_halfstep_Q7: **completed**. Reference: `quality__S1_pristine_zz_verified_Q4`.
Ending strain 0.24875 → 0.25563; maximum recorded stress change +4.74%; recorded integral change +5.39%. All recorded residual checks pass: True. A single refinement does not establish a unique converged failure value.

quality__S1_pristine_zz_quarterstep_Q8: **completed**. Reference: `quality__S1_pristine_zz_halfstep_Q7`.
Ending strain 0.25563 → 0.25156; maximum recorded stress change -7.65%; recorded integral change -3.69%. All recorded residual checks pass: True. A single refinement does not establish a unique converged failure value.

Three-level control review: the final refinement stability screen does not pass. All accepted states at these levels pass recorded and independent static residual checks, but endpoint, maximum stress and integral remain sensitive to the increment. The nonmonotonic values are retained; no level is selected as uniquely converged. One trajectory per increment does not separately estimate same-setting repeatability. This bounded three-level control diagnostic is complete; no further refinement is automatically launched.

### Independent static checks of saved loading states

Selected accepted configurations are reevaluated in CPU/float64 without relaxation. These checks verify saved-state residuals; they do not continue the loading path or establish strain-increment convergence. Saved positions have float32 precision.

quality__S1_pristine_zz_halfstep_Q7: static audit **completed**, 53/53 selected frames evaluated, 53 pass both residual checks. Source: `diagnostics/Q4_pristine_Q7_static_20260924T1735/record.json`.

quality__S1_pristine_zz_quarterstep_Q8: static audit **completed**, 103/103 selected frames evaluated, 103 pass both residual checks. Source: `diagnostics/Q4_pristine_Q8_static_20260924T1742/record.json`.

quality__S7_slit_20deg_verified_Q4: static audit **completed**, 3/3 selected frames evaluated, 3 pass both residual checks. Source: `diagnostics/Q4_slit_drop_static_20260924T1700/record.json`.

quality__S7_slit_20deg_verified_Q4: static audit **completed**, 64/64 selected frames evaluated, 64 pass both residual checks. Source: `diagnostics/Q4_slit_full_static_20260924T1742/record.json`.

ext__S7_slit_20deg: static audit **completed**, 3/3 selected frames evaluated, 2 pass both residual checks. Source: `diagnostics/original_setting_20deg_static_20260924T2150/record.json`.

ctrl__S2_slit_0deg: static audit **completed**, 2/2 selected frames evaluated, 1 pass both residual checks. Source: `diagnostics/original_setting_ctrl__S2_slit_0deg_static_20260924T2350/record.json`.

ext__P1_slit_35deg: static audit **completed**, 3/3 selected frames evaluated, 2 pass both residual checks. Source: `diagnostics/original_setting_ext__P1_slit_35deg_static_20260924T2350/record.json`.

ext__P3_H2_square_W8_ellipse_30deg: static audit **completed**, 3/3 selected frames evaluated, 1 pass both residual checks. Source: `diagnostics/original_setting_ext__P3_H2_square_W8_ellipse_30deg_static_20260925T0025/record.json`.

ext__S5_H2_D40_W8_ellipseX: static audit **completed**, 3/3 selected frames evaluated, 1 pass both residual checks. Source: `diagnostics/original_setting_ext__S5_H2_D40_W8_ellipseX_static_20260924T2350/record.json`.
