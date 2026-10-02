# Hierarchy follow-up: results against the pre-registered predictions

Evaluated 2026-09-26 09:00; 78 of 78 runs completed; predictions written 2026-09-23 14:04:20 (SHA-256 d5f0c3bd124bf5a8…); runs 2026-09-23 14:00 → 2026-09-26 08:59; 236.7 process-hours.

## T1 crack arrest by veins

Verdict: **FAIL** — {"L20_r0": {"retention": 0.8974501491820058, "rule": 0.918, "over_rule": 0.9776145415925989, "ellipse_retention": 0.7304571171965331, "arrested_at_peak": true, "passes": false}, "L40_r0": {"retention": 0.6392453335716661, "rule": 0.699, "over_rule": 0.9145140680567471, "ellipse_retention": 0.6090511203736866, "arrested_at_peak": true, "passes": false}, "L20_r1": {"retention": 0.966941572175905, "rule": 0.909, "over_rule": 1.063742103603856, "ellipse_retention": 0.739847674642352, "arrested_at_peak": true, "passes": false}, "L40_r1": {"retention": 0.5882300339969215, "rule": 0.697, "over_rule": 0.843945529407348, "ellipse_retention": 0.622161322938767, "arrested_at_peak": true, "passes": false}}

| design | σ cracked | σ uncracked | retention | rule | mechanism | obs/rule | W ratio | first damage ε | arrested at peak | first vein damage ε | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1_pristine_L20_r0 | 22.4 | 38.8 | 0.578 | 0.834 | 0.478 | 0.69 | 0.20 | 0.088 | pending | pending | abrupt brittle (single avalanche) |
| H1_pristine_L40_r0 | 17.5 | 38.8 | 0.452 | 0.675 | 0.472 | 0.67 | 0.13 | 0.071 | pending | pending | abrupt brittle (single avalanche) |
| H1_pristine_L20_r1 | 19.0 | 38.8 | 0.490 | 0.847 | 0.478 | 0.58 | 0.14 | 0.081 | pending | pending | abrupt brittle (single avalanche) |
| H1_pristine_L40_r1 | 16.9 | 38.8 | 0.437 | 0.675 | 0.472 | 0.65 | 0.13 | 0.073 | pending | pending | abrupt brittle (single avalanche) |
| H1_ellipse_L20_r0 | 13.4 | 18.4 | 0.730 | 0.876 | 0.876 | 0.83 | 0.46 | 0.084 | pending | pending | progressive (multiple events, load retained) |
| H1_ellipse_L40_r0 | 11.2 | 18.4 | 0.609 | 0.696 | 0.696 | 0.88 | 0.43 | 0.090 | pending | pending | progressive (multiple events, load retained) |
| H1_ellipse_L20_r1 | 13.6 | 18.4 | 0.740 | 0.851 | 0.851 | 0.87 | 0.56 | 0.094 | pending | pending | progressive (multiple events, load retained) |
| H1_ellipse_L40_r1 | 11.4 | 18.4 | 0.622 | 0.670 | 0.670 | 0.93 | 0.48 | 0.084 | pending | pending | progressive (multiple events, load retained) |
| H1_veinsX_L20_r0 | 17.6 | 19.6 | 0.897 | 0.918 | 1.056 | 0.98 | 0.90 | 0.114 | True | 0.139 | progressive (multiple events, load retained) |
| H1_veinsX_L40_r0 | 12.5 | 19.6 | 0.639 | 0.699 | 0.804 | 0.91 | 0.50 | 0.074 | True | 0.119 | progressive (multiple events, load retained) |
| H1_veinsX_L20_r1 | 19.0 | 19.6 | 0.967 | 0.909 | 1.045 | 1.06 | 0.94 | 0.140 | True | 0.150 | progressive (multiple events, load retained) |
| H1_veinsX_L40_r1 | 11.5 | 19.6 | 0.588 | 0.697 | 0.802 | 0.84 | 0.76 | 0.076 | True | 0.116 | stepwise (few events) |
| H1_slit_L20_r0 | 15.6 | 25.4 | 0.614 | 0.923 | 0.923 | 0.67 | 0.54 | 0.071 | pending | pending | progressive (multiple events, load retained) |
| H1_slit_L40_r0 | 14.1 | 25.4 | 0.556 | 0.738 | 0.738 | 0.75 | 0.34 | 0.104 | pending | pending | progressive (multiple events, load retained) |
| H1_slit_L20_r1 | 14.1 | 25.4 | 0.556 | 0.938 | 0.938 | 0.59 | 0.35 | 0.066 | pending | pending | progressive (multiple events, load retained) |
| H1_slit_L40_r1 | 15.0 | 25.4 | 0.590 | 0.723 | 0.723 | 0.82 | 0.37 | 0.109 | pending | pending | progressive (multiple events, load retained) |

## T4 designed two-stage failure

Verdict: **FAIL** — {"per_width": {"16": false, "20": false}}

| design | σ | σ rule | veins alone (rule) | residual after 1st avalanche | rule | second peak | post-peak energy | control | post-avalanche strain range | W | truncated | mode | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H4_veinsX_W16_r0 | 18.0 | 18.2 | 11.5 | 0.84 | 0.63 | pending | 0.215 | 0.208 | 0.025 | 1.93 | False | progressive (multiple events, load retained) | False |
| H4_veinsX_W16_r1 | 18.0 | 18.2 | 11.5 | 0.84 | 0.63 | pending | 0.189 | 0.208 | 0.020 | 1.90 | False | progressive (multiple events, load retained) | False |
| H4_veinsX_W20_r0 | 17.8 | 19.5 | 14.5 | 0.57 | 0.74 | 0.15 | 0.288 | 0.208 | 0.070 | 1.78 | False | stepwise (few events) | False |
| H4_veinsX_W20_r1 | 17.8 | 19.5 | 14.5 | 0.58 | 0.74 | 0.13 | 0.302 | 0.208 | 0.075 | 1.79 | False | stepwise (few events) | False |
| S4_H2_D40_W8_veins_x (phase 1) | 16.0 | pending | pending | 0.51 | pending | pending | 0.138 | 0.208 | 0.025 | 1.63 | pending | progressive (multiple events, load retained) | pending |
| S5_H2_D40_W16 (phase 1) | 14.8 | pending | pending | 0.79 | pending | pending | 0.185 | 0.208 | 0.020 | 1.36 | pending | progressive (multiple events, load retained) | pending |
| S5_H2_D40_W20 (phase 1) | 15.6 | pending | pending | 0.24 | pending | pending | 0.061 | 0.208 | 0.005 | 1.34 | pending | abrupt brittle (single avalanche) | pending |
| S7_H2_veinsX_W12_ellipseX (phase 1) | 19.4 | pending | pending | 0.72 | pending | 0.11 | 0.359 | 0.208 | 0.070 | 1.98 | pending | stepwise (few events) | pending |

## T2 Cook–Gordon seams

Verdict: **FAIL** — {"seams": {"seamX_s40": {"W_ratios": [1.0176545098137848, 1.0656339343769052], "deflected": [false, false], "passes": false}, "seamX_s30": {"W_ratios": [1.0677508572274312, 0.9745191484275428], "deflected": [true, false], "passes": false}, "weakseamX_s20": {"W_ratios": [0.972495448662796, 1.055043105555341], "deflected": [true, false], "passes": false}, "seamX_s20": {"W_ratios": [0.8750632118392033, 0.9448211304271288], "deflected": [true, false], "passes": false}}, "controls_below_1p1": false, "control_W_ratios": {"H2_seamY_s20_r0": 0.5984489896225782, "H2_seamY_s20_r1": 1.2826877445853258, "H2_random_s20_r0": 0.7245490445000917, "H2_random_s20_r1": 0.9032765777129168}}

| design | σ | σ ref | σ rule | σ ratio | W | W ref | W ratio | rule | hypothesis | deflected | events in seams | damage extent x (Å) | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2_seamX_s20_r0 | 16.3 | 22.4 | 17.4 | 0.73 | 1.19 | 1.36 | 0.88 | 0.93 | 1.30 | True | 0.14 | 31.4 | progressive (multiple events, load retained) |
| H2_seamX_s20_r1 | 14.9 | 19.0 | 17.4 | 0.79 | 0.86 | 0.91 | 0.94 | 0.92 | 1.30 | False | 0.08 | 15.7 | progressive (multiple events, load retained) |
| H2_seamX_s30_r0 | 20.5 | 22.4 | 17.9 | 0.92 | 1.45 | 1.36 | 1.07 | 0.95 | 1.30 | True | 0.09 | 36.0 | progressive (multiple events, load retained) |
| H2_seamX_s30_r1 | 17.5 | 19.0 | 17.9 | 0.92 | 0.89 | 0.91 | 0.97 | 0.95 | 1.30 | False | 0.14 | 10.3 | abrupt brittle (single avalanche) |
| H2_seamX_s40_r0 | 21.1 | 22.4 | 18.1 | 0.94 | 1.38 | 1.36 | 1.02 | 0.96 | 1.30 | False | 0.02 | 10.8 | abrupt brittle (single avalanche) |
| H2_seamX_s40_r1 | 17.5 | 19.0 | 18.1 | 0.92 | 0.97 | 0.91 | 1.07 | 0.96 | 1.30 | False | 0.11 | 16.7 | progressive (multiple events, load retained) |
| H2_weakseamX_s20_r0 | 15.0 | 22.4 | 17.1 | 0.67 | 1.32 | 1.36 | 0.97 | 0.91 | 1.30 | True | 0.23 | 33.0 | progressive (multiple events, load retained) |
| H2_weakseamX_s20_r1 | 14.3 | 19.0 | 17.0 | 0.75 | 0.96 | 0.91 | 1.06 | 0.90 | 1.30 | False | 0.13 | 17.1 | progressive (multiple events, load retained) |
| H2_seamY_s20_r0 | 11.8 | 22.4 | 17.3 | 0.53 | 0.81 | 1.36 | 0.60 | 0.92 | 0.92 | True | pending | 109.5 | progressive (multiple events, load retained) |
| H2_seamY_s20_r1 | 11.9 | 19.0 | 17.3 | 0.63 | 1.17 | 0.91 | 1.28 | 0.92 | 0.92 | True | pending | 109.7 | progressive (multiple events, load retained) |
| H2_random_s20_r0 | 14.3 | 22.4 | 17.3 | 0.64 | 0.98 | 1.36 | 0.72 | 0.92 | 0.92 | True | 0.33 | 36.9 | stepwise (few events) |
| H2_random_s20_r1 | 13.0 | 19.0 | 17.3 | 0.68 | 0.82 | 0.91 | 0.90 | 0.92 | 0.92 | True | 0.23 | 39.0 | progressive (multiple events, load retained) |

## T3 scale separation (300 Å)

Verdict: **PASS (W)** — {"r0": {"premium_sigma_vs_phase1_trend": -3.4057113346545336, "sigma_premium_vs_fine": 4.018931505078612, "sigma_premium_vs_coarse": 3.796133745123152, "W_premium_vs_fine": 0.4931433260576825, "W_premium_vs_coarse": 0.7595887997955899, "retention_premium_vs_fine": 0.08413623334128084, "W_beyond_scatter": true, "retention_beyond_rule": false, "strength_beyond_scatter_same_cell": true}, "r1": {"premium_sigma_vs_phase1_trend": -3.5259165035406603, "sigma_premium_vs_fine": 3.9780119231772595, "sigma_premium_vs_coarse": 3.764824998299753, "W_premium_vs_fine": 0.8111167929829339, "W_premium_vs_coarse": 1.230355332244386, "retention_premium_vs_fine": 0.00486909305713612, "W_beyond_scatter": true, "retention_beyond_rule": false, "strength_beyond_scatter_same_cell": true}}

| design | σ | σ rule | σ trend(A) | A | σ − phase-1 trend | σ − fine | σ − coarse | W | W − fine | W − coarse | retention | rule | obs/rule | retention − fine | arrested at peak | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H3_2L_r0 | 19.3 | 17.5 | 22.1 | +0.56 | -3.4 | +4.0 | +3.8 | 1.91 | +0.49 | +0.76 | pending | pending | pending | pending | pending | progressive (multiple events, load retained) |
| H3_1L_fine_r0 | 15.3 | 13.9 | 18.4 | +0.32 | -3.3 | pending | pending | 1.42 | pending | pending | pending | pending | pending | pending | pending | progressive (multiple events, load retained) |
| H3_1L_coarse_r0 | 15.5 | 16.0 | 11.6 | +0.02 | +2.0 | pending | pending | 1.15 | pending | pending | pending | pending | pending | pending | pending | abrupt brittle (single avalanche) |
| H3_2L_L60_r0 | 15.5 | 14.9 | pending | pending | pending | pending | pending | 1.68 | pending | pending | 0.802 | 0.851 | 0.94 | +0.084 | True | progressive (multiple events, load retained) |
| H3_1L_fine_L60_r0 | 11.0 | 11.4 | pending | pending | pending | pending | pending | 1.37 | pending | pending | 0.717 | 0.817 | pending | pending | pending | progressive (multiple events, load retained) |
| H3_1L_coarse_L60_r0 | 11.0 | 10.7 | pending | pending | pending | pending | pending | 1.21 | pending | pending | 0.707 | 0.667 | pending | pending | pending | stepwise (few events) |
| H3_2L_r1 | 19.3 | 17.5 | 22.1 | +0.57 | -3.5 | +4.0 | +3.8 | 2.35 | +0.81 | +1.23 | pending | pending | pending | pending | pending | stepwise (few events) |
| H3_1L_fine_r1 | 15.3 | 13.9 | 18.4 | +0.33 | -3.5 | pending | pending | 1.54 | pending | pending | pending | pending | pending | pending | pending | progressive (multiple events, load retained) |
| H3_1L_coarse_r1 | 15.5 | 16.0 | 11.6 | +0.02 | +1.9 | pending | pending | 1.12 | pending | pending | pending | pending | pending | pending | pending | abrupt brittle (single avalanche) |
| H3_2L_L60_r1 | 14.7 | 14.7 | pending | pending | pending | pending | pending | 1.56 | pending | pending | 0.764 | 0.837 | 0.91 | +0.005 | True | progressive (multiple events, load retained) |
| H3_1L_fine_L60_r1 | 11.6 | 11.3 | pending | pending | pending | pending | pending | 1.04 | pending | pending | 0.759 | 0.815 | pending | pending | pending | progressive (multiple events, load retained) |
| H3_1L_coarse_L60_r1 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## T5 composite hierarchies (300 Å)

Verdict: **provisional FAIL** — {"VF_E": {"residuals": [0.5944418314008524, 0.45925487338988313], "energy_ratios": [2.704165182130782, 2.4935422293701803], "retention": 0.8267742191192711, "over_rule": 0.99611351701117, "ref_retention": 0.801575060918592, "passes": false}, "VF_R": {"residuals": [0.5597743353873962, 0.6896914858178357], "energy_ratios": [0.8721070644534001, 0.9716219833493593], "retention": 0.8682738521041437, "over_rule": 1.043598379932865, "ref_retention": 0.8419529587179668, "passes": false}, "VE_R": {"retention": 0.6604190608909681, "ref_retention": 0.8419529587179668, "no_advantage": true}, "VR_E": {"retention": 0.6528408327497022, "ref_retention": 0.7174388275773111, "no_advantage": true}, "GR_0": {"residual": 0.4743193631451169, "strain_range": 0.07500000000000005, "retention": 0.746402628067779, "over_rule": 0.9014524493572211, "ref_retention": 0.6035874600912872, "passes": false}, "GFx_0": {"residual": 0.6190070355649928, "strain_range": 0.04500000000000004, "retention": 0.8347249596196077, "over_rule": 0.9496302157219655, "ref_retention": 0.6035874600912872, "passes": false}, "V3_E": {"sigma": 18.57855020178931, "sigma_ref": 19.29974673125478, "retention": 0.800762145381267, "ref_retention": 0.801575060918592, "passes": false}}

| design | σ | σ rule | σ rule b | reference | σ ref | W | residual after 1st avalanche | second peak | post-peak energy | / reference | post-peak strain range | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H5_VF_E_r0 | 18.1 | 16.5 | 21.1 | H3_2L_r0 | 19.3 | 2.37 | 0.59 | 0.12 | 0.862 | 2.70 | 0.200 | stepwise (few events) |
| H5_VF_E_r1 | 18.0 | 16.5 | 21.1 | H3_2L_r0 | 19.3 | 2.31 | 0.46 | pending | 0.795 | 2.49 | 0.155 | stepwise (few events) |
| H5_VF_R_r0 | 16.1 | 16.2 | 16.2 | H5_VS_R_r0 | 15.6 | 2.16 | 0.56 | pending | 0.839 | 0.87 | 0.130 | progressive (multiple events, load retained) |
| H5_VF_R_r1 | 16.4 | 16.2 | 16.2 | H5_VS_R_r0 | 15.6 | 2.36 | 0.69 | 0.15 | 0.934 | 0.97 | 0.200 | stepwise (few events) |
| H5_VE_R_r0 | 14.8 | 13.4 | 15.1 | H5_VS_R_r0 | 15.6 | 1.95 | 0.80 | 0.18 | 0.693 | 0.72 | 0.200 | stepwise (few events) |
| H5_VE_R_r1 | 14.6 | 13.4 | 15.2 | H5_VS_R_r0 | 15.6 | 2.09 | 0.79 | 0.26 | 0.866 | 0.90 | 0.200 | progressive (multiple events, load retained) |
| H5_VR_E_r0 | 15.0 | 13.7 | 18.1 | H3_2L_r0 | 19.3 | 1.79 | 0.66 | 0.17 | 0.701 | 2.20 | 0.200 | stepwise (few events) |
| H5_VR_E_r1 | 15.0 | 13.6 | 18.1 | H3_2L_r0 | 19.3 | 1.91 | 0.72 | 0.24 | 0.817 | 2.56 | 0.200 | stepwise (few events) |
| H5_VS_R_r0 | 15.6 | 17.1 | 17.1 | H3_1L_fine_r0 | 15.3 | 2.20 | 0.54 | pending | 0.962 | 2.58 | 0.200 | progressive (multiple events, load retained) |
| H5_VS_R_r1 | 15.9 | 17.2 | 17.2 | H3_1L_fine_r0 | 15.3 | 1.50 | 0.78 | pending | 0.278 | 0.75 | 0.045 | progressive (multiple events, load retained) |
| H5_GR_0_r0 | 11.1 | 9.5 | 9.5 | H5_GS_r0 | 18.4 | 0.96 | 0.47 | 0.15 | 0.204 | 4.36 | 0.075 | stepwise (few events) |
| H5_GR_0_r1 | 11.4 | 9.7 | 9.7 | H5_GS_r0 | 18.4 | 0.99 | 0.66 | pending | 0.302 | 6.47 | 0.055 | progressive (multiple events, load retained) |
| H5_GFx_0_r0 | 15.8 | 17.6 | 17.6 | H5_GS_r0 | 18.4 | 1.32 | 0.62 | pending | 0.333 | 7.13 | 0.045 | progressive (multiple events, load retained) |
| H5_GFx_0_r1 | 15.6 | 17.6 | 17.6 | H5_GS_r0 | 18.4 | 1.22 | 0.60 | pending | 0.266 | 5.69 | 0.030 | progressive (multiple events, load retained) |
| H5_V3_E_r0 | 18.6 | 17.8 | 21.9 | H3_2L_r0 | 19.3 | 2.67 | 0.74 | pending | 1.261 | 3.95 | 0.200 | progressive (multiple events, load retained) |
| H5_V3_E_r1 | 19.7 | 17.8 | 21.9 | H3_2L_r0 | 19.3 | 2.47 | 0.56 | 0.12 | 0.774 | 2.43 | 0.200 | stepwise (few events) |
| H5_GS_r0 | 18.4 | 18.8 | 18.8 | H3_1L_coarse_r0 | 15.5 | 1.28 | 0.02 | pending | 0.047 | 0.66 | 0.005 | abrupt brittle (single avalanche) |
| H5_GS_r1 | 18.4 | 18.8 | 18.8 | H3_1L_coarse_r0 | 15.5 | 1.30 | 0.02 | pending | 0.047 | 0.66 | 0.005 | abrupt brittle (single avalanche) |

| design | σ cracked | σ uncracked | retention | rule | mechanism | obs/rule | reference | reference retention | W ratio | arrested at peak | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H5_VF_E_L60_r0 | 15.0 | 18.1 | 0.827 | 0.830 | 0.954 | 1.00 | H3_2L_L60_r0 | 0.802 | 1.00 | False | progressive (multiple events, load retained) |
| H5_VF_E_L60_r1 | 14.8 | 18.0 | 0.819 | 0.826 | 0.950 | 0.99 | H3_2L_L60_r0 | 0.802 | 1.14 | False | progressive (multiple events, load retained) |
| H5_VF_R_L60_r0 | 13.9 | 16.1 | 0.868 | 0.832 | 0.956 | 1.04 | H5_VS_R_L60_r0 | 0.842 | 1.10 | False | progressive (multiple events, load retained) |
| H5_VE_R_L60_r0 | 9.8 | 14.8 | 0.660 | 0.836 | 0.836 | 0.79 | H5_VS_R_L60_r0 | 0.842 | 0.52 | False | progressive (multiple events, load retained) |
| H5_VR_E_L60_r0 | 9.8 | 15.0 | 0.653 | 0.819 | 0.819 | 0.80 | H3_1L_fine_L60_r0 | 0.717 | 0.50 | False | progressive (multiple events, load retained) |
| H5_VS_R_L60_r0 | 13.1 | 15.6 | 0.842 | 0.841 | 0.841 | 1.00 | H3_2L_L60_r0 | 0.802 | 0.56 | True | progressive (multiple events, load retained) |
| H5_GR_0_L40_r0 | 8.3 | 11.1 | 0.746 | 0.828 | 0.953 | 0.90 | H5_GS_L40_r0 | 0.604 | 0.76 | False | progressive (multiple events, load retained) |
| H5_GR_0_L40_r1 | 9.3 | 11.4 | 0.820 | 0.797 | 0.917 | 1.03 | H5_GS_L40_r0 | 0.604 | 0.76 | False | progressive (multiple events, load retained) |
| H5_GFx_0_L40_r0 | 13.2 | 15.8 | 0.835 | 0.879 | 1.011 | 0.95 | H5_GS_L40_r0 | 0.604 | 1.47 | False | progressive (multiple events, load retained) |
| H5_V3_E_L60_r0 | 14.9 | 18.6 | 0.801 | 0.821 | 0.903 | 0.98 | H3_2L_L60_r0 | 0.802 | 0.84 | False | progressive (multiple events, load retained) |
| H5_GS_L40_r0 | 11.1 | 18.4 | 0.604 | 0.801 | 0.801 | 0.75 | pending | pending | 0.64 | False | stepwise (few events) |

## Standard prediction evaluation (all designs with a completed run)

- modulus_2d_Nm: median |rel. error| 0.03, mean |abs. error| 9.33 (n = 44)
- failure_strain: median |rel. error| 0.13, mean |abs. error| 0.03 (n = 38)
- work_to_failure_J_m2: median |rel. error| 0.15, mean |abs. error| 0.35 (n = 49)
- damage_localization: median |rel. error| 0.15, mean |abs. error| 0.04 (n = 6)
- fracture-mode class accuracy: 0.41