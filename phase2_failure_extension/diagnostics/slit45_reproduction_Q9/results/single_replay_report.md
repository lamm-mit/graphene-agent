# Direct 45-degree rerun completed (Q9a)

The requested standalone rerun completed on September 26 at 14:17 UTC (10:17 a.m. EDT), after 51.68 minutes. It used the original initial geometry, original numerical parameters and original stopping rules. The outcome closely reproduces the original high-stress response. It is not an exact atom-by-atom reproduction or a convergence certificate.

## Numerical comparison

| Quantity | Phase 1 single shot | New original-rule single replay | Existing Phase-2 primary |
|---|---:|---:|---:|
| Maximum recorded stress (N/m) | 18.54298 | 18.36080 | 12.20464 |
| Strain at that maximum | 0.32375 | 0.32250 | 0.26125 |
| Stress at strain 0.32, linear interpolation (N/m) | 17.93355 | 17.95069 | 4.39704 |
| Integral through strain 0.32 (J/m²) | 1.241918 | 1.238261 | 1.295382 |
| First recorded connection loss | 0.17375 | 0.17250 | 0.17125 |
| New connections at last saved strain ≤0.32 | 190 at 0.31875 | 192 at 0.31750 | 98 at 0.31625 |

The endpoint maxima differ by -0.982%, but their endpoint strains differ by 0.00125, so this is not a same-strain strength comparison. Both are still rising at their prescribed total-strain stop; operational failure was not observed. At the common strain of 0.32, the scalar-interpolated stress difference is 0.096%. The normalized stress-curve RMSE over 0–0.32 is 0.705% (normalization: original maximum on the comparison grid), and the integral differs by -0.294%. Interpolation is only for scalar descriptive diagnostics, not bond events or configurations. These integrals are not fracture toughness.

## What explains the discrepancy, and what remains unresolved

1. The original black curve was correctly plotted. All 79 frozen source/input checksums pass. Initial coordinates and cells were verified identical in float32 before the run, and the force-field parameter checksum matched. Geometry seed differences and a wrong input design are not supported explanations.
2. A standalone run can recover the original high-stress response. Restoring the original batch layout is therefore not necessary to obtain that response; batch size alone does not explain why the existing Phase-2 primary follows a lower-stress path. A same-setting repeat was deferred, so repeatability statistics are not available.
3. Differences arise before the longer observation window. The new single replay includes an accepted 0.165 state, as the original did; the existing primary and suspended batch accepted 0.17 directly. Original and replay graphs match at every exactly shared saved strain through 0.17125. Their first shared-grid graph difference is at 0.17250: the replay has eight lost connections and the original has none (24 edges differ). Original first loss follows at 0.17375. The existing primary already differs from both at 0.17125. These are recorded adaptive-step and topology differences, not a different random seed or a consequence of later stopping.
4. The new trace identifies a concrete relaxation limitation at its first-damage state. The main relaxation passes with maximum force 0.01635 eV/Å, but the final transverse relaxation exhausts its 400-step budget and leaves 1.61340 eV/Å (target 0.02). The stored FIRE-converged flag refers to the main relaxation and does not certify the final state. All 51 saved force norms were independently checked against the saved arrays. Across this run, 8/51 frames exceed the force target and 19/51 exceed the transverse-stress target. Its final state meets both recorded targets. The original and existing Phase-2 paths also contain residual flags, with different force-evaluation provenance for the original audit.
5. Small coordinate/stress differences already exist after initial relaxation, before topology changes. Floating-point execution, neighbor rebuild history, adaptive bisection and finite transverse/minimizer budgets remain possible contributors. The recorded rebuild/minimizer traces are preserved; none isolates a unique causal trigger. The earlier synthetic neighbor-list guard issue is not established as the cause of these real trajectories. A close macroscopic replay does not prove a unique or converged physical path.

## Disposition and reproducibility

R2 is complete. R1 and its old scheduler remain intentionally SIGSTOP-suspended; R1 has 26 saved states through strain 0.20250 and is not a completed material response. R3 remains deferred. No further simulations have been launched or resumed. The Q9a follow-up is ready to pause after this review; additional causal interventions would need their own explicit, preserved branch.

The manuscript, Figure 10, original Figure S11, all original simulations and the hierarchy follow-up are unchanged by this diagnostic review. Figure 10 continues to show its existing Phase-2 paths; these diagnostic results have not been substituted.

Reproduce with the PyTorch environment's Python: `scripts/compare.py`, then `scripts/inspect_first_damage.py`, then `scripts/final_review.py`. Outputs: `results/final_review.json`, `results/final_review_summaries.csv`, `results/final_review_plotted_data.csv`, `results/final_review_interpolated_stress.csv`, `results/R2_exact_matched_comparisons.csv`, `results/R2_transition_trace.json`, and editable SVG/PNG `figures/single_replay_comparison` with its separate caption. Scientific trajectories and full checkpoints remain under `runs/`.
