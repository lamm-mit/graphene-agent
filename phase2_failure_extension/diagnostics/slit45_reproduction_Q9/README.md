# 45-degree slit-array reproduction investigation (Q9)

## Completion review: September 26

R2 completed at 14:17 UTC in 51.68 minutes and its numerical/visual review is complete. The high-stress original response is closely reproduced: stress at strain 0.32 is 17.95069 versus 17.93355 N/m (scalar linear interpolation); normalized curve RMSE through 0.32 is 0.705%, and the integral differs by -0.294%. Intermediate states retain residual flags. See `results/single_replay_report.md`, `results/final_review.json` and `figures/single_replay_comparison.png`. The original three-run schedule remains superseded: no calculation is advancing, R1 and its scheduler remain suspended, and R3 is deferred. No diagnostic curve was inserted into the manuscript.

## Current scope: one 45-degree-only rerun (Q9a)

The user asked to simplify to a direct rerun of the 45-degree case. **Amendment Q9a supersedes the original three-run schedule below.** Only `R2_original_single` is active, using unchanged original settings and stopping rules. The original scheduler (PID 55231) and R1 batch worker (PID 55233) are intentionally suspended with SIGSTOP; saved data, checkpoints and process memory are retained. R3 is deferred. Do not resume either suspended process or launch another simulation automatically. `results/execution_control.json` records the current priorities and status overrides; raw R1 run files are preserved as written. The single-case supervisor is `scripts/run_priority_single.py`; it ends after R2 and launches nothing else. Compare R2 with the original and existing Phase-II paths promptly, then report and pause the follow-up.

User-authorized September 26 investigation, separate from the completed longer-loading study. Target: `S2_slit_45deg`, original run `run_20260905_091359_4df2c2`. This is the same design shown in original Figure S11. The current Figure 10 uses the two existing Phase-II paths, with its configurations from the primary path. The original curve remains in S11 and the paired SI comparisons.

## Findings established before the new replays

- The black curve was plotted correctly from the original trajectory. Both existing reruns start from the same initial coordinates and cell, with the same geometry seed, 3,088 atoms and force-field checksum. Regenerating the original geometry also gives identical float32 coordinates. The loading/minimization path adds no thermal noise or randomized initial perturbation.
- The original was part of the four-structure `stage2_b06` batch; each Phase-II rerun ran individually. The original and current OS versions differ. The available scientific source snapshot and recorded settings are copied here, but the full historical OS/backend and full-project source fingerprint are not reconstructed.
- At identical saved coordinates, six original configurations give batch-versus-single MPS stress differences of at most 0.0000344 N/m, much smaller than the curve separation. CPU float64 batch-versus-single force components match exactly in these tests. Repeated MPS evaluations show small floating-point variation. These static checks do not establish trajectory repeatability or convergence.
- At matched strain 0.170, the original and primary stresses are 2.5804 and 2.5819 N/m and the geometric bond graphs match. At 0.17125, stresses are 2.6447 and 1.2821 N/m: the primary has four cumulative lost connections while the original has zero. Their graphs differ by 12 edges. This identifies the first sharp topology divergence on the common saved strain grid, after smaller earlier stress/position differences. It does not establish a unique cause.
- The primary force residual at 0.17125 is 0.0769 eV/Angstrom, above the original 0.02 target. The original has residual failures at other states, including 1.773 eV/Angstrom at strain 0.15. A recorded FIRE convergence flag does not certify final post-transverse-relaxation forces. Neither path is established as a fully converged reference.
- The original ends at strain 0.32375 under its prescribed total-strain limit (nominally 0.32), still rising at 18.543 N/m. The existing primary peaks at 12.205 N/m before that limit. Loading that primary farther does not change its peak. Therefore this peak discrepancy is a reproduction issue, not an effect of observing a longer loading window.

## Controlled replays in progress

`protocol.json` and `jobs.json` freeze three sequential runs on the original recorded numerical settings and original stopping rules:

1. R1: restore the original four structures, ordering and batch size.
2. R2: run the target alone.
3. R3: repeat R2 with identical settings.

These runs test batch-layout dependence and same-setting repeatability; they are not new longer-loading runs or replacement material-property curves. Review all three before choosing any further intervention. One original batch historically took about four hours, so no short completion time is assumed.

The sole queue is `scripts/dispatch_serial.py`, with an exclusive lock and status in `results/status.json`. Accepted-state progress is in `runs/<id>/progress.json`. Each run preserves its accepted structures, forces, curves, minimizer-call sequence, neighbor-list rebuild log and full driver checkpoints around important strains. Instrumentation subclasses inherited routines for observation; it does not change the scientific updates. `source_checksums.json` freezes inputs and copied scientific source.

The first startup attempt failed before simulation because a wrapper named `queue.py` shadowed the Python standard library during PyTorch import. The wrapper was renamed, all original logs were retained under `logs/startup_attempt_01_*`, and the recovered process was verified alive. See `results/startup_recovery_01.json`. Do not overwrite any later failed attempt or restart without accounting for active queue/child processes.

## Diagnostics and interpretation

`scripts/static_audit.py` generated `results/static_audit.json`; no rerun is needed while a loading job is using the device. A separate synthetic affine-compression probe exposed a neighbor-list rebuild-guard limitation at 10% transverse compression, but the tested 0.5–5% contractions agreed with fresh lists. This is not evidence that the limitation caused the actual discrepancy. Do not silently change the frozen force kernel, minimizer or neighbor policy.

The batch-wide second minimization, global neighbor-list refresh schedule, floating-point path, first-damage refinement and finite transverse-relaxation budgets are candidates to inspect in the new traces. Already-converged members may be frozen during an extra batch minimization, so its presence alone does not prove a cause. Do not describe this as a different-seed effect or a measured distribution of physical randomness.

Run `scripts/compare.py` with the PyTorch environment's Python to regenerate saved-path CSV/JSON comparisons. This is read-only for simulations and works while a run is in progress. Partial maxima and last sampled strains are explicitly progress values. `results/common_strain_divergence.json` preserves the initial matched-strain audit; the newer comparison files specify the displacement convention and include the new replays as they become available.

All diagnostic results remain here. Further scientific interventions require separate recorded amendments, and no diagnostic curves enter the manuscript automatically. Completion requires interpreting the controlled replays, checking numerical qualifications, generating and reviewing diagnostic comparisons, and explaining established versus unresolved causes to the user.

## Frozen Phase-I source cross-check

A separate read-only check against the user's `RESULTS_nano_v2 SINGLE SHOT/carbon_discovery` folder found the AQS driver, minimizer, force kernel, neighbor routines and geometry sources identical to the Q9 copies. Of all 59 copied Python files, 52 match, four differ only in campaign/launcher wrappers, and three later analysis/helper files are absent from the frozen folder. The `run_batch.py` difference is device-name bookkeeping, with the same simulation setup. Its full-tree content hash still differs from the run's historical project hash (the completed snapshot includes a different set of campaign/analysis files); exact historical environment reconstruction is not claimed. Details and source diffs are retained in `results/frozen_source_check.json` and `results/source_diff_*.txt`.

## Replay review at September 26 12:14 UTC

The original four-structure replay is alive and has accepted strain 0.17; the single-case repeats remain queued. All 79 frozen source/input checksums pass. At 0.17 the replay's bond graph matches the original and its stress is 2.5548 versus 2.5804 N/m. The accepted loading histories already differ: the original log rejects its first attempted 0.17 state after post-trial processing, then accepts an intermediate 0.165 state; the new replay accepts 0.17 directly. This is a specific adaptive-step difference to investigate, not evidence that the intermediate step alone explains the later high-strain divergence.

The trace currently contains one full-budget retry with the target active and already converged from the short trial: its target coordinates move by exactly zero. Thus the mere presence of an additional batch minimization cannot be treated as evidence of extra target relaxation. The replay has saved full checkpoints at 0.16 and 0.17. No recovery or scientific protocol amendment was needed.

`scripts/review_progress.py` creates the machine-readable increment-history audit and `figures/transition_diagnostic.png` / `.svg`, with plotted CSV and a separate caption. The first interim visualization was checked for values, explicit in-progress labeling and layout. These files can be regenerated from later saved states; the dated review record preserves this assessment.

## Replay review at September 26 12:48 UTC

The batch replay has passed its first damage event and remains alive. First recorded loss occurs at 0.1725 with eight lost and sixteen formed connections, compared with four lost at 0.17375 in the original and four lost at 0.17125 in the existing Phase-II primary. The R1 graph still matches the original at 0.17125 (stress 2.6713 versus 2.6447 N/m); it differs by 24 edges at 0.1725 (stress 1.6982 versus 2.7461 N/m). Restoring the batch has not recovered the exact original onset. This does not yet isolate the cause of the larger high-strain stress difference.

The new trace makes the final-state qualification explicit: at the accepted R1 first-damage state, the full relaxation reached 0.01214 eV/Angstrom, but after three transverse relaxations the last minimization exhausted its 400-step budget at 0.58409 eV/Angstrom. The saved `fire_converged=true` refers to the earlier main relaxation. The final force norm independently calculated from the saved force array matches the reported 0.58409. The stress and network in that state are therefore under-relaxed; it is not a numerically validated fracture event. The preceding 0.17125 state meets the force target.

`scripts/inspect_first_damage.py` extracts this staged relaxation history reproducibly. The comparison CSV/JSON and transition graphic have been refreshed and visually reviewed, with a dated plot/data snapshot under `figures/reviews/20260926T1248`. All frozen inputs/source checksums still pass. No recovery, scientific protocol change or manuscript edit was made. R2 and R3 remain pending; complete those before selecting any further intervention.
