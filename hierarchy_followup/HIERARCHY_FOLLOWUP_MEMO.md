# Memo: follow-up campaign "hierarchy" — can hierarchical structuring show a benefit?

**Status: PLANNED, NOT LAUNCHED** (memo written 2026-09-23 by the agent, on request of M. J. Buehler).
No simulation of this campaign has been run. Launching costs roughly 100–140 process-hours (3–4 days on the M4 Max
with three workers); get the user's go-ahead first. Everything below is written so that a future agent can execute it
without the conversation that produced it.

Project root: `RESULTS_nano_v2/carbon_discovery/`. Python: `/opt/homebrew/Caskroom/miniconda/base/envs/PyTorch/bin/python`
(conda env `PyTorch`: torch + ASE + Atomistica). Run everything from the project root.

---

## 1. Why this campaign exists

The main campaign (96 discovery runs, 2026-09-04..06) and the pre-registered paper sweeps (18 runs, 2026-09-09) found **no
benefit of pore hierarchy at equal mass**. The manuscript ("Models building models …", `paper/`) states it as one of its two
new results: once load-path alignment is measured, nested (two- and three-level) meshes add nothing.

What the database says today (all at porosity 0.17–0.23, ordered meshes only: 18 one-level, 21 nested; computed
2026-09-23 with `experiments/db.py` + `analysis/campaign_analysis.curve`):

| property | one-level median | nested median | nested minus one-level trend in the alignment index *A* |
|---|---|---|---|
| strength (N/m) | 13.5 | 13.0 | −1.6 ± 1.4 (95 % bootstrap −3.1 to −0.4; one-level scatter 2.0) |
| work to failure (J/m²) | 1.27 | 1.34 | −0.10 ± 0.21 (scatter 0.31) |
| failure strain | 0.17 | 0.19 | +0.02 ± 0.05 |
| post-peak load retention | 0.47 | 0.39 | +0.01 ± 0.15 |
| work to failure at matched *strength* | — | — | +0.02 ± 0.18 J/m² |
| second load maximum > 60 % of the peak after the first major drop | 2 of 18 | 4 of 21 | no clear effect |
| worst-case strength over load directions 0°/90° | coarse square mesh 17.0 (isotropic) | best nested 12.8 | hierarchy loses |

Two reasons this is expected, and they define what is still worth testing:

1. **Strength is bounded by the net-section rule.** σ_max ≈ (A_min/A) · σ_lig. Ligament-class medians in this model:
   straight along the load 30 N/m, wide (≥ 18 Å) round-pore ligaments 30, narrow (< 18 Å) 17, random networks 20
   (`paper/figures/mech_numbers.json`). Nesting fine pores inside veins replaces wide ligaments by narrow ones, so at fixed
   porosity hierarchy can only tie or lose on strength. Nature does not use hierarchy for strength at fixed mass either; it
   uses it for **flaw tolerance, crack arrest and graceful (two-stage) failure**. None of those has been tested here.
2. **Our "hierarchy" barely has two scales.** Veins are 8–12 Å wide, fine ligaments ≈ 7 Å (ratio ≈ 1.5), three pore rows per
   40 Å domain. Real hierarchical materials separate levels by a factor ≥ 10. A referee can say we never tested hierarchy.

The follow-up therefore tests hierarchy **where it is used in nature** (T1, T2, T4) and **at a scale where the levels are
separated** (T3). Strength at matched mass is *not* the target; it is recorded but expected to stay at or below the rule.

### Ideas already answered by existing data (do not repeat)
- Isotropy / robustness to load direction (0° and 90°): the single-level coarse square mesh (`S4_square_D40`, 17.0 N/m,
  symmetric) beats every nested mesh (`S2_H2_D40_W8` 12.8; veins-x/veins-y 16.0/12.0). Elongated-pore one-level mesh:
  19.0 / 8.6; slit array 25.4 / 4.0. Hierarchy does not buy isotropy at equal mass.
- Fibre-bundle load sharing as a general rule for the fracture mode: across 79 matched-porosity designs no geometric
  descriptor predicts avalanche size or post-peak retention (|Spearman| ≤ 0.35; number of parallel paths −0.11). The claim
  was removed from the paper. T4 tests a *designed* two-stage failure instead, which is a different, narrower claim.

---

## 2. Ground rules (inherited from phase 1, unchanged)

- Mechanics only from the screened REBO2 engine (`atomistics/`), validated 20/20 against Atomistica. No other potentials.
- Loading: athermal quasi-static uniaxial tension along **x**, protocol constants `experiments/campaign_design.AQS`
  (strain step 1 % elastic, 0.5 % after damage, bisected to 0.125 %, FIRE fmax 0.02 eV/Å, transverse stress relaxed,
  max strain 0.32). Device `mps`, dtype `float32` (validated; strength reproducible to 0.8 %).
- 2D stress in N/m, never converted; work to failure is an area under the curve, **not** a toughness. Model results only; no
  synthesizability claims.
- **Pre-registration is mandatory**: predictions written with `experiments/stage_tools.write_predictions(name, preds, notes)`
  (timestamped, SHA-256) **before** `write_specs` and before the first launch; evaluate afterwards with
  `stage_tools.evaluate_predictions(pred_file, stage)`, which matches records by structure name and stage.
- Porosity matching by bisection: `experiments/campaign_design.match(family, params, target, key, lo, hi)` (tolerance 0.008).
  Compare designs at 0.20 ± 0.01 unless the test says otherwise.
- Replicates: ordered designs are replicated by lattice registry (`offset` in `nanomesh_single`, `vein_offset` in
  `nanomesh_hier`), not by random seeds. Two registries per key design.
- Chirality: keep the sheet `orientation="zigzag"` with the load along x for every design. Rotating a pattern by 90° also
  changes the loaded lattice direction to armchair (≈ 20 % weaker); never compare across that.
- Ellipse convention (`design_space._in_shape`): `angle_deg = 90` puts the long axis of the pores **along** the load.
- Database safety: **never glob-delete inside `experiments/database/`**; explicit run ids only, dry-run first
  (a phase-1 accident deleted stage-1 records and they had to be rebuilt with `scripts/rebuild_records.py`).
- Figures: Arial, SVG + PNG + PDF, zero conflicts in the overlap audit (`paper/check_overlaps.py`); every number in text via
  generated macros (`paper/figures/numbers.tex`, `facts.tex`, `mech_numbers.tex`).

---

## 3. Infrastructure map (what exists, where)

| purpose | file / function |
|---|---|
| structure generators | `atomistics/structures/design_space.py`: `nanomesh_single`, `nanomesh_hier`, `slit_array`, `precrack`, `graded_pores`, `ring_around_hole`, …; `generate(family, **params)`; masks via `_in_shape`, removal `_remove`, pruning `_finish` |
| geometric descriptors | `atomistics/descriptors/descriptors.py: compute_descriptors(atoms, design)` → `min_solid_fraction_across_x`, `ligament_width_mean_A`, `anisotropy_index`, `ligament_orientation_deg`, … |
| alignment index, slit geometry | `experiments/campaign_paper_sweeps.py: alignment_index(desc)`, `slit_geometry(params, design)`, `sigma_net_straight()` |
| campaign template (copy this) | `experiments/campaign_paper_sweeps.py: build()` — `add()` matches porosity, generates, computes descriptors, stores predictions; then `write_predictions` → `write_specs` |
| spec helpers | `experiments/campaign_design.py: S(name, family, params, seed, reason, hypothesis, tags, parent, prediction, notes)`, `spec(campaign, stage, batch_name, structures, aqs)`, `write_specs(stage_dir, batches)` |
| launcher | `scripts/run_queue.py <spec_dir> --workers 3 --device mps` (one `scripts/run_batch.py` per spec; specs move to `done/`, `failed/`, logs in `logs/`) |
| progress | `scripts/status.py` |
| records | `experiments/db.py: all_records()`; each record has `name, family, params, campaign, stage, tags, parent, porosity, descriptors, metrics, trajectory, broken_bond_events, prediction` |
| curves and metrics | `analysis/campaign_analysis.py: curve(rec) -> (strain, stress N/m, n_broken)`, `M(rec, key)` |
| trajectories, frames, damage | `analysis/fracture_viz.py: load_traj, select_event_frames, draw_frame, damaged_atoms_until, progression_panel, make_movie` |
| paper figures | `paper/make_figures.py` (+ `check_overlaps.py`), `paper/extra_figures.py`, `paper/mechanism_figure.py`, `paper/build.sh` |
| prior pre-registration files | `experiments/predictions/holdout_predictions.json` (round 1), `paper_sweeps_predictions.json` (round 2); evaluations in `experiments/holdouts/` |

Reference records to compare against (name → strength N/m, other):

| record | value | note |
|---|---|---|
| `S1_pristine_zz` | 38.8 (Y = 250 N/m) | reference sheet |
| `S1_precrack_L10 / L20 / L30` (100 Å cell) | 24.1 / 18.5 / 18.7 | crack ⟂ load; `crack_w = 3.0` |
| `S4_precrack_L30_cell150 / L40 / S7_precrack_L50` | 19.4 / 17.9 / 16.1 | 150 Å cell |
| `S2_ellipse_90deg` | 19.0, W 1.94, A = +0.35 | best one-level mesh (pores along load), `period 16, aspect 2, angle 90, pore_d 5.234` |
| `S7_H2_veinsX_W12_ellipseX` | 19.4, W 1.98, A = +0.59, stepwise | best nested mesh: `Lx=Ly=120, levels 2, period 12, domain 40, vein_w 12, vein_dirs x, aspect 2, angle 90, pore_d 4.812` |
| `S2_slit_0deg` | 25.4, W 3.25 | `slit_len 26.5, slit_w 4, period_x 30, period_y 16` |
| `S4_square_D40` | 17.0 | coarse square mesh, `period 40, pore_d 17.594, shape square` |
| `S2_H1_p16` / `S2_H1_p12` | 11.7 / 13.3 | fine one-level meshes |
| `S2_H2_D40_W8` | 12.8 | nested, square veins, round pores |

---

## 4. Code changes required before launch

### 4.1 Generic crack insertion for any family (T1, T3)
Add to `design_space.py` (keeps every existing design reproducible; only new keys trigger it):

```python
def add_crack(atoms, crack_len, crack_w=3.0, crack_angle_deg=90.0, crack_center=(0.5, 0.5)):
    """Remove a slit of length crack_len (Å) centred at fractional cell position crack_center from an already
    generated structure.  crack_angle_deg = 90 -> crack perpendicular to the load (x)."""
    L = np.array([atoms.cell[0, 0], atoms.cell[1, 1]]); c = np.array([[crack_center[0] * L[0], crack_center[1] * L[1]]])
    mask = _in_shape(atoms.positions, c, "slit", crack_len, aspect=crack_w, angle=math.radians(crack_angle_deg), L=L)
    design = dict(atoms.info["design"]); design.update(crack_len=crack_len, crack_w=crack_w, crack_angle_deg=crack_angle_deg,
                                                        crack_center=list(crack_center), cracked=True)
    n0 = atoms.info.get("n_initial", len(atoms))
    new = _finish(_remove(atoms, mask, design), design); new.info["n_initial"] = n0
    return new

def generate(family, **params):
    crack = {k: params.pop(k) for k in ("crack_len", "crack_w", "crack_angle_deg", "crack_center") if k in params}
    atoms = FAMILIES[family](**params)
    return add_crack(atoms, **crack) if crack.get("crack_len") else atoms
```
Notes: match porosity on the **base** parameters first (no crack keys), then add the crack keys to the spec params; record the
uncracked design as `parent=` in `S(...)` and tag `cracked`. The descriptors of the cracked structure (minimum section etc.)
come out of `compute_descriptors` unchanged, which is what the net-section prediction uses. Keep `crack_w = 3.0` as in
the `precrack` family. Place the crack centre mid-domain so that the tips point at the veins (for `nanomesh_hier` with
`vein_offset = 0` the veins sit at y = 0, domain, 2·domain …; a domain centre is at y = 0.5·domain, i.e. `crack_center=(0.5, 0.5·domain/Ly)` rounded to the nearest domain centre).

### 4.2 Seam generator (T2)
New family `seam_pores` in `design_space.py`, registered in `FAMILIES`: rows of round pores of diameter `pore_d` at pitch
`pitch` along x, rows spaced `seam_spacing` along y (rows at y = seam_spacing·(j + 0.5)), optional `seam_dir="x"|"y"`.
Design dict must include `family="seam_pores", hierarchy_levels=1, pore_d, pitch, seam_spacing, seam_dir, Lx, Ly`.
Porosity ≈ (π/4)·pore_d² / (pitch·seam_spacing); `match` on `pore_d`. Combine with the crack via 4.1.

### 4.3 Large cells (T3)
No generator change. Before launching, check memory and speed with one energy/force evaluation of a 300 Å
`nanomesh_hier` (≈ 30,000 atoms) on `mps`: the engine pads pairs/triplets into dense tensors; 128 GB unified memory
should suffice but verify. Expect 12–25 process-hours per run (phase-1 scaling: a 100 Å cell, 3000 atoms, took 1.2–2.6 h).

### 4.4 Campaign script
Create `experiments/campaign_hierarchy_followup.py` by copying the structure of `campaign_paper_sweeps.build()`:
`CAMPAIGN, STAGE = "hierarchy", "hierarchy_followup"`, `SPEC_DIR = experiments/specs/hierarchy`, one design per spec
(so workers share evenly), predictions file `hierarchy_followup_predictions.json`. Put the rules of Section 5 into the
`predicted` dict of every design (`strength_Nm_rule`, plus test-specific keys such as `retention_rule`,
`W_ratio_hypothesis`, `residual_fraction_rule`) and the success criterion into `rationale`.

---

## 5. The experiments

### T1 — Crack arrest by veins (flaw tolerance)
**Rationale.** In hierarchical materials a stiff, unbroken layer ahead of a crack arrests it: a crack running from a
compliant region into a stiffer one sees a reduced driving force. A vein of solid graphene along the load is stiffer
(Y ≈ 250 N/m) than the fine-pored domain (≈ 100 N/m). If hierarchy has any mechanical value at this scale, it should show
up first as flaw tolerance. The halo designs of phase 1 (pores *around* a hole) reduced strength, but they put pores in the
crack path; T1 puts solid across it, which is the opposite geometry.

**Designs** (porosity 0.20 ± 0.01 before the crack; crack ⟂ load, `crack_w 3`, lengths 20 and 40 Å; two registries):
1. one-level aligned mesh: `S2_ellipse_90deg` parameters in a 120 Å cell (re-match `pore_d`); control.
2. two-level aligned mesh: `S7_H2_veinsX_W12_ellipseX` parameters (120 Å cell, domain 40, vein 12); the 20 Å crack lies
   inside one domain with its tips 4 Å from the veins; the 40 Å crack already cuts one vein.
3. slit array `S2_slit_0deg` parameters (strips are decoupled: the crack removes strips and nothing else).
4. reference: pristine sheet with the same cracks (exists: `S1_precrack_L20`; add L40 in a 120 Å cell).
Uncracked versions of 1–3 in the same cell and registry (three exist; run the missing registries).

**Metrics.** Notch sensitivity = σ_cracked / σ_uncracked (same registry); work-to-failure ratio; crack path from the
trajectory (does the crack arrest at a vein? use `damaged_atoms_until` and frame panels); strain at first damage.

**Predictions to register.** Rule: σ_cracked = (A_min/A)_cracked · σ_lig(class), with the descriptor computed on the
cracked structure; retention_rule = σ_cracked / σ_uncracked. For the slit array the rule is expected to be exact (strips
are independent). For the veins mesh the mechanism hypothesis is retention ≥ 1.15 × retention_rule at both crack lengths
and both registries, with the crack arrested at the first vein at peak load. Secondary prediction: every porous mesh is far
less notch-sensitive than the pristine sheet (18.5/38.8 = 0.48 at 20 Å), because pores already blunt the tip.

**Success criterion.** Veins-mesh retention exceeds the rule by ≥ 15 % and exceeds the one-level control by ≥ 10 % at both
lengths and registries. Runs: 3 structures × 2 lengths × 2 registries + 3 uncracked registries + 1 pristine L40 = 16,
≈ 25 process-hours.

### T2 — Cook–Gordon seams (crack deflection)
**Rationale.** Nacre and bone toughen by deflecting cracks into weak interfaces. Ahead of a mode-I tip the stress parallel
to the crack is about a fifth of the opening stress; an interface weaker than that opens ahead of the tip and blunts it
(Cook–Gordon). Graphene has no interfaces, but a row of small pores **along the load** is weak in tension across the row,
i.e. exactly the direction that is loaded ahead of a transverse crack. This is the hierarchical toughening motif that the
main campaign never contained.

**Designs** (100 Å cell, pristine sheet + central crack 20 Å ⟂ load + seams; seam pores `pore_d ≈ 4`, `pitch 8`):
seam spacing 20, 30, 40 Å (porosity ≈ 0.08, 0.05, 0.04). Controls: the same number of pores placed randomly
(`vacancies`-like, `organisation="random"` or a jittered lattice); seams **across** the load (`seam_dir="y"`); the plain
`S1_precrack_L20` reference (exists). Three registries for the best-performing spacing afterwards.

**Metrics.** W_cracked(seams) / W(`S1_precrack_L20`); strength; crack path (deflection into a seam yes/no, read from the
frames); post-peak retention.

**Predictions.** Rule: strength = 18.5 N/m × (1 − seam column loss), i.e. a drop of roughly the seam porosity; W follows
strength (r = 0.82 between W and σ for one-level meshes) → W ratio ≈ 0.9. Mechanism hypothesis: the crack deflects into
the first seam it meets, the strength stays within 10 % of the reference and W ratio ≥ 1.3. Controls predicted not to help.

**Success criterion.** W ratio ≥ 1.3 with visible deflection at ≥ 2 of 3 registries; random-pore and across-load controls
below 1.1. Runs: 3 spacings + 2 controls + 2 extra registries + 1 = 8, ≈ 12 process-hours.

### T3 — Scale separation
**Rationale.** The strongest objection to the current "no" is scale (Section 1, point 2). Rebuild the hierarchy with a
level ratio of about 4 and eight pore rows per domain, so that "vein" and "fine ligament" are genuinely different objects.

**Designs** (300 × 300 Å, porosity 0.20, veins along the load):
- two-level: `nanomesh_hier(Lx=300, Ly=300, levels=2, period=12, domain=100, vein_w=30, vein_dirs="x", aspect=2,
  angle_deg=90)`, match `pore_d` (bisection gives 4.50 Å, long axis 9.0 Å, N = 27,570 at porosity 0.193; checked 2026-09-23.
  With `aspect=2` the long axis is 2·pore_d and must stay below the 12 Å period, otherwise the pores merge into slits:
  pore_d = 6 gives porosity 0.88);
- one-level fine: `nanomesh_single(period=12, aspect=2, angle_deg=90)` in the same cell, match `pore_d`;
- one-level coarse: `nanomesh_single(period=100, shape="square")`, match `pore_d` (≈ 45 Å; ligaments ≈ 55 Å wide);
- cracked variants (60 Å crack ⟂ load, mid-domain) of the two-level and the coarse one-level design (T1 at scale).

**Metrics.** Strength and W premium of the two-level design relative to the one-level trend in *A* (recompute the trend
with the new points; the small-cell trend may not transfer); notch sensitivity of the cracked pair.

**Predictions.** Rule estimates: coarse one-level ≈ 0.55 × 30 ≈ 16.5 N/m; two-level ≈ 0.3 × 30 + 0.7 × 0.45 × 17 ≈ 14 N/m
(veins carry 30 N/m over 30 % of the section, fine ligaments 17 N/m over the rest); fine one-level ≈ 0.65 × 17 ≈ 11 N/m.
Compute the registered numbers from the actual descriptors, not from these estimates. Hypothesis under test: the two-level
design shows a W or crack-retention premium beyond the one-level scatter (0.3 J/m², 2 N/m) even though its strength does not.

**Success criterion.** Premium > scatter in W or in crack retention, in the same direction at both registries (if a second
registry is affordable). Runs: 3 uncracked + 2 cracked = 5 (+2 registries optional), ≈ 60–100 process-hours; run last.

### T4 — Designed two-stage failure (damage tolerance)
**Rationale.** Fig. 6d of the paper shows the mechanism once: in the best nested design the fine-pore rows fail first and
the veins keep carrying load. Across all nested designs this is not systematic (4 of 21). Design it deliberately: veins
sized by the rule to carry a stated fraction of the peak alone, plus a fine level that fails first. The property being
bought is residual load capacity after the first failure event, which a one-level mesh cannot have at all.

**Designs** (120 Å cell, porosity 0.20, veins along the load, fine pores along the load `aspect 2, angle 90`):
vein_w 16 and 20 Å at domain 40 (vein fractions 0.4 and 0.5; fine-level porosity inside the domains 0.33 and 0.40),
two registries each. Controls: the one-level aligned mesh at 0.20 (exists), and the 0° slit array (exists: strips of one
width, no weak level).

**Metrics.** Residual fraction = highest stress after the first major drop (> 15 % of the peak) divided by the peak; energy
absorbed after the peak (∫σ dε beyond the peak); strength; W.

**Predictions.** Rule: veins alone carry f_vein × 30 N/m (12 and 15 N/m); the peak ≈ veins + fine level
(≈ 12 + 0.6 × 0.5 × 17 ≈ 17 N/m and ≈ 15 + 4 ≈ 19 N/m) → residual fraction ≈ 0.7–0.8; the one-level control has no second
stage (residual ≈ 0.3–0.5 in the database); the slit array fails in one avalanche (0.13).

**Success criterion.** Residual fraction ≥ 0.6 at both registries and post-peak energy ≥ 1.5 × that of the one-level
control at the same strength. Runs: 2 widths × 2 registries = 4 (+2 controls if new registries are needed), ≈ 8 process-hours.

---

## 6. Order, budget, commands

| order | test | runs | cell | process-hours | new code |
|---|---|---|---|---|---|
| 1 | T1 crack arrest | 16 | 120 Å | ≈ 25 | 4.1 |
| 2 | T4 two-stage | 4–6 | 120 Å | ≈ 8 | none |
| 3 | T2 seams | 8 | 100 Å | ≈ 12 | 4.2 (+4.1) |
| 4 | T3 scale | 5–7 | 300 Å | ≈ 60–100 | memory check |

```bash
cd carbon_discovery
python experiments/campaign_hierarchy_followup.py                 # writes predictions (hashed) THEN specs/hierarchy/*.json
python scripts/run_queue.py experiments/specs/hierarchy --workers 3 --device mps
python scripts/status.py                                          # progress; logs in experiments/specs/hierarchy/logs/
python -c "from experiments.stage_tools import evaluate_predictions as E; E('experiments/predictions/hierarchy_followup_predictions.json', 'hierarchy_followup')"
```
Record the timestamps of the prediction file, the first and the last run and the evaluation (as in `paper/figures/facts.json`),
so that the paper's pre-registration timeline (Fig. 9e) can gain a third row.

## 7. What to write afterwards

- If any test succeeds: a new Results subsection "Where hierarchy pays" with one figure (schematic + failure snapshot +
  measurement, same style as Fig. 8, built with `paper/mechanism_figure.py` conventions) and a row in the claims table; the
  abstract's hierarchy sentence becomes "hierarchy buys alignment, not strength, but it buys flaw tolerance / graceful
  failure", whichever holds.
- If all four fail: strengthen the existing statement to "in this model hierarchy does not improve strength, work to
  failure, flaw tolerance or damage tolerance at these scales, tested where nature uses it", cite T1–T4 in the SI with
  their pre-registered predictions, and keep the honest limitation that the level ratio is at most 4.
- Either way the analysis goes through the macro pipeline (`paper/make_figures.py` → `numbers.tex`) so that text and data
  cannot drift apart.

## 8. Pitfalls
- `_finish` prunes dangling atoms; crack tips may open by one bond. Check the actual crack length in the generated
  structure (report it in the predictions file) rather than the nominal one.
- Porosity of cracked designs is slightly above 0.20; match on the base design and report both porosities.
- `min_solid_fraction_across_x` counts solid in a column whether or not it is connected along the load (it was blind to
  the bridge-bending regime of the slit sweep). For T1 and T3 check the crack column by hand.
- Two registries are the minimum for any claim; phase 1 showed registry-to-registry differences of up to 2 N/m.
- float32 MPS runs are not bit-reproducible (strength within 0.8 %); do not read differences below 1 N/m as effects.
- Never overwrite or delete records; new designs get new names (prefix `H1_`, `H2_`, `H3_`, `H4_` for T1–T4).

---

## 9. Running this on another machine (added 2026-09-23)

**If the other machine sees this folder through Dropbox (the situation on 2026-09-23):** the memo, the platform, the
database and the trajectories are already there; nothing needs to be cloned or shared. Three rules, all because Dropbox
syncs every file write and creates "conflicted copy" files when two machines touch the same file:

1. **Do not run simulations inside the Dropbox folder.** Copy the tree to a local, unsynced directory once
   (`rsync -a --exclude __pycache__ "<Dropbox>/RESULTS_nano_v2/carbon_discovery/" ~/work/carbon_discovery/`, about 1 GB
   including trajectories), run the campaign there, and merge the results back with the copy-only helper
   `python scripts/merge_runs_from.py ~/work/carbon_discovery --campaign hierarchy --specs` (dry run; add `--apply`),
   run from the Dropbox tree. The helper never overwrites or deletes an existing run id and rebuilds `index.jsonl` /
   `summary.csv`. Run it when the other machine's campaign is finished, not while runs are writing.
2. **Do not use git from two machines on `github/graphene-agent/`** (a `.git` directory under Dropbox sync is a known
   way to corrupt a repository). Push from the machine that created it, or `git clone` from GitHub on the other one.
3. **Only one Claude Code session edits a given file at a time.** The session on this machine (M4 Max) has finished
   writing to the tree; treat the memo, `paper/` and `github/` as owned by whichever session is active.

Each machine needs its own Python environment (`carbon_discovery/environment.yml`, or the `PyTorch` conda env on the
M4 Max); the interpreter path in this memo is the M4 Max's. Mark the folder "available offline" in Dropbox before
copying, otherwise the trajectories are placeholders that download on first read.

**If the other machine does not have the folder:** The public repository `lamm-mit/graphene-agent` (private on GitHub until the paper is out) contains the platform
under `carbon_discovery/` with the same layout as this tree, the database records of all 132 runs, and the phase-2
analysis under `paper_analysis/` (this tree calls it `paper/`; adjust paths in Sections 2 and 3 accordingly). This memo,
the campaign script it asks for and the results of the follow-up are **not** in the public repository; keep them in a
private place (a private repository such as `lamm-mit/graphene-agent-internal`, or copy them by hand) and never commit
them to `graphene-agent` before the decision to release them.

```bash
gh auth login                                    # or ssh keys; the repo is private
git clone https://github.com/lamm-mit/graphene-agent.git && cd graphene-agent
conda env create -f environment.yml && conda activate graphene-agent      # or pip install -r requirements.txt
# copy this memo to carbon_discovery/experiments/ (scp, Dropbox or the private repo); it is not git-ignored, so do not `git add` it
cd carbon_discovery && python -m pytest -q tests/                          # 10 tests, seconds
python validation/suite.py                                                 # TESTs 1,2,5-8,18,19 need Atomistica (pip install atomistica; needs gfortran)
```

**In both cases**, before the campaign: `python -m pytest -q tests/`, then a smoke batch
(`python scripts/run_batch.py experiments/specs/test/test_batch.json --device auto --no-traj`; it writes three tiny
records with campaign `infrastructure_test` into the *local* database, which the merge helper can be told to leave
behind with `--campaign hierarchy`), then `python validation/suite.py` if Atomistica is installed.

**Devices.** The engine runs on CUDA, Apple MPS or CPU; `run_queue.py`/`run_batch.py` default to `--device auto` (CUDA
if present, else MPS, else CPU) and record the device actually used in every record. Several GPUs:
`python scripts/run_queue.py experiments/specs/hierarchy --workers 4 --device cuda:0,cuda:1`. CUDA also supports float64,
so the float64 reference checks of the validation suite can run on the GPU there. The CUDA path is identical in code to
the MPS path but has **never been run**: on the new machine, run the tests and `validation/suite.py` first, then a short
smoke batch (`experiments/specs/test/test_batch.json` with `--device auto`) before launching the campaign.

**Speed and memory (measured on the M4 Max, MPS float32, one energy+force+virial evaluation, from this tree):**

| cell | design | atoms | GPU memory | time per evaluation |
|---|---|---|---|---|
| 120 Å | `S7_H2_veinsX_W12_ellipseX` parameters | 4,393 | 1.1 GB | 69 ms (15.7 µs/atom) |
| 200 Å | T3 two-level, matched | 12,246 | 2.3 GB | 175 ms (14.3 µs/atom) |
| 300 Å | T3 two-level, matched | 27,570 | 3.4 GB | 285 ms (10.3 µs/atom) |

A T3 run needs ≈ 300 loading steps × ≈ 300–2000 FIRE iterations, i.e. 1–6 × 10⁵ evaluations: 8–50 h per run on the
M4 Max (single worker). Memory is not a constraint on any current NVIDIA card (≤ 4 GB per worker at 300 Å; 24 GB cards
can hold several workers).

**Expectation for CUDA (not measured):** the engine is a chain of a few hundred small elementwise/gather kernels per
evaluation over dense padded tensors; on the M4 Max it is launch-bound below ≈ 3,000 atoms and bandwidth-bound above.
An A100/H100-class GPU has 3–4× the memory bandwidth and much lower launch latency, so 3–10× per evaluation is a
reasonable expectation, more for the small cells, plus `torch.compile` (unusable on MPS, worth trying on CUDA) and
batching several structures per process (the driver supports it; the paper sweeps used one structure per process).
Multi-process sharing of one NVIDIA GPU is time-sliced; start the NVIDIA MPS daemon (`nvidia-cuda-mps-control -d`) or
prefer batching over processes. float32 results on CUDA will differ from MPS at the 0.8 % level seen in the
reproduction check; do not compare CUDA and MPS runs below 1 N/m.

