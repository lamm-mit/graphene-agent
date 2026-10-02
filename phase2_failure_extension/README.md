# Later Phase-II loading and mechanism studies

This is the curated public-code candidate for the completed loading-limit
study and its mechanism/reproduction analyses. It preserves the scientific
source variants, original inputs, job settings, negative outcomes and recorded
quality flags. See [reproduction instructions](../docs/REPRODUCING.md) and the
[figure-source map](../docs/FIGURE_CODE_MAP.md).

- `source_original/`: original scientific implementation copied for comparison.
- `code/`: primary driver with saved-frame and residual instrumentation.
- `code_verified/`: separate stricter numerical sensitivity branch.
- `protocol/`: historical settings, amendments, parent mapping and review gates.
- `inputs/baseline/`: copied original records and curves; large states in the data bundle.
- `runs/`: 34 targeted extensions, two controls, four resolution cases and five
  additional numerical loading checks. These are 45 paths, not 45 independent
  new materials.
- `diagnostics/`: preserved numerical investigations including the Q9 45-degree
  single/batch reproduction tests.
- `candidate_figures/`: paired data, unchanged-panel comparisons, SI builders,
  reconnection and lattice plots, schematics and the atlas figure.

Source-phase attribution remains explicit. Phase-I predictions and their
original evaluations are frozen. Longer loading retains the operational
failure criterion and changes the stop-rule limits. A driver failure event
does not establish numerical convergence. Censored values remain endpoint
observations with conditional bounds; integrated response is not fracture
toughness. See `results/FINAL_REPORT.md` and the Q9 final review.

Folders retain historical names so existing relative scientific imports and
data lineage remain intact. These names do not imply different paper phases.
Release-only portability changes are documented separately from the executed
scientific protocols.
