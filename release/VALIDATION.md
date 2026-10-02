# Release validation

Status: passed for the tested scope. No full campaign was rerun.

## Scientific and data checks

- Original core suite: **10 passed**, including all three independent Atomistica energy/force/stress comparisons, finite-difference forces, neighbor lists, invariances, generator/coordinate checks and a short AQS smoke run.
- Hierarchy and release utilities: **10 passed** (seven hierarchy tests and three data-restoration tests). These include regeneration of the original reference geometries, crack/seam/registry controls, reference-curve metrics, overwrite protection and archive integrity/path checks.
- Atlas: **27 exact recipe regenerations spanning all 24 groups**. Geometry digests, full coordinates, cell and periodic boundaries match the stored examples. Extended-XYZ round trips preserve these quantities.
- Atlas map: all **256,000** IDs and group counts checked; embedding file unchanged; selected neighbor ranks verified in the displayed 3D coordinates.
- Loading analysis: regenerated designs, family hull vertices and quantitative summary match the research values (JSON key ordering ignored). All 96 discovery designs, 40 alignment designs and 45 loading/check paths are retained with source-phase and censoring/residual flags.
- Four data bundles restored with full per-file checksums and then rechecked as archives.
- Current explorer JavaScript passes Node syntax checking. The complete live browser/network interface was not retested during code packaging.

## Reproduction checks

The one-to-one figure builder and panel exporter completed. The SI builder produced 14 assets, numerical macros and two tables. The standalone loading analysis produced its four figure outputs and SI tables. Reconnection, 45-degree snapshots, lattice detail, atlas rendering and the hierarchy scale/flaw builders completed. The hierarchy quantitative evaluation completed using `--no-panels`; two representative hierarchy figures were separately regenerated with zero reported label overlaps.

The atlas, angle-dependence, 20-degree reconnection and 45-degree mechanism PNGs are byte-identical to their working versions. The two hierarchy PNGs have identical decoded pixels; binary metadata differs. Their plotted numerical tables and hypothesis verdicts match the working results. The atlas preview was inspected visually.

Initial packaging checks exposed missing output directories and missing small dependency records. These were fixed in this release only and affected commands were rerun successfully. Full hierarchy fracture-panel rendering was stopped after the numerical/representative figure checks were available, then the evaluator was run successfully with `--no-panels`. All local attempt logs remain in the adjacent validation workspace, outside the code archive.

## Limits and environment

Validation used Python 3.12 on macOS and the installed scientific libraries listed in `validated_environment.json`, through an isolated virtual environment that inherited those libraries. ReportLab was installed into that environment. This was **not** a from-zero dependency installation or a Linux test run. A Linux GitHub Actions workflow is included but has not run on GitHub; without optional Atomistica, three reference checks skip explicitly.

These checks establish source preservation, selected numerical regression, data restoration and representative reproducibility. They do not establish convergence of every historical trajectory, validate all 256,000 geometries mechanically, or certify experimental accuracy of the force field. Existing numerical limitations and phase-specific stopping rules remain documented.

`integrity_audit.json` records the source-preservation and curation checks. `portability_changes.patch` records the release-only source differences. `data_bundles.json` records every sidecar member. The original working folders and original Git checkout were verified unchanged for all curated source files.
