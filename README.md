# Graphene Agent

Code for **Models building models for discovery of graphene metamaterial design principles in the context of failure** — Markus J. Buehler, MIT.

An AI-built atomistic instrument implements screened REBO2 in PyTorch, generates graphene architectures, applies athermal quasistatic tension, records failure and stopping events, and supports hypothesis-driven analysis. Later studies examine loading limits, reconnection, numerical reproducibility and hierarchy. The geometry atlas expands the design vocabulary to 256,000 unrelaxed structures across 24 groups, with standard atomic coordinates and an interactive explorer.

**Source release candidate 1.1.0-rc.1.** This package extends the existing public repository while preserving its original scientific implementation. Large study data are prepared as separate verified archives; instructions distinguish locally prepared bundles from data already on Hugging Face.

## Start here

- [Installation, validation and reproduction commands](docs/REPRODUCING.md)
- [Figure-to-code map](docs/FIGURE_CODE_MAP.md)
- [Release scope and provenance](PROVENANCE.md)
- [Recorded release validation](release/VALIDATION.md)
- [Original public README](docs/ORIGINAL_README.md)

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest carbon_discovery/tests -q
```

Atomistica is an optional independent reference, installed separately; its comparisons skip when absent. The local release validation includes it.

## Contents

| Directory | Purpose |
|---|---|
| `carbon_discovery/` | Preserved original platform, validation suite, 132 campaign records, frozen report and original/early follow-up studies |
| `hierarchy_followup/` | 78 later hierarchy runs, generators, registry/crack controls, predictions, analyses and tests |
| `phase2_failure_extension/` | Longer-loading studies, numerical branches, Q9 reproduction diagnostics and current comparison/mechanism plotting code |
| `atlas/` | 64K precursor and 256K expansion generators, embedding code, coordinate tools and static explorer |
| `paper_analysis/` | Preserved original figure workflow, current plotting helpers and diagram sources |
| `release/` | Source inventory, data-bundle checksums, release-only changes and validation evidence |

Historical study paths are retained to preserve relative imports and result lineage. Original, primary follow-up and stricter numerical branches remain distinct. The internal hierarchy name “phase3” refers to later collaborative Phase-II work in the manuscript.

## Data

- [Original trajectories and supporting assets](https://huggingface.co/datasets/lamm-mit/graphene-agent-data)
- [256K geometry dataset](https://huggingface.co/datasets/lamm-mit/graphene-design-universe-256k)
- [Interactive design explorer](https://huggingface.co/spaces/lamm-mit/graphene-design-explorer-256k)

The new loading, hierarchy and diagnostic archives accompany this local release in a sibling `data/` folder. Restore them with:

```sh
python scripts/restore_release_data.py --bundle-dir ../data --all
```

This verifies checksums and refuses to overwrite different existing files. The atlas explorer is static; mechanical simulation is a separate workflow.

## Scientific scope

Screened REBO2 is a published reactive empirical interatomic potential. Agreement with a reference implementation is numerical validation of that model, not an ab initio calculation or proof of experimental accuracy. The recorded studies use their stated planar/periodic geometry and AQS settings.

Peak stress is the maximum in the recorded interval. Runs ending under a stop rule without an observed failure event remain censored. Stress–strain integrals are not fracture toughness. Original-to-rerun differences are separated from the effect of extending an individual rerun. Residual flags and unresolved convergence limits are retained; stricter branches are not silently substituted.

The atlas has no computed mechanical-property labels. Its map represents geometry similarity, not mechanical performance or an energy landscape.

## License and citation

Apache-2.0 for the authored code; see [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff). Original published artifacts and their attribution remain preserved. Plotly retains its MIT notice; Atomistica is an external reference dependency.
