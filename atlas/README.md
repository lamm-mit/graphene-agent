# Geometry atlas and explorer

The source snapshots for the original 64K geometry collection and the expanded
256K collection are kept as sibling workdirs because the expansion uses the
original population. `DATASET_CARD.md` in each records the published dataset
description; tools retrieve standard atomistic coordinates and preserve cell
size and periodic boundaries.

- Dataset: https://huggingface.co/datasets/lamm-mit/graphene-design-universe-256k
- Explorer: https://huggingface.co/spaces/lamm-mit/graphene-design-explorer-256k
- Code: `graphene_design_dataset_256k/`
- Static explorer: `explorer/`

These are unrelaxed geometries, not a set of mechanical simulations. Property
schemas support later calculations but do not fill in unknown properties.
The embedding encodes geometry descriptors. See [REPRODUCING.md](../docs/REPRODUCING.md)
for small validation, single-design retrieval, complete workdir restoration,
figure rendering and local explorer instructions. Apache-2.0 for the authored
release; bundled Plotly retains its MIT notice.
