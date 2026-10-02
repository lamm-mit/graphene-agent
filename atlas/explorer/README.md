---
title: Graphene Design Universe 256K
emoji: ⚛️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
fullWidth: true
header: mini
short_description: Explore 256,000 graphene designs with atomistic coordinates
---

# Graphene Design Universe

An interactive geometry-similarity map of [256,000 graphene designs](https://huggingface.co/datasets/lamm-mit/graphene-design-universe-256k).

Drag to orbit, scroll to zoom, click a point or nearby design to inspect it. Filter by design group, size and porosity. Download standard extended XYZ coordinates including the full cell and periodic boundaries.

The map contains **unrelaxed geometries without mechanical-property labels**. It is a descriptor/PCA/UMAP embedding, not a force-field prediction, energy landscape or AI hidden state. Neighborhoods are computed in displayed embedding coordinates. Public Apache-2.0 release for LAMM, MIT.

The Space runs entirely in your browser and uses no paid compute. Dataset file ranges are fetched directly from the public Hub. See the dataset card for scientific definitions, sampling, provenance, validation and limitations.

Plotly.js is bundled under its MIT license; see `PLOTLY_LICENSE`.
