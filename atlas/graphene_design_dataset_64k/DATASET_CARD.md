---
license: apache-2.0
language:
- en
pretty_name: Graphene Design Universe 64K
size_categories:
- 10K<n<100K
tags:
- materials-science
- graphene
- atomistic-design
- metamaterials
- molecular-dynamics
- geometry
- umap
configs:
- config_name: designs
  default: true
  data_files:
  - split: train
    path: data/*.parquet
---

# Graphene Design Universe 64K

**64,000 atomistic graphene geometries, with images, periodic cells, generator parameters, standard extended XYZ coordinates and a geometry-similarity embedding.**

[Explore the interactive map](https://huggingface.co/spaces/lamm-mit/graphene-design-explorer).

![Actual graphene designs in a neighborhood of the geometry embedding](media/molecular_neighborhood.jpg)

This is a **geometry-only design library** derived from the graphene metamaterial generators developed in an AI-built atomistic modeling workflow. It preserves the preceding 16,384-design visualization atlas and adds 47,616 newly generated designs. **These 64,000 designs were not all simulated in the paper or discovered autonomously in its original single-shot experiment.** No relaxation, force-field evaluation or mechanical simulation was performed to build this library. Energies, forces, stresses and mechanical properties are **not provided**, and missing properties must not be interpreted as zero.

## What is included

- Exactly 64,000 distinct saved coordinate realizations, with stable IDs `GDU-000000` through `GDU-063999`.
- 16 design groups with 4,000 designs each, using 11 underlying generator families.
- 384 × 384 PNG previews drawn from the actual retained carbon sites. Small designs show lattice bonds; dense designs show site occupancy. Images are not force-field or stress maps.
- Standard **extended XYZ** coordinates for every design, with species, Cartesian positions, the full 3 × 3 cell and periodic boundary flags. Each file is independently gzip-compressed and packed into an indexed tar shard.
- Viewer-compatible Parquet metadata including an image column, cell, atom counts, porosity, orientation and the complete generator parameter dictionary.
- A 3D UMAP embedding, the measured geometry descriptors, PCA parameters, and reproducible generation, packaging and visualization code.
- A text-free 4K movie exploring the expanded atlas and embedding.

The complete catalog contains **1,756,618,061 carbon sites** across all structures. Cell widths range from **3.69 to 349.84 nm**; the largest design contains **3,922,663 atoms**. These are geometry counts, not evaluated molecular-dynamics steps.

The dataset includes 744 independent ASE roundtrip checks, plus a separate check of the global size extremes and last record. All exported coordinate values were parsed back exactly. Four designs retain a small fraction of sites with fewer than two graph neighbors after the bounded geometric cleanup; these are identified by the `geometric_cleanup_flag` and `fraction_coordination_lt2` catalog columns and documented in the geometric descriptors and should not be assumed mechanically equilibrated.

## Geometry and units

All atoms are carbon. Lengths and Cartesian coordinates are in **ångström (Å)**; the convenience width and height columns are in nanometers. The graphene lattice constant is **2.460177 Å**. Atoms lie at `z = 10 Å` in an orthogonal cell with `Lz = 20 Å`. This third cell dimension is a **vacuum convention, not a material thickness**. Periodicity is `[True, True, False]`: periodic in the graphene plane and nonperiodic out of plane. The requested cell sizes are rounded to commensurate lattice repeats; **the saved cell is authoritative**.

Generators remove carbon sites to form pores, slits, networks, gradients, defects and hierarchical vein patterns. The geometric cleanup performs up to ten rounds of removal of sites with fewer than two graph neighbors, then retains the largest periodic connected component. This is not an energy minimization. Edges are unpassivated; some candidates may have thin connections, remaining undercoordinated sites or insufficient periodic load paths. Mechanical stability and suitability for a chosen simulation must be evaluated separately.

The 16 groups are round pores, square pores, hexagonal pores, oriented pores, angled slits, strut networks, polygon networks, pore gradients, flaws and halos, defects and seams, parallel veins, crossed veins, nested hierarchy, fibrous veins, composite grids and subvein hierarchy. Sizes and shape parameters follow the documented sampler, rather than a uniform sample of all possible graphene designs. Large hierarchical exemplars extend to approximately 350 nm. The library is broad but does not exhaust the continuous design space.

**Uniqueness means distinct coordinate realizations**, as checked using the cell, lattice registry and retained-site mask. It does not establish 64,000 topologically distinct structures, nor uniqueness under all rotations, translations or changes of scale.

## Load metadata and images

```python
from datasets import load_dataset
designs = load_dataset("lamm-mit/graphene-design-universe-64k",
                       "designs", split="train", streaming=True)
row = next(iter(designs))
print(row["design_id"], row["cell_angstrom"], row["pbc"])
image = row["image"]  # PIL image
```

The `train` split is a storage convention for the full catalog, **not a curated machine-learning training split**. Closely related designs are present; split by generator and parameter neighborhood when evaluating generalization.

## Read coordinates into ASE

Each metadata row provides the tar path, member name, byte offset and compressed size. You can download and extract a full shard with standard tools, or use the included helper to retrieve one member by HTTP range:

```python
from huggingface_hub import hf_hub_download
import importlib.util

path = hf_hub_download("lamm-mit/graphene-design-universe-64k",
                       "tools/load_structure.py", repo_type="dataset")
spec = importlib.util.spec_from_file_location("gdu", path)
gdu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdu)

atoms = gdu.load_structure(1234)
print(atoms.get_chemical_formula(), atoms.cell, atoms.pbc)
atoms.write("design.extxyz")
# Example conversion to a LAMMPS atomic data file:
atoms.write("design.data", format="lammps-data", atom_style="atomic", masses=True)
```

When converting to LAMMPS data, also set the intended boundaries in the LAMMPS input (here `boundary p p f`); that file format does not carry the ASE periodicity flags.

The helper verifies the uncompressed file checksum. Bond topology is **not encoded as permanent MD bonds**; a reactive force field should infer interactions from the coordinates. Extxyz files contain no velocities, thermal state or relaxation history.

## Embedding interpretation

The embedding uses measured occupancy spectra, coarse occupancy, density distributions, directional profiles, cell dimensions, porosity and local coordination statistics. Descriptors are weighted as declared in `code/embed_designs.py`, compressed by a 64-component PCA, and embedded using 3D UMAP with 45 neighbors, `min_dist=0.16`, `spread=1.5`, 400 epochs and seed 41. A rigid rotation and one uniform scale are applied for display. **Family labels and mechanical properties do not enter the embedding.** Colors identify design groups only after fitting.

Axes are arbitrary. Nearby points often have similar geometry under these descriptors, but global distances and cluster shapes are approximate and should not be read as a quantitative materials law. This is neither an energy landscape nor the hidden state of an AI agent. The explorer's neighborhood thumbnails identify proximity in the displayed embedding, not predicted mechanical equivalence.

## Validation and provenance

The release checks every design's source mask, cell and atom count; every exported coordinate is parsed back and compared exactly against the saved geometry. Independent ASE roundtrip checks cover every coordinate shard plus the largest and smallest designs. Shard-member offsets, compression and checksums are also verified. Reports are in `provenance/`. Generator equivalence checks compare the accelerated geometry implementation with the copied research generators.

All earlier atlas designs retain their IDs and saved coordinates. The original paper simulations, manuscript and movies are unchanged. The public `v1.0.0` tag pins the initial release; the Space uses this tag for coordinate and image retrieval. Dataset preparation, expansion and embedding are subsequent work and must not be attributed to the original Phase-I experiment.

## Reproduce the visuals and geometries

`tools/reproduce.py` reconstructs one design from its preserved site mask; `tools/regenerate_design.py` regenerates it from the saved generator parameters and verifies the resulting geometry digest. `tools/restore_workdir.py` restores the inputs for the movie scripts. These require the packages listed in `code/requirements.txt`. The supplied embedding can be reused without refitting UMAP. See the tool docstrings for commands.

## Use and future properties

Use this library for geometry exploration, design retrieval, atomistic preprocessing, methodological benchmarks or selecting candidates for subsequent calculations. For new properties, join by `design_id` and record the calculator, force field/version, loading direction, boundary conditions, temperature, strain history, relaxation settings and quality criteria. Keep properties from different protocols distinct. Nothing in this release establishes strength, ductility, toughness, manufacturability or thermodynamic stability.

## License and attribution

Released by **LAMM, MIT** under **Apache License 2.0**, at the author's direction. See `LICENSE` and `NOTICE`. Third-party libraries retain their own licenses. Cite this dataset URL and the exact Hub revision used so results remain reproducible. No unverified paper title or DOI is assigned here.

## Movies

The expanded 64,000-design tour is 2 minutes, silent and entirely text-free on a black background.

- [4K MP4](media/Graphene_64000_Design_Universe_NoText_3840x2160.mp4)
- [1080p MP4](media/Graphene_64000_Design_Universe_NoText_1920x1080.mp4)
