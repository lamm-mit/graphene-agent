---
license: apache-2.0
language:
- en
pretty_name: Graphene Design Universe 256K
size_categories:
- 100K<n<1M
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

# Graphene Design Universe 256K

**256,000 unrelaxed atomistic graphene designs, with images, full periodic cells, standard extended XYZ coordinates, reproducible recipes, geometry-quality flags and a geometry-similarity explorer.**

[Interactive 256K explorer](https://huggingface.co/spaces/lamm-mit/graphene-design-explorer-256k) · [Preserved 64K release](https://huggingface.co/datasets/lamm-mit/graphene-design-universe-64k) · [4K movie](media/Graphene_256000_Design_Universe_NoText_3840x2160.mp4)

![Actual designs in a neighborhood of the expanded geometry map](media/molecular_neighborhood.jpg)

This expansion preserves **all 64,000 prior designs, IDs and coordinate-file bytes** and adds **192,000 new designs**. It broadens the original 16 groups and adds eight hybrid motif groups. The geometries derive from an AI-built graphene modeling workflow and subsequent author–AI dataset development. **These 256,000 designs were not all simulated in the paper or discovered in its original independent single-shot experiment.** No force-field evaluation, relaxation, molecular dynamics or mechanical-property calculation was performed for this dataset.

## Contents

- Stable IDs `GDU-000000` through `GDU-255999`, one per saved coordinate realization.
- 24 groups: 8,000 designs in each of the original 16 groups; 16,000 in each of eight new hybrid groups.
- 8,440,339,308 retained carbon sites in total; 100–18,933,353 atoms per design. Cell widths span 3.69–497.28 nm.
- 384 × 384 PNG previews drawn from actual retained sites, alongside metadata in 1,000 Parquet shards of 256 rows each.
- Standard extended XYZ, independently gzip compressed inside indexed tar shards. Every metadata row gives the member path, byte range and uncompressed coordinate checksum.
- Exact lattice-occupancy masks, generator parameters, source code, original and expanded descriptor representations, and independent validation records.
- A new silent, entirely text-free 4K movie. The ordered contact-sheet opening of the 64K movie is replaced by a continuous camera approach through the geometry map.
- A geometry registry and result/protocol schemas for future mechanics. **No mechanical labels are included.** Missing values are not zero.

## More diverse geometries and descriptor gaps

The original groups cover round, square, hexagonal and oriented pores; angled slits; strut and polygon networks; pore gradients; flaws/halos and defects/seams; parallel/crossed veins; nested hierarchy; fibrous veins; composite grids; and subvein hierarchy. New candidates sample a wider continuous size/aspect/orientation range and combinations within these generators.

Eight new groups interpolate and combine geometric ideas: **morphing pores, pore–slit mixtures, curved vein hierarchies, branching networks, graded orientations, multiscale pore fields, correlated networks and hybrid domains**. These are new geometric motifs, not newly established physical mechanisms. Their periodic scalar fields select sites from pristine graphene; atoms are not displaced by interpolating between unrelated structures. The code defines every parameter, and stored masks preserve every exact geometry.

The recorded selection pools contained 288,560 candidates after screening, to select the 192,000 additions. The global uniqueness audit found 35 repeated selections produced by different parameter settings after atomic-lattice rounding. These new-cohort records were replaced from separate deterministic 16-candidate pools using their original selection strategy; their earlier versions and the repair history are preserved. No original 64K design was changed. New candidates must retain at least 100 atoms, have porosity no greater than 0.94, and lose no more than 35% of pristine sites during cleanup; these are geometric sampling screens, not stability tests. The original 64K cohort is preserved without applying new screens. Within each deterministic pool, approximately one third favor novelty in the frozen 64K descriptor space, one third approach midpoint queries between nearby designs from different groups, and one third preserve low-discrepancy parameter coverage. Descriptors are measured on the actual atom-deletion result after geometric cleanup. Nearest-neighbor candidate searches use 16 PCA coordinates and are reranked in all 64; they are approximate, not exhaustive exact nearest-neighbor calculations.

**Empty regions of UMAP are not automatically missing feasible designs.** We do not move points to fill them. We generate actual geometries, measure their descriptors and then refit the map. Cross-group midpoint queries are targets for geometric coverage, not interpolated atomic structures or proofs of physical feasibility.

On **2,971 held-out descriptor midpoint queries**, whose anchor IDs were excluded from selection-target construction, the median nearest-candidate distance changes from **0.2007** to **0.1999** in the frozen 64K descriptor coordinates. **7.3%** of queries improve; **1.6%** improve by more than 10%. This held-out audit uses exhaustive nearest-neighbor search in all 64 descriptor coordinates. These are coverage diagnostics under a declared descriptor, not mechanical-property improvements. The full query pairs, distances and matched design IDs are provided in `embedding/bridge_coverage.npz`.

Sampling is intentionally nonuniform and remains finite. Distinct coordinate realizations do not imply distinct topology modulo translation, rotation or scale. Related designs are expected; the catalog is broad, not exhaustive.

## Coordinates, cells and geometric quality

Carbon only; Cartesian coordinates and cell vectors in **Å**. The graphene lattice constant is **2.460177 Å**. All atoms initially lie at **z = 10 Å** in a cell with **Lz = 20 Å** and PBC **[true, true, false]**. The z dimension is a vacuum-cell convention, not a material thickness. Requested in-plane sizes are rounded to graphene lattice repeats; saved cell vectors are authoritative. Edges are unpassivated and no velocities or thermal state are provided.

Cleanup removes sites with fewer than two pristine-graph neighbors for at most ten rounds, then keeps the largest periodic connected component. This is deletion-based geometry processing, **not energy minimization**. 9,339 designs retain some sites with fewer than two neighbors; `fraction_coordination_lt2` and `geometric_cleanup_flag` identify them.

`periodic_winding_rank` counts independent nonzero winding cycles in the retained periodic neighbor graph. Rank 0 indicates no infinite periodic path; rank 1 indicates one independent path; rank 2 indicates two. A bond crossing a box boundary alone is not sufficient evidence of periodic connectivity. The full rank counts are `{'2': 249528, '1': 6242, '0': 230}`. `periodic_winding_x` and `periodic_winding_y` indicate nonzero winding components; a diagonal rank-1 path can have both true. These fields are geometric screening aids. They do not establish mechanical stability, equilibrium or strength.

Bond topology is not stored as permanent force-field bonds. Reactive potentials determine interactions from the coordinates. A future campaign should decide explicitly how to treat rank-0/rank-1 systems, thin ligaments and undercoordinated edges.

## Load metadata and images

```python
from datasets import load_dataset
rows = load_dataset("lamm-mit/graphene-design-universe-256k", "designs",
                    split="train", streaming=True)
row = next(iter(rows))
print(row["design_id"], row["cell_angstrom"], row["periodic_winding_rank"])
image = row["image"]
```

The `train` split stores the full catalog; it is not an independently designed machine-learning split. Separate related geometry groups or parameter neighborhoods when testing generalization.

## Retrieve one structure

```python
from huggingface_hub import hf_hub_download
import importlib.util
path = hf_hub_download("lamm-mit/graphene-design-universe-256k",
                       "tools/load_structure.py", repo_type="dataset")
spec = importlib.util.spec_from_file_location("gdu", path)
gdu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdu)
atoms = gdu.load_structure(200000)
atoms.write("design.extxyz")
# Optional conversion for a later LAMMPS workflow:
atoms.write("design.data", format="lammps-data", atom_style="atomic", masses=True)
```

The helper retrieves one compressed tar member and verifies its checksum. When using LAMMPS atomic data, specify the intended boundaries in the input (here `boundary p p f`); that file format does not preserve ASE PBC flags. No LAMMPS input or chosen force-field protocol is implied by the example.

## Embedding and movie interpretation

Descriptors consist of occupancy spectra, coarse occupancy, occupancy histograms, directional profiles, cell dimensions, porosity and coordination statistics. Declared block weights are inherited from the 64K workflow. The expanded map fits a new 64-component PCA and a 3D UMAP with 60 neighbors, `min_dist=0.18`, `spread=1.5`, 400 epochs and seed 41. One rigid rotation and uniform scale set the display orientation. Family labels and mechanical properties do not enter the fit.

The frozen 64K PCA transform is used separately for selection and coverage comparisons. It is applied consistently to both populations; the original released UMAP is preserved. The explorer's population selector shows original-only, additions-only or all designs within the **same newly fitted map**, so point positions remain fixed while comparing populations.

Axes are arbitrary, local neighborhoods are approximate, and global distances or visual white spaces are not material laws. This is not an AI hidden state or energy landscape. The browser's neighbor thumbnails use displayed-map distance; the movie's magnified examples use 64D feature neighbors, with leader lines marking their map locations. Magnification is an editorial aid; glyph sizes do not represent physical cell size. Every point represents one actual saved design.

## Validation and reproduction

The release verifies 256,000 geometry records and all exported coordinate values. Prior coordinate shards are byte-identical to the 64K release; new coordinate text is parsed back exactly. Independent ASE roundtrips cover 2,989 records, with maximum coordinate error 0.0e+00 Å. Cell/PBC, masks, atom counts, tar offsets, compression and checksums are checked. Periodic-field and graph-winding tests are included in `provenance/preflight.json`.

`tools/reproduce.py` reconstructs exact saved masks. `tools/regenerate_design.py` regenerates both inherited and hybrid recipes and verifies their geometry digest. `tools/restore_workdir.py` restores masks, images and sprites for the movie scripts. Source versions are listed in `code/requirements.txt`. A full replay runs `expand_diverse.py`, checks the global uniqueness audit, and, if needed, follows the recorded `repair_uniqueness.py` and `consolidate_generation.py` steps. Exact full selection replay additionally requires the preserved 64K workdir; the saved candidate-selection model and per-pool records document that decision process.

All original research runs, manuscript files and prior movies remain unchanged. This expansion is subsequent dataset development. Public tag `v1.0.0` pins this release, and the new Space uses it for retrieval.

## Ready for future mechanical properties

`mechanics/design_inventory.parquet` is a compact registry with IDs, coordinate checksums, cells/PBC, atom counts and geometric screening fields. `mechanics/protocol.template.json` deliberately contains unfilled settings and is not executable. `mechanics/results.schema.json` and `tools/validate_results.py` define and check future result records.

Attach properties by **design_id + initial-coordinate checksum + protocol_id + run_id**. Record force-field implementation, relaxation, degrees of freedom, loading direction, temperature, increments/rate, transverse conditions, stop/failure rules and residuals. Preserve every attempt and numerical branch. Store relaxed coordinates as run outputs; do not replace initial designs.

Maximum recorded stress, strain at the maximum, terminal strain, observed operational failure and the stress–strain integral are distinct quantities. Use null with explicit censoring when failure is unobserved. State whether the integral is signed or positive-part; do not call it fracture toughness. No strength, ductility, toughness or stability conclusion follows from this geometry-only library.

## License

Public **Apache 2.0**, released by **LAMM, MIT** at the author's direction. See `LICENSE` and `NOTICE`. Third-party libraries retain their licenses. Cite the dataset URL and exact revision used; no unverified paper DOI is assigned.

## Movies

New 150-second, 30-fps, black-background movie with no text or audio:

- [4K MP4](media/Graphene_256000_Design_Universe_NoText_3840x2160.mp4)
- [1080p MP4](media/Graphene_256000_Design_Universe_NoText_1920x1080.mp4)
