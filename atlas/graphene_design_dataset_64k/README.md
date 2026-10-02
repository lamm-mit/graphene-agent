# Reproducing the design library and visuals

The copied research generators are in `source/structures/`. `fast_geometry.py` accelerates lattice construction, shape queries and geometric cleanup while preserving the masks; equivalence results are provided in `../provenance/`.

For an individual structure, use the public `tools/load_structure.py` to read standard extxyz, `tools/reproduce.py` to reconstruct the exact saved sites, or `tools/regenerate_design.py` to regenerate from the recorded parameters and check the geometry digest. The complete accepted recipes are in `provenance/manifest.json.gz` and each Parquet record.

To reproduce the movie, install `requirements.txt` and FFmpeg, then run `tools/restore_workdir.py` into a new directory. In that directory:

```sh
python embed_designs.py --reuse-embedding
python prepare_tour.py
python render_latent.py --preview
python render_latent.py
python finish_delivery.py
```

`embed_designs.py` without `--reuse-embedding` refits the PCA/UMAP map. The supplied embedding is the authoritative map for release v1.0.0; numerical library changes may affect a fresh fit. The 3D axes have no physical meaning.

`generate_atlas.py` and `expand_atlas.py` document the staged sampling history; the latter expects its preserved predecessor atlas in a sibling `graphene_latent_universe_4k` directory. These historical sampling drivers are not required to retrieve or reconstruct any published structure. No simulation is run by any of these geometry or movie scripts.

`package_dataset.py` exports cells and coordinates, creates images and Parquet tables, and performs coordinate roundtrips. `validate_population.py` records population, preservation and export coverage checks.

`add_geometry_flags.py` adds the measured graph-coordination fractions and the explicit cleanup flag to the viewer catalog after packaging. This flag identifies fewer than two graph neighbors; it is not a force-residual or energy-convergence test.
