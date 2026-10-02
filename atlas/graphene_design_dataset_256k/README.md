# Reproduction code

Use `tools/restore_workdir.py` to retrieve exact masks, metadata, images, sprites and embedding. Then run `prepare_tour.py`, `render_latent.py --preview`, `render_latent.py`, and `finish_delivery.py`. The source preserves all 256,000 actual designs; no simulation is invoked.

For individual structures, use `load_structure.py` (published coordinates), `reproduce.py` (saved occupancy mask), or `regenerate_design.py` (recipe). All three paths are geometry-only.

To replay the entire selection process, restore the earlier 64K workdir as sibling `graphene_design_dataset_64k`; `expand_diverse.py` preserves it and samples new candidates. `embed_expanded.py` refits the map and measures held-out coverage. `embed_designs.py` supplies the inherited descriptor definition; its older standalone main routine is not the 256K entry point. `package_stream.py` and `package_dataset.py` export and validate coordinates. Public production data should never be overwritten during a replay.

The movie has no text-rendering calls. Coordinates in callouts are magnified while leader lines retain map anchors. The opening no longer arranges glyphs in a rectangular contact sheet.
