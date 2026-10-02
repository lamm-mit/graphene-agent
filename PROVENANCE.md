# Release provenance

This candidate starts from the existing public repository at the commit in `release/base_commit.txt`. The original `carbon_discovery/` scientific source, original results and frozen report are preserved. Its earlier provenance text is retained in `docs/ORIGINAL_PROVENANCE.md`.

New source is copied from the author's local research folders. Every curated input has a source-relative path and SHA-256 in `release/source_manifest.json`. Bulky scientific states and trajectories are delivered separately using the member-level inventories in `release/data_bundles.json`. Original source locations are recorded relative to the working-project root; the release does not depend on that root.

`release/portability_changes.json` and `.patch` describe the release-only changes. They replace machine-specific interpreter/CPU metadata probes, route figure exports into `generated/`, add font fallbacks and repair paths resulting from packaging. Force-field parameters, scientific engine implementations, original stop criteria and recorded numerical outcomes are not silently edited. Tests and generated release documentation are new release-maintenance work.

Source phases and historical predictions remain distinguishable. AQS snapshots are not collapsed into one implementation. Q9 checkpoints and strict numerical branches remain diagnostic evidence. Dataset sources reproduce geometry; they do not add simulated property labels.

The source inventory documents the research materials. It is not a replacement for historical campaign checksum manifests. Release entry points with portability changes have a separate execution manifest, where applicable.

One-time manuscript edits, working caches, credentials, duplicate review renders, videos and personal editing backups are not included as canonical release code. The current Overleaf manuscript is outside this code release. Scientific run records may retain original machine/environment metadata as historical data; portable drivers locate their actual inputs relative to this repository.

The source release is maintained on GitHub. The accompanying new scientific-data archives are separate preparation artifacts; this source commit does not upload those archives or change the existing Hugging Face datasets and Space.
