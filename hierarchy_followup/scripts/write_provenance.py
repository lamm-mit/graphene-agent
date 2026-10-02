"""Write PROVENANCE.md: which files of this tree are byte-identical copies of the phase-1 tree (checked against the
phase-1 manifest, reference/phase1_manifest.json), which were modified, and which are new; plus the environment."""
from __future__ import annotations
import sys, os, json, hashlib, time, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SKIP = {"__pycache__", ".pytest_cache", "reference", "experiments/database", "trajectories", "figures", "movies", "results", "experiments/specs", "experiments/predictions", "experiments/holdouts", "validation/results"}
RENAMED = {"PLATFORM_README.md": "README.md", "HIERARCHY_FOLLOWUP_MEMO.md": "experiments/HIERARCHY_FOLLOWUP_MEMO.md"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    man = {f["path"]: f["sha256"] for f in json.load(open(os.path.join(ROOT, "reference", "phase1_manifest.json")))["files"]}
    same, changed, new = [], [], []
    for dp, dn, fn in os.walk(ROOT):
        rel_dir = os.path.relpath(dp, ROOT)
        dn[:] = [d for d in dn if d not in SKIP and os.path.normpath(os.path.join(rel_dir, d)) not in SKIP and not d.startswith(".")]
        for f in sorted(fn):
            if f.endswith(".pyc") or f == "PROVENANCE.md" or f.startswith("."):
                continue
            rel = os.path.normpath(os.path.join(rel_dir, f)); src = RENAMED.get(rel, rel); h = sha(os.path.join(dp, f))
            (same if man.get(src) == h else changed if src in man else new).append((rel, h, src))
    try:
        import torch, ase, numpy, scipy, atomistica
        env = f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, ase {ase.__version__}, atomistica {getattr(atomistica, '__version__', '?')}"
    except Exception as e:
        env = f"(environment probe failed: {e})"
    L = ["# Provenance of the hierarchy follow-up tree (phase 3)\n",
         f"Written {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/write_provenance.py`. The tree was created on 2026-09-23 from `RESULTS_nano_v2/carbon_discovery/` "
         "(the phase-1/2 tree, archived as `carbon_discovery_20260923.zip`; code identical to the public release `lamm-mit/graphene-agent`). Only platform code was copied; "
         "no phase-1 run, trajectory, figure or report is duplicated except the 132 `record.json` + `stress_strain.csv` pairs under `reference/phase1_runs/` (read-only inputs of the assessment and the rules).\n",
         f"Environment of the follow-up: {env} (conda env `graphene-agent`, `environment.yml`).\n",
         f"## Byte-identical to the phase-1 manifest ({len(same)} files)\n", "| file | SHA-256 |", "|---|---|"]
    L += [f"| `{r}` | `{h[:16]}…` |" for r, h, _ in same]
    L += [f"\n## Modified relative to the phase-1 manifest ({len(changed)} files)\n", "| file | SHA-256 (this tree) | change |", "|---|---|---|"]
    why = {"atomistics/structures/design_space.py": "phase-3 additions: `add_crack`, family `seam_pores`, `generate` handles crack keys, `offset` for `slit_array` and for the fine lattice of `nanomesh_hier` (defaults reproduce phase 1; tested)",
           "experiments/db.py": "read-only access to the phase-1 records (`reference_records`, `stress_strain`, `trajectory_path`); `load_record` falls back to the reference copy",
           "analysis/campaign_analysis.py": "`curve` reads trajectories through `db.trajectory_path`", "analysis/fracture_viz.py": "trajectory path through `db.trajectory_path`",
           "scripts/run_batch.py": "identical to the 2026-09-23 12:58 version of the phase-1 tree (device `auto`); the manifest predates that edit",
           "scripts/run_queue.py": "identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit",
           "experiments/campaign_design.py": "identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit",
           "experiments/stage_tools.py": "identical to the 2026-09-23 12:58 version of the phase-1 tree; the manifest predates that edit",
           "environment.yml": "numeric packages from pip (single OpenMP runtime), see the file header", "requirements.txt": "copied", "HIERARCHY_FOLLOWUP_MEMO.md": "copied from experiments/ of the phase-1 tree (the memo may have been edited after the manifest)"}
    L += [f"| `{r}` | `{h[:16]}…` | {why.get(r, 'copied after the manifest was written')} |" for r, h, _ in changed]
    L += [f"\n## New in this tree ({len(new)} files)\n", "| file | SHA-256 | purpose |", "|---|---|---|"]
    purpose = {"analysis/assess_phase1.py": "assessment of the phase-1 database; defines the two-stage metrics; writes reference/phase1_reference_numbers.json",
               "experiments/campaign_hierarchy_followup.py": "the follow-up campaign: designs, geometry analysis, pre-registered predictions, specs",
               "analysis/hierarchy_followup_analysis.py": "evaluation against the pre-registration, tables, figures, fracture panels",
               "tests/test_followup.py": "tests of the additions (phase-1 regeneration, cracks, seams, registry offsets, curve metrics)",
               "scripts/write_provenance.py": "this file", "scripts/run_campaign.sh": "launches the two queues (T1/T4/T2, then T3) with nohup", "scripts/merge_runs_from.py": "copied from the phase-1 tree (post-dates its manifest)",
               "README.md": "runbook of the follow-up", "ASSESSMENT_phase1.md": "generated by analysis/assess_phase1.py"}
    L += [f"| `{r}` | `{h[:16]}…` | {purpose.get(r, '')} |" for r, h, _ in new]
    open(os.path.join(ROOT, "PROVENANCE.md"), "w").write("\n".join(L) + "\n")
    print(f"identical {len(same)}, modified {len(changed)}, new {len(new)} -> PROVENANCE.md")


if __name__ == "__main__":
    main()
