"""Merge simulation results produced in another copy of this tree (e.g. a local working copy on a second machine)
into this database.  Copy-only: run directories, trajectories and spec logs that exist in the source and not here are
copied; existing run ids are never overwritten or deleted; the index is rebuilt afterwards.

    python scripts/merge_runs_from.py /path/to/other/carbon_discovery            # dry run: lists what would be copied
    python scripts/merge_runs_from.py /path/to/other/carbon_discovery --apply    # copy and rebuild index.jsonl / summary.csv
    python scripts/merge_runs_from.py ... --campaign hierarchy                    # restrict to one campaign
"""
from __future__ import annotations
import os, sys, json, glob, shutil, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="root of the other carbon_discovery tree")
    ap.add_argument("--apply", action="store_true", help="copy (default: dry run)")
    ap.add_argument("--campaign", default=None, help="only runs whose record has this campaign")
    ap.add_argument("--specs", action="store_true", help="also copy experiments/specs/<stage dirs> that do not exist here (specs, done/, failed/, logs/)")
    a = ap.parse_args()
    src = os.path.abspath(a.source)
    if os.path.samefile(src, ROOT):
        sys.exit("source is this tree")
    src_runs = os.path.join(src, "experiments", "database", "runs"); src_traj = os.path.join(src, "trajectories")
    if not os.path.isdir(src_runs):
        sys.exit(f"no database at {src_runs}")
    new, skipped, traj_new, traj_missing = [], [], [], []
    for d in sorted(glob.glob(os.path.join(src_runs, "run_*"))):
        rid = os.path.basename(d); rec_path = os.path.join(d, "record.json")
        if not os.path.exists(rec_path):
            continue
        rec = json.load(open(rec_path))
        if a.campaign and rec.get("campaign") != a.campaign:
            continue
        if os.path.exists(os.path.join(db.RUNS_DIR, rid)):
            skipped.append(rid); continue
        new.append((rid, rec))
        t = os.path.join(src_traj, rid + ".npz")
        (traj_new if os.path.exists(t) else traj_missing).append(rid)
    print(f"source: {src}\n  new runs: {len(new)}   already here (skipped): {len(skipped)}   trajectories to copy: {len(traj_new)}   runs without trajectory in source: {len(traj_missing)}")
    for rid, rec in new:
        m = rec.get("metrics") or {}
        print(f"  + {rid}  {rec.get('campaign')}/{rec.get('stage')}  {rec.get('name')}  status={rec.get('status')}  sigma={m.get('strength_Nm', float('nan')):.2f}")
    spec_dirs = []
    if a.specs:
        for d in sorted(glob.glob(os.path.join(src, "experiments", "specs", "*"))):
            if os.path.isdir(d) and not os.path.exists(os.path.join(ROOT, "experiments", "specs", os.path.basename(d))):
                spec_dirs.append(d)
        print(f"  spec directories to copy: {[os.path.basename(d) for d in spec_dirs]}")
    if not a.apply:
        print("dry run; add --apply to copy"); return
    os.makedirs(db.RUNS_DIR, exist_ok=True); os.makedirs(db.TRAJ_DIR, exist_ok=True)
    for rid, rec in new:
        shutil.copytree(os.path.join(src_runs, rid), os.path.join(db.RUNS_DIR, rid))
        t = os.path.join(src_traj, rid + ".npz")
        if os.path.exists(t):
            shutil.copy2(t, os.path.join(db.TRAJ_DIR, rid + ".npz"))
    for d in spec_dirs:
        shutil.copytree(d, os.path.join(ROOT, "experiments", "specs", os.path.basename(d)))
    db.rebuild_index()
    print(f"copied {len(new)} runs and {len(traj_new)} trajectories; index rebuilt ({len(db.all_records())} records)")


if __name__ == "__main__":
    main()
