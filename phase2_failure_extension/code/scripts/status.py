"""Compact campaign status: queues, running batches (last step line), database counts."""
import os, glob, sys, json, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for qd in sorted(glob.glob(os.path.join(ROOT, "experiments", "specs", "*"))):
    if not os.path.isdir(qd) or os.path.basename(qd) in ("test",):
        continue
    pending = len(glob.glob(os.path.join(qd, "*.json"))); running = glob.glob(os.path.join(qd, "*.running")); done = len(glob.glob(os.path.join(qd, "done", "*.json"))); failed = len(glob.glob(os.path.join(qd, "failed", "*.json")))
    print(f"[{os.path.basename(qd)}] pending={pending} running={len(running)} done={done} failed={failed}")
    for r in running:
        log = os.path.join(qd, "logs", os.path.basename(r).replace(".json.running", ".log"))
        if os.path.exists(log):
            lines = [l for l in open(log).read().splitlines() if l.startswith("step") or "bisection" in l or "[done]" in l or "numerical" in l]
            last = lines[-1] if lines else "(starting)"
            print("   ", os.path.basename(r).replace(".json.running", ""), ":", re.sub(r"\s+", " ", last)[:170])
recs = glob.glob(os.path.join(ROOT, "experiments", "database", "runs", "*", "record.json"))
n_ok = 0; by_stage = {}
for p in recs:
    r = json.load(open(p)); st = r.get("stage"); by_stage[st] = by_stage.get(st, 0) + 1; n_ok += r.get("status") == "completed"
print(f"database: {len(recs)} records, {n_ok} completed; by stage: {by_stage}")
np_ = subprocess.run(["pgrep", "-fc", "run_batch.py"], capture_output=True, text=True).stdout.strip()
print("run_batch processes:", np_)
