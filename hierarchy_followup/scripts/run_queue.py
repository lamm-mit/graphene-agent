"""Run all batch specs in a directory with N concurrent worker processes (each worker runs
scripts/run_batch.py on one spec at a time).  Specs move to done/ or failed/ subdirectories.

    python scripts/run_queue.py experiments/specs/<stage> --workers 3                    # device auto: cuda > mps > cpu
    python scripts/run_queue.py experiments/specs/<stage> --workers 4 --device cuda:0,cuda:1   # workers cycle over the listed devices

Several workers may share one GPU (the campaign ran 2-4 per Apple GPU); on NVIDIA GPUs the kernels of concurrent
processes are time-sliced unless the NVIDIA MPS daemon is running."""
from __future__ import annotations
import sys, os, glob, time, subprocess, argparse, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_dir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="device for run_batch.py; a comma-separated list assigns workers round-robin (e.g. cuda:0,cuda:1)")
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()
    done_dir = os.path.join(args.spec_dir, "done"); fail_dir = os.path.join(args.spec_dir, "failed"); log_dir = os.path.join(args.spec_dir, "logs")
    for d in (done_dir, fail_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    devices = [d.strip() for d in args.device.split(",") if d.strip()] or ["auto"]
    running = {}; launched = 0
    while True:
        pending = sorted(glob.glob(os.path.join(args.spec_dir, "*.json")))
        # launch
        while pending and len(running) < args.workers:
            spec = pending.pop(0)
            name = os.path.basename(spec)
            lf = open(os.path.join(log_dir, name.replace(".json", ".log")), "w")
            rspec = spec + ".running"
            os.rename(spec, rspec)          # mark as running BEFORE launching (child reads the renamed file)
            dev = devices[launched % len(devices)]; launched += 1
            p = subprocess.Popen([PY, os.path.join(ROOT, "scripts", "run_batch.py"), rspec, "--device", dev], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            running[rspec] = (p, lf, time.time())
            print(f"[queue] launched {name} on {dev} (pid {p.pid}); running={len(running)}", flush=True)
        # reap
        for spec in list(running.keys()):
            p, lf, t0 = running[spec]
            if p.poll() is not None:
                lf.close()
                dest = done_dir if p.returncode == 0 else fail_dir
                shutil.move(spec, os.path.join(dest, os.path.basename(spec).replace(".running", "")))
                print(f"[queue] finished {os.path.basename(spec)} rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
                del running[spec]
        if not running and not glob.glob(os.path.join(args.spec_dir, "*.json")):
            break
        time.sleep(args.poll)
    print("[queue] all done", flush=True)


if __name__ == "__main__":
    main()
