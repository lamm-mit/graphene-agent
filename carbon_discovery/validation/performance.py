"""Computational performance benchmark: energy+force evaluation throughput vs system size for the
fast (MPS/CUDA float32), CPU float32 and reference (CPU float64) backends, neighbour-list build cost,
multi-process concurrency scaling, and the Atomistica reference for comparison.  Writes
validation/results/performance.json and figures/validation/performance.{png,svg,pdf}."""
from __future__ import annotations
import sys, os, json, time, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from atomistics.structures.design_space import pristine, nanomesh_single
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr, select_device
from simulation.system import BatchedSystem


def bench_engine(device, dtype, sizes, nrep=5):
    eng = TorchRebo2Scr(device=device, dtype=dtype)
    rows = []
    for L in sizes:
        atoms = nanomesh_single(Lx=L, Ly=L, pore_d=8, period=16)
        sysb = BatchedSystem([atoms], eng)
        t0 = time.perf_counter(); nl = sysb.build_neighbor_list(skin=0.3); tnl = time.perf_counter() - t0
        out = eng.evaluate(sysb.positions, sysb.cell, sysb.sid, nl)
        if device == "mps": torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(nrep):
            out = eng.evaluate(sysb.positions, sysb.cell, sysb.sid, nl, compute_stress=True); f = out["forces"]
        if device == "mps": torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / nrep
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(nrep):
                E = eng.energy(sysb.positions, sysb.cell, sysb.sid, nl)
        if device == "mps": torch.mps.synchronize()
        dtf = (time.perf_counter() - t0) / nrep
        rows.append({"device": str(eng.device), "dtype": str(dtype)[6:], "n_atoms": len(atoms), "M": nl.M, "Mb": nl.Mb, "nl_build_s": tnl,
                     "eval_energy_forces_stress_s": dt, "eval_energy_only_s": dtf, "us_per_atom": dt / len(atoms) * 1e6, "atoms_per_s": len(atoms) / dt})
        print(rows[-1])
    return rows


def bench_atomistica(sizes, nrep=3):
    from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator
    rows = []
    for L in sizes:
        atoms = nanomesh_single(Lx=L, Ly=L, pore_d=8, period=16); atoms.calc = make_reference_calculator()
        atoms.get_forces()
        t0 = time.perf_counter()
        for _ in range(nrep):
            atoms.positions[0, 0] += 1e-6   # force recalculation
            atoms.get_forces()
        dt = (time.perf_counter() - t0) / nrep
        rows.append({"device": "cpu (Fortran, Atomistica)", "dtype": "float64", "n_atoms": len(atoms), "eval_energy_forces_stress_s": dt, "us_per_atom": dt / len(atoms) * 1e6})
        print(rows[-1])
    return rows


def bench_concurrency(L=100, nproc=(1, 2, 4), device="mps"):
    script = os.path.join(ROOT, "validation", "_bench_worker.py")
    open(script, "w").write(f"""
import sys, time, torch; sys.path.insert(0, {ROOT!r})
from atomistics.structures.design_space import nanomesh_single
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
eng = TorchRebo2Scr(device={device!r}, dtype=torch.float32); a = nanomesh_single(Lx={L}, Ly={L}, pore_d=8, period=16)
s = BatchedSystem([a], eng); nl = s.build_neighbor_list(skin=0.3); eng.evaluate(s.positions, s.cell, s.sid, nl)
if {device!r} == 'mps': torch.mps.synchronize()
t0 = time.perf_counter(); n = 20
for _ in range(n): out = eng.evaluate(s.positions, s.cell, s.sid, nl); f = out['forces']
if {device!r} == 'mps': torch.mps.synchronize()
print((time.perf_counter() - t0) / n)
""")
    rows = []
    for k in nproc:
        procs = [subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) for _ in range(k)]
        times = [float(p.communicate()[0].decode().strip().splitlines()[-1]) for p in procs]
        rows.append({"n_processes": k, "mean_eval_s": float(np.mean(times)), "aggregate_evals_per_s": float(sum(1 / t for t in times))})
        print(rows[-1])
    os.remove(script)
    return rows


if __name__ == "__main__":
    import platform
    res = {"machine": platform.platform(), "torch": torch.__version__, "fast_device": str(select_device("auto"))}
    sizes = [30, 60, 100, 140, 200]
    res["torch_fast"] = bench_engine("auto", torch.float32, sizes)
    res["torch_cpu32"] = bench_engine("cpu", torch.float32, sizes[:4], nrep=3)
    res["torch_cpu64"] = bench_engine("cpu", torch.float64, sizes[:4], nrep=3)
    res["atomistica"] = bench_atomistica(sizes[:4])
    res["concurrency"] = bench_concurrency()
    # FIRE cost per iteration and typical AQS statistics are taken from the campaign database
    os.makedirs(os.path.join(ROOT, "validation", "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "validation", "results", "performance.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, lab, st in [("torch_fast", f"Torch {res['fast_device']} float32", "o-"), ("torch_cpu32", "Torch cpu float32", "s-"), ("torch_cpu64", "Torch cpu float64 (reference)", "^-"), ("atomistica", "Atomistica Fortran (cpu)", "d-")]:
        r = res[key]; ax[0].plot([x["n_atoms"] for x in r], [x["eval_energy_forces_stress_s"] * 1e3 for x in r], st, label=lab)
        ax[1].plot([x["n_atoms"] for x in r], [x["us_per_atom"] for x in r], st, label=lab)
    ax[0].set_xlabel("atoms"); ax[0].set_ylabel("energy + forces + virial (ms)"); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("atoms"); ax[1].set_ylabel("µs per atom per evaluation"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].grid(alpha=0.3)
    c = res["concurrency"]; ax[2].plot([x["n_processes"] for x in c], [x["aggregate_evals_per_s"] for x in c], "o-"); ax[2].set_xlabel("concurrent processes (fast device)"); ax[2].set_ylabel("aggregate evaluations / s (N≈3000)"); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(ROOT, "figures", "validation", "performance." + ext), dpi=200 if ext == "png" else None)
    print("done")
