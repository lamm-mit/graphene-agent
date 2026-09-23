"""Re-run only the Torch-driven parts of TEST 18 (after the fixed-transverse driver fix); reuse the stored Atomistica curves."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from validation.reference_aqs import torch_aqs
from atomistics.structures.design_space import precrack, nanomesh_single
p = "validation/results/reference_aqs.json"
out = json.load(open(p))
strains = np.round(np.arange(0.01, 0.161, 0.01), 4)
for name, atoms in [("precrack", precrack(Lx=40, Ly=40, crack_len=10)), ("nanomesh", nanomesh_single(Lx=48, Ly=48, pore_d=8, period=16))]:
    t0 = time.time(); tor64 = torch_aqs(atoms, "cpu", torch.float64, strains); t1 = time.time()
    tor32 = torch_aqs(atoms, "mps", torch.float32, strains); t2 = time.time()
    out[name]["torch_cpu64_batched_fire"] = tor64; out[name]["torch_mps32_batched_fire"] = tor32
    out[name]["timing_s"]["torch64"] = t1 - t0; out[name]["timing_s"]["torch32"] = t2 - t1
    a = np.array(out[name]["atomistica_ase_fire"]["sxx"])
    for k in ["torch_cpu64_batched_fire", "torch_mps32_batched_fire"]:
        b = np.array(out[name][k]["sxx"][1:1 + len(a)]); m = min(len(a), len(b))
        # compare up to (and including) the last pre-instability point of the reference
        ip = int(np.argmax(a))
        out[name][k + "_max_abs_dsigma_Nm"] = float(np.abs(a[:m] - b[:m]).max())
        out[name][k + "_max_abs_dsigma_prepeak_Nm"] = float(np.abs(a[:ip + 1] - b[:ip + 1]).max())
        print(name, k, "max|dsigma| all:", out[name][k + "_max_abs_dsigma_Nm"], "pre-peak:", out[name][k + "_max_abs_dsigma_prepeak_Nm"], flush=True)
json.dump(out, open(p, "w"), indent=1); print("saved")
