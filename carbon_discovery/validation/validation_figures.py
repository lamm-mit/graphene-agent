"""Publication figures for the force-field validation: (1) energy/force/stress agreement with the
authoritative implementation per configuration class, (2) fast vs reference precision incl. near
fracture, (3) material properties (tensile curves zigzag/armchair, elastic constants), (4) convergence
(minimiser tolerance, strain step, system size), (5) reference-backend AQS comparison."""
from __future__ import annotations
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(ROOT, "validation", "results"); FIG = os.path.join(ROOT, "figures", "validation")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})


def save(fig, name, caption):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=220 if ext == "png" else None, bbox_inches="tight")
    open(os.path.join(FIG, f"{name}.caption.txt"), "w").write(caption)
    plt.close(fig)


def fig_agreement():
    rows = json.load(open(os.path.join(RES, "energy_force_vs_reference.json")))
    labels = [r["config"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    y = np.arange(len(rows))
    ax[0].barh(y, [max(r["dE_per_atom"], 1e-16) for r in rows], color="#3a7ca5"); ax[0].set_xscale("log"); ax[0].set_xlabel("|E_torch − E_Atomistica| / atom (eV)"); ax[0].axvline(1e-9, color="r", ls="--", lw=0.8, label="tolerance")
    ax[1].barh(y, [max(r["max_abs_dF"], 1e-16) for r in rows], color="#e07a5f"); ax[1].set_xscale("log"); ax[1].set_xlabel("max |F_torch − F_Atomistica| (eV/Å)"); ax[1].axvline(1e-8, color="r", ls="--", lw=0.8)
    ax[2].barh(y, [max(r["max_abs_dstress"], 1e-18) for r in rows], color="#81b29a"); ax[2].set_xscale("log"); ax[2].set_xlabel("max |σ_torch − σ_Atomistica| (eV/Å³)")
    for a in ax:
        a.set_yticks(y); a.grid(alpha=0.3, axis="x")
    ax[0].set_yticklabels(labels, fontsize=7); ax[1].set_yticklabels([]); ax[2].set_yticklabels([])
    for i, r in enumerate(rows):
        ax[1].text(r["max_abs_dF"] * 3, i, f"|F|max={r['max_abs_F']:.3g}", va="center", fontsize=6, color="0.3")
    ax[0].legend()
    fig.suptitle("TorchRebo2Scr (cpu float64) vs Atomistica Rebo2Scr: energies, forces and stress for ordinary and difficult configurations", fontsize=9)
    save(fig, "agreement_with_reference", "Agreement between the PyTorch implementation of screened REBO2 (cpu, float64) and the authoritative Atomistica Rebo2Scr implementation for 12 configuration classes including pore edges, crack tips and near-rupture/post-rupture frames taken from AQS trajectories. Bars: per-atom energy difference, maximum force-component difference (annotated with the largest force magnitude of the configuration) and maximum stress-component difference. Dashed lines: acceptance tolerances (1e-9 eV/atom, 1e-8 eV/A). All differences are at the level of double-precision round-off.")


def fig_precision():
    pt = json.load(open(os.path.join(RES, "precision_tests.json")))
    c = pt["cases"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    y = np.arange(len(c)); lab = [x["case"] for x in c]
    ax[0].barh(y, [abs(x["dE_per_atom_eV"]) for x in c], color="#3a7ca5"); ax[0].set_xscale("log"); ax[0].set_xlabel("|ΔE| per atom (eV), mps f32 − cpu f64"); ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=7)
    ax[1].barh(y, [x["max_abs_dF_eV_A"] for x in c], color="#e07a5f", label="max |ΔF|"); ax[1].barh(y, [x["rms_dF_eV_A"] for x in c], color="#f2cc8f", height=0.4, label="rms ΔF"); ax[1].set_xscale("log"); ax[1].set_xlabel("force difference (eV/Å)"); ax[1].axvline(0.02, color="r", ls="--", lw=0.8, label="FIRE tolerance fmax"); ax[1].legend(); ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[2].barh(y, [x["max_abs_dsigma_Nm"] for x in c], color="#81b29a"); ax[2].set_xscale("log"); ax[2].set_xlabel("max |Δσ| 2D stress (N/m)"); ax[2].set_yticks(y); ax[2].set_yticklabels([])
    for i, x in enumerate(c):
        ax[2].text(x["max_abs_dsigma_Nm"] * 2, i, f"σxx={x['sigma_xx_ref_Nm']:.1f} N/m", va="center", fontsize=6, color="0.3")
    for a in ax: a.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Fast discovery mode ({pt['fast_device']} float32) vs reference mode (cpu float64) on relaxed, pre-damage, peak-load and post-peak configurations", fontsize=9)
    save(fig, "precision_fast_vs_reference", "Numerical precision of the fast discovery mode (Apple MPS, float32) relative to the reference mode (cpu, float64, identical to Atomistica) for frames taken from three AQS trajectories (progressive nanomesh, precracked sheet, pristine zigzag graphene): relaxed, pre-damage, peak-load, post-peak and final frames. Energy differences are ~1e-6 eV/atom, force differences ~1e-4 eV/A (two orders of magnitude below the minimisation tolerance of 0.02 eV/A), 2D-stress differences ~1e-5 N/m.")


def fig_material():
    mp = json.load(open(os.path.join(RES, "material_properties.json")))
    tc = mp["tensile_curves"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for o, col in [("zigzag", "#1f77b4"), ("armchair", "#d62728")]:
        r = tc[f"atomistica_{o}"]; ax[0].plot(r["strain"], r["sxx"], "-", color=col, lw=2, label=f"Atomistica, {o}")
        t = tc[f"torch_cpu64_{o}"]; ax[0].plot(t["strain"], t["sxx"], "k--", lw=1, label=f"Torch cpu f64, {o}" if o == "zigzag" else None)
        f = tc[f"torch_fast32_{o}"]; ax[0].plot(f["strain"], f["sxx"], ":", color="0.4", lw=1, label=f"Torch mps f32, {o}" if o == "zigzag" else None)
        ax[1].plot(r["strain"], np.abs(np.array(t["sxx"]) - np.array(r["sxx"])), "-", color=col, label=f"cpu f64, {o}")
        ax[1].plot(r["strain"], np.abs(np.array(f["sxx"]) - np.array(r["sxx"])), "--", color=col, label=f"mps f32, {o}")
    ax[0].set_xlabel("engineering strain (uniaxial strain, transverse fixed)"); ax[0].set_ylabel("2D stress σxx (N/m)"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("homogeneous tensile response (relaxed internal coordinates)", fontsize=8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("engineering strain"); ax[1].set_ylabel("|Δσ| vs Atomistica (N/m)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    # elastic constants and lattice bar chart
    keys = ["a0", "bond_length", "cohesive_energy_per_atom"]
    lab = ["Atomistica", "Torch cpu f64", "Torch mps f32"]; src = ["atomistica", "torch_cpu64", "torch_fast32"]
    txt = "\n".join([f"{l:14s} a0={mp[s]['a0']:.6f} Å  bond={mp[s]['bond_length']:.6f} Å  Ecoh={mp[s]['cohesive_energy_per_atom']:.6f} eV" for l, s in zip(lab, src)])
    txt += "\n\n" + "\n".join([f"{l:14s} C11={mp[s]['elastic']['C11']:.2f} C12={mp[s]['elastic']['C12']:.2f} Y2D={mp[s]['elastic']['Y2D']:.2f} N/m  ν={mp[s]['elastic']['nu']:.4f}" for l, s in zip(lab, src)])
    txt += "\n\nexperiment (Lee 2008): Y2D ≈ 340 N/m, σmax ≈ 42 N/m; ν(graphene) ≈ 0.17\n(REBO2 underestimates Y2D and overestimates ν — a documented\nlimitation of the published potential, not of the port)"
    ax[2].axis("off"); ax[2].text(0.0, 0.98, txt, family="monospace", fontsize=6.5, va="top")
    save(fig, "material_properties", "Lattice constant, bond length, cohesive energy, elastic constants and homogeneous tensile response (zigzag and armchair loading, fixed transverse strain, relaxed internal coordinates) of graphene from the Torch implementation compared with Atomistica. Left: stress-strain curves; middle: absolute stress differences; right: tabulated properties with experimental reference values.")


def fig_convergence():
    p = os.path.join(RES, "convergence_summary.json")
    if not os.path.exists(p):
        return
    cv = json.load(open(p))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, xl in zip(ax, ["minimizer", "strain_step", "size"], ["FIRE force tolerance (eV/Å)", "strain increment", "cell size (Å)"]):
        d = cv.get(key)
        if not d: continue
        for m, col in [("strength_Nm", "#1f77b4"), ("failure_strain", "#d62728"), ("first_damage_strain", "#2ca02c")]:
            if m in d["series"]:
                x = d["series"]["x"]; y = d["series"][m]
                aa = a if m == "strength_Nm" else a.twinx() if m == "failure_strain" else a
                a.plot(x, [v / d["series"][m][0] if d["series"][m][0] else np.nan for v in y], "o-", color=col, label=m)
        a.set_xlabel(xl); a.set_ylabel("value relative to first setting"); a.grid(alpha=0.3); a.legend(fontsize=7); a.set_title(d["name"], fontsize=8)
        if key != "size": a.set_xscale("log")
    save(fig, "convergence", "Convergence tests: strength, failure strain and first-damage strain of a 64 A nanomesh as a function of the FIRE force tolerance (TEST 13) and of the strain increment with/without bisection refinement (TEST 14), and size dependence for periodic repeats of the same mesh pattern and for a fixed crack in cells of increasing size (TEST 15); values normalised by the first setting.")


def fig_reference_aqs():
    p = os.path.join(RES, "reference_aqs.json")
    if not os.path.exists(p):
        return
    ra = json.load(open(p))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (name, d) in zip(ax, ra.items()):
        r = d["atomistica_ase_fire"]; a.plot(r["strain"], r["sxx"], "k-", lw=2.5, label="Atomistica + ASE FIRE (reference backend)")
        t = d["torch_cpu64_ase_fire"]; a.plot(t["strain"], t["sxx"], "r--", lw=1.2, label="Torch cpu f64 + ASE FIRE")
        b = d["torch_cpu64_batched_fire"]; a.plot(b["strain"], b["sxx"], "b:", lw=1.5, label="Torch cpu f64 + batched FIRE (AQS driver)")
        m = d["torch_mps32_batched_fire"]; a.plot(m["strain"], m["sxx"], "g-.", lw=1.2, label="Torch mps f32 + batched FIRE")
        a.set_title(f"{name} (N={d['n_atoms']}); max|Δσ| cpu64: {d['torch_cpu64_batched_fire_max_abs_dsigma_Nm']:.2e}, mps32: {d['torch_mps32_batched_fire_max_abs_dsigma_Nm']:.2e} N/m", fontsize=7.5)
        a.set_xlabel("engineering strain"); a.set_ylabel("2D stress (N/m)"); a.grid(alpha=0.3); a.legend(fontsize=6.5)
    save(fig, "reference_backend_aqs", "TEST 18: complete AQS stress-strain curves (uniaxial strain, fixed transverse, 1% increments, FIRE fmax = 0.02 eV/A) for a precracked sheet and a nanomesh obtained with the authoritative backend (Atomistica + ASE FIRE) and with the Torch engine in three configurations (ASE FIRE / batched FIRE driver, cpu float64 and mps float32).")


if __name__ == "__main__":
    fig_agreement(); fig_precision(); fig_material(); fig_convergence(); fig_reference_aqs()
    print("validation figures written:", sorted(os.listdir(FIG)))
