"""Validation suite: collects TESTS 1-20 into one table (expected, observed, error, tolerance, PASS/FAIL).

Fast tests (1, 2, 3, 5, 6, 9-12, 17, 20) are executed here; long tests (4, 7, 8, 13-16, 18, 19)
are read from their result files produced by the dedicated scripts (bond_separation.py,
material_properties.py, precision_tests.py, reference_aqs.py, convergence analysis)."""
from __future__ import annotations
import sys, os, json, glob, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from ase import Atoms
RES = os.path.join(ROOT, "validation", "results")


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def test_energy_forces_vs_reference():
    """TEST 1 & 2: energies / forces / stress vs Atomistica for ordinary and difficult configurations."""
    from potentials.rebo2scr.pytorch.ase_calculator import TorchRebo2ScrCalculator
    from potentials.rebo2scr.reference.atomistica_reference import make_reference_calculator
    from atomistics.structures.design_space import pristine, nanomesh_single, precrack, vacancies
    rng = np.random.default_rng(7)
    configs = []
    g = pristine(Lx=25, Ly=25); configs.append(("pristine graphene", g))
    h = g.copy(); h.positions += rng.normal(0, 0.12, h.positions.shape); configs.append(("randomly displaced (0.12 A)", h))
    s = g.copy(); c = np.array(s.cell); c[0, 0] *= 1.18; s.set_cell(c, scale_atoms=True); configs.append(("18% strained", s))
    v = vacancies(Lx=25, Ly=25, fraction=0.03, organisation="random", seed=2); v.positions += rng.normal(0, 0.05, v.positions.shape); configs.append(("vacancies 3%", v))
    m = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16); m.positions += rng.normal(0, 0.05, m.positions.shape); configs.append(("pore edges (nanomesh)", m))
    pc = precrack(Lx=40, Ly=40, crack_len=12); c = np.array(pc.cell); c[0, 0] *= 1.10; pc.set_cell(c, scale_atoms=True); configs.append(("crack tip, 10% strain", pc))
    # near-rupture frames from stored AQS trajectories
    for name in ["precrack", "mesh_progressive", "pristine_zz"]:
        p = os.path.join(RES, "test_frames", name + ".npz")
        if os.path.exists(p):
            d = np.load(p); ip = int(np.argmax(d["sigma_xx"]))
            a = Atoms("C" * d["positions"].shape[1], positions=d["positions"][ip].astype(float), cell=d["cells"][ip], pbc=[True, True, False])
            configs.append((f"near rupture: {name} at peak stress", a))
            b = Atoms("C" * d["positions"].shape[1], positions=d["positions"][min(ip + 1, len(d['positions']) - 1)].astype(float), cell=d["cells"][min(ip + 1, len(d['positions']) - 1)], pbc=[True, True, False])
            configs.append((f"post-rupture: {name}", b))
    ref = make_reference_calculator()
    t64 = TorchRebo2ScrCalculator(device="cpu", dtype=torch.float64, skin=0.0)
    rows = []
    for label, atoms in configs:
        a = atoms.copy(); a.calc = ref; E0 = a.get_potential_energy(); F0 = a.get_forces(); S0 = a.get_stress(); e0 = a.get_potential_energies()
        b = atoms.copy(); b.calc = t64; E1 = b.get_potential_energy(); F1 = b.get_forces(); S1 = b.get_stress(); e1 = b.get_potential_energies()
        rows.append({"config": label, "n_atoms": len(atoms), "dE_per_atom": float(abs(E1 - E0) / len(atoms)), "max_abs_dE_atom": float(np.abs(e1 - e0).max()),
                     "max_abs_dF": float(np.abs(F1 - F0).max()), "max_abs_F": float(np.abs(F0).max()), "max_abs_dF_norm": float(np.abs(np.linalg.norm(F1, axis=1) - np.linalg.norm(F0, axis=1)).max()),
                     "max_abs_dstress": float(np.abs(S1 - S0).max()), "max_abs_stress": float(np.abs(S0).max())})
    return rows


def test_export_roundtrip():
    from atomistics.io import roundtrip_check
    from atomistics.structures.design_space import nanomesh_single
    a = nanomesh_single(Lx=32, Ly=32, pore_d=8, period=16)
    return roundtrip_check(a, os.path.join(RES, "roundtrip"))


def build_table():
    T = []
    def add(tid, name, expected, observed, error, tol, ok, note=""):
        T.append({"test": tid, "name": name, "expected": expected, "observed": observed, "error": error, "tolerance": tol, "status": "PASS" if ok else "FAIL", "note": note})
    # TEST 1/2
    rows = test_energy_forces_vs_reference()
    json.dump(rows, open(os.path.join(RES, "energy_force_vs_reference.json"), "w"), indent=1)
    worstE = max(r["dE_per_atom"] for r in rows); worstF = max(r["max_abs_dF"] for r in rows); worstS = max(r["max_abs_dstress"] for r in rows)
    add("TEST 1", "energy vs Atomistica Rebo2Scr (cpu f64), 12 configurations incl. pore edges, crack tips, near rupture", "identical", f"max |dE|/atom = {worstE:.2e} eV", worstE, 1e-9, worstE < 1e-9)
    add("TEST 2", "forces vs Atomistica (every component), same configurations", "identical", f"max |dF| = {worstF:.2e} eV/A", worstF, 1e-8, worstF < 1e-8, f"max |dsigma| = {worstS:.1e} eV/A^3")
    # TEST 3
    ft = _load("force_tests.json")
    if ft:
        e = max(v["max_abs_force_error"] for v in ft["TEST3_finite_difference"].values()); fm = max(v["max_abs_force"] for v in ft["TEST3_finite_difference"].values())
        add("TEST 3", "finite-difference forces (central, h=1e-4 A) and stress (h=1e-5)", "F = -dE/dR", f"max |F - F_fd| = {e:.1e} eV/A (|F| up to {fm:.0f})", e, 1e-4, e < 1e-4)
        nl = ft["TEST9_neighbor_list"]; ok = all(v["identical"] for v in nl.values())
        add("TEST 9", "cell-list neighbour list vs brute-force enumeration (pair sets)", "identical pair sets", "identical" if ok else "differ", 0 if ok else 1, 0, ok, f"{nl['nanomesh']['n_pairs_brute']} pairs")
        t = ft["TEST10_translation"]; add("TEST 10", "translation invariance (rigid shift 7.3,-11.1 A)", "dE = 0", f"dE = {t['dE']:.1e} eV, max dF = {t['max_dF']:.1e}", max(t["dE"], t["max_dF"]), 1e-9, max(t["dE"], t["max_dF"]) < 1e-9)
        t = ft["TEST11_force_balance"]; add("TEST 11", "total force balance sum_i F_i", "0", f"{t['sum_F']:.1e} eV/A", t["sum_F"], 1e-10, t["sum_F"] < 1e-10)
        t = ft["TEST12_periodic_images"]; err = max(t["dE_wrap"], t["max_dF_wrap"], t["dE_per_atom_2x_supercell"], t["max_dF_2x_supercell"])
        add("TEST 12", "periodic-image invariance (wrapping, 2x supercell)", "invariant", f"max deviation {err:.1e}", err, 1e-9, err < 1e-9)
    # TEST 4 / 16 / 17
    pt = _load("precision_tests.json")
    if pt:
        c = pt["cases"]; dE = max(abs(x["dE_per_atom_eV"]) for x in c); dF = max(x["max_abs_dF_eV_A"] for x in c); ds = max(x["max_abs_dsigma_Nm"] for x in c)
        add("TEST 4", "fast mode (mps float32) vs reference (cpu float64): energies/forces/stress on relaxed structures", "agreement within float32 round-off", f"|dE|/atom <= {dE:.1e} eV, |dF| <= {dF:.1e} eV/A, |dsigma| <= {ds:.1e} N/m", dF, 5e-3, dF < 5e-3 and dE < 1e-4)
        near = [x for x in c if any(k in x["case"] for k in ["peak", "post_peak", "pre_damage"])]
        dFn = max(x["max_abs_dF_eV_A"] for x in near); dsn = max(x["max_abs_dsigma_Nm"] for x in near)
        add("TEST 16", "precision near fracture (pre-damage, peak, post-peak frames), mps f32 vs cpu f64", "within float32 round-off", f"|dF| <= {dFn:.1e} eV/A, |dsigma| <= {dsn:.1e} N/m", dFn, 5e-3, dFn < 5e-3)
        r = pt["reproducibility_evaluation"]
        add("TEST 17", "reproducibility of repeated evaluations (fast device)", "identical up to reduction-order round-off", f"dE = {r['dE']:.1e} eV total, max dF = {r['max_dF']:.1e} eV/A", r["max_dF"], 1e-5, r["max_dF"] < 1e-5, "cpu f64 is bit-reproducible; MPS reductions are order-nondeterministic at 1e-7 relative")
    # TEST 5-8
    mp = _load("material_properties.json")
    if mp:
        a_ref = mp["atomistica"]["a0"]; a_t = mp["torch_cpu64"]["a0"]; a_f = mp["torch_fast32"]["a0"]
        add("TEST 5", "relaxed lattice constant / bond length (cpu f64 vs Atomistica; fast mode)", f"a0 = {a_ref:.6f} A (bond {a_ref/np.sqrt(3):.6f})", f"a0 = {a_t:.6f} A (f64), {a_f:.6f} A (f32)", abs(a_t - a_ref), 1e-6, abs(a_t - a_ref) < 1e-6, f"fast mode deviation {abs(a_f-a_ref):.1e} A")
        e_ref = mp["atomistica"]["cohesive_energy_per_atom"]; e_t = mp["torch_cpu64"]["cohesive_energy_per_atom"]
        add("TEST 6", "graphene cohesive energy", f"{e_ref:.6f} eV/atom", f"{e_t:.6f} eV/atom", abs(e_t - e_ref), 1e-8, abs(e_t - e_ref) < 1e-8, "REBO2 value; experiment ~ -7.4 eV/atom")
        el_ref = mp["atomistica"]["elastic"]; el_t = mp["torch_cpu64"]["elastic"]
        err = max(abs(el_t[k] - el_ref[k]) for k in ["C11", "C12", "Y2D", "nu"])
        add("TEST 7", "small-strain elastic response (C11, C12, Y2D, nu) relaxed", f"Y2D = {el_ref['Y2D']:.2f} N/m, nu = {el_ref['nu']:.4f} (Atomistica)", f"Y2D = {el_t['Y2D']:.2f} N/m, nu = {el_t['nu']:.4f}", err, 1e-3, err < 1e-3, "known REBO2 deficiency vs experiment (340 N/m, 0.17)")
        tc = mp["tensile_curves"]
        errs = []
        for o in ["zigzag", "armchair"]:
            errs.append(max(abs(np.array(tc[f"torch_cpu64_{o}"]["sxx"]) - np.array(tc[f"atomistica_{o}"]["sxx"]))))
        add("TEST 8", "armchair vs zigzag tensile response (homogeneous, relaxed, to 30% strain)", f"zz peak {tc['atomistica_zigzag']['peak_stress']:.2f} N/m, ac peak {tc['atomistica_armchair']['peak_stress']:.2f} N/m", f"zz {tc['torch_cpu64_zigzag']['peak_stress']:.2f}, ac {tc['torch_cpu64_armchair']['peak_stress']:.2f} N/m; max |dsigma| = {max(errs):.1e}", max(errs), 1e-3, max(errs) < 1e-3)
    # TEST 13-15 convergence
    cv = _load("convergence_summary.json")
    if cv:
        for tid, key in [("TEST 13", "minimizer"), ("TEST 14", "strain_step"), ("TEST 15", "size")]:
            if key in cv:
                d = cv[key]
                add(tid, d["name"], d["expected"], d["observed"], d["error"], d["tolerance"], d["pass"], d.get("note", ""))
    # TEST 18
    ra = _load("reference_aqs.json")
    if ra:
        exact = max(v["torch_cpu64_ase_fire_max_abs_dsigma_Nm"] for v in ra.values())
        pre64 = max(v.get("torch_cpu64_batched_fire_max_abs_dsigma_prepeak_Nm", v["torch_cpu64_batched_fire_max_abs_dsigma_Nm"]) for v in ra.values())
        pre32 = max(v.get("torch_mps32_batched_fire_max_abs_dsigma_prepeak_Nm", v["torch_mps32_batched_fire_max_abs_dsigma_Nm"]) for v in ra.values())
        add("TEST 18", "complete stress-strain curve: Torch engine vs Atomistica backend in an AQS (same protocol: 1% steps, fixed transverse, FIRE fmax 0.02), precrack + nanomesh", "identical curves with the same minimiser; batched driver within the fmax-induced stress uncertainty (~0.2 N/m)",
            f"Torch+ASE FIRE vs Atomistica+ASE FIRE: max |dsigma| = {exact:.1e} N/m (identical incl. fracture); batched FIRE driver: {pre64:.2f} N/m (cpu f64), {pre32:.2f} N/m (mps f32) up to the peak", pre64, 0.2, exact < 1e-6 and pre64 < 0.2, "different FIRE variants converge to states differing within fmax; post-instability branches can bifurcate")
    # TEST 19
    bs = _load("bond_separation_summary.json")
    if bs:
        worst = max(v.get("maxabs_dF_torch_cpu64", v.get("maxabs_dS_torch_cpu64", 0)) for v in bs.values())
        add("TEST 19", "bond separation E(r), F(r) through rupture (dimer, in-lattice bond, crack tip, homogeneous stretch) vs Atomistica", "identical", f"max |dF| = {worst:.1e} eV/A (cpu f64); fast f32 within {max(v.get('maxabs_dF_torch_fast32', v.get('maxabs_dS_torch_fast32', 0)) for v in bs.values()):.1e}", worst, 1e-8, worst < 1e-8, "no spurious discontinuities beyond those intrinsic to the published screening/cutoff functions")
    # TEST 20
    rt = test_export_roundtrip()
    json.dump(rt, open(os.path.join(RES, "export_roundtrip.json"), "w"), indent=1)
    ok = all(v["ok"] and (v["max_dpos_sorted_norms"] is not None and v["max_dpos_sorted_norms"] < 1e-3) for v in rt.values())
    add("TEST 20", "export/data consistency (extxyz, xyz, traj, LAMMPS data, CIF, POSCAR round trip)", "positions/cell preserved", "; ".join(f"{k}: {'ok' if v['ok'] else v['error']}" for k, v in rt.items()), max((v.get("max_dpos_sorted_norms") or 0) for v in rt.values()), 1e-3, ok)
    return T


if __name__ == "__main__":
    T = build_table()
    json.dump(T, open(os.path.join(RES, "validation_table.json"), "w"), indent=1)
    for r in T:
        print(f"{r['test']:8s} {r['status']:4s} {r['name'][:70]:70s} | {str(r['observed'])[:80]}")
