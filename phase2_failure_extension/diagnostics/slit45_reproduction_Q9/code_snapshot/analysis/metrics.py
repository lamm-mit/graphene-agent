"""Mechanical / fracture metrics from an AQS result dictionary (2D units)."""
from __future__ import annotations

import numpy as np
from simulation.topology import damage_localization

EV_A2_TO_N_M = 16.0217663
EV_PER_A2_TO_J_PER_M2 = 16.0217663   # 1 eV/A^2 = 16.02 J/m^2


def compute_metrics(result, elastic_strain_max=0.02):
    eps = np.asarray(result["eps_x"], float)
    sxx = np.asarray(result["sigma_xx"], float)   # eV/A^2
    syy = np.asarray(result["sigma_yy"], float)
    epsy = np.asarray(result["eps_y"], float)
    E = np.asarray(result["energy"], float)
    n_atoms = int(result["n_atoms"])
    L0 = np.asarray(result["L0"], float)
    A0 = float(L0[0] * L0[1])
    m = {}
    # initial modulus (linear fit over small strains)
    sel = (eps <= elastic_strain_max) & (eps >= 0)
    if sel.sum() >= 3:
        p = np.polyfit(eps[sel], sxx[sel], 1)
        m["modulus_2d_Nm"] = float(p[0] * EV_A2_TO_N_M)
        pp = np.polyfit(eps[sel], epsy[sel], 1)
        m["poisson_ratio"] = float(-pp[0])
    else:
        m["modulus_2d_Nm"] = float("nan"); m["poisson_ratio"] = float("nan")
    ip = int(np.argmax(sxx))
    m["strength_Nm"] = float(sxx[ip] * EV_A2_TO_N_M)
    m["strain_at_peak"] = float(eps[ip])
    m["first_damage_strain"] = float(result.get("first_damage_strain", np.nan))
    # failure strain: strain at which stress first drops below 10% of peak after the peak (or loss of spanning)
    span = np.asarray(result["spanning"], bool)
    fail_idx = None
    for k in range(ip, len(eps)):
        if (not span[k]) or sxx[k] < 0.1 * sxx[ip]:
            fail_idx = k
            break
    m["failure_strain"] = float(eps[fail_idx]) if fail_idx is not None else float(eps[-1])
    m["failure_observed"] = fail_idx is not None
    # work to failure per area (integral of sigma d eps up to failure), and specific work per atom
    kend = fail_idx if fail_idx is not None else len(eps) - 1
    w = float(np.trapezoid(sxx[:kend + 1], eps[:kend + 1]))   # eV/A^2
    m["work_to_failure_eV_A2"] = w
    m["work_to_failure_J_m2"] = w * EV_PER_A2_TO_J_PER_M2
    m["specific_work_eV_per_atom"] = w * A0 / n_atoms
    m["energy_stored_to_failure_eV_per_atom"] = float((E[kend] - E[0]) / n_atoms)
    # post-peak behaviour
    post = sxx[ip:kend + 1]
    m["post_peak_load_retention"] = float(np.mean(post) / sxx[ip]) if len(post) > 1 else 0.0
    m["strain_range_post_peak"] = float(eps[kend] - eps[ip])
    # damage events
    nb_new = np.asarray(result["n_broken_new"], int)
    m["n_broken_total"] = int(np.asarray(result["n_broken_cum"], int)[-1])
    m["n_damage_steps"] = int((nb_new > 0).sum())
    ev = result.get("broken_bond_events", [])
    if ev:
        e_strains = np.array([e[0] for e in ev])
        # "major" damage events: steps where >= 5% of the eventual broken bonds break
        counts = np.array([nb for nb in nb_new if nb > 0])
        m["n_major_damage_events"] = int((counts >= max(2, 0.05 * m["n_broken_total"])).sum())
        m["max_single_step_broken_fraction"] = float(counts.max() / max(m["n_broken_total"], 1))
        pos = np.array([e[4] for e in ev])
        L, H, _ = damage_localization(pos, np.asarray(result["cells"][-1], float) if len(result.get("cells", [])) else np.diag([L0[0], L0[1], 1]))
        m["damage_localization"] = float(L)
        m["damage_entropy"] = float(H)
        m["fraction_atoms_damaged"] = float(len(set([e[1] for e in ev] + [e[2] for e in ev])) / n_atoms)
        m["crack_initiation_xy"] = [float(pos[0][0]), float(pos[0][1])]
        m["crack_initiation_strain"] = float(e_strains[0])
        # crack extent: spread of damage midpoints along y (perpendicular to load) and x
        m["damage_extent_x"] = float(np.ptp(pos[:, 0])) if len(pos) > 1 else 0.0
        m["damage_extent_y"] = float(np.ptp(pos[:, 1])) if len(pos) > 1 else 0.0
    else:
        m["n_major_damage_events"] = 0
        m["damage_localization"] = float("nan")
        m["fraction_atoms_damaged"] = 0.0
    # fracture mode classification
    if m["n_broken_total"] == 0:
        mode = "no damage"
    elif m["strain_range_post_peak"] <= 0.011 and m["max_single_step_broken_fraction"] > 0.5:
        mode = "abrupt brittle (single avalanche)"
    elif m["n_damage_steps"] >= 4 and m["post_peak_load_retention"] > 0.3:
        mode = "progressive (multiple events, load retained)"
    elif m["n_damage_steps"] >= 2:
        mode = "stepwise (few events)"
    else:
        mode = "abrupt brittle"
    m["fracture_mode"] = mode
    m["termination"] = result.get("termination", "")
    return m
