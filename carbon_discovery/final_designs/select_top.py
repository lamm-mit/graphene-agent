"""Select ~6-10 scientifically informative structures (not merely the strongest), generate
event-selected fracture-progression panels, before/after panels, comparative panels and movies,
and write final_designs/top_structures.json (consumed by the app and the report).

Selection categories (each filled from the database if available):
  reference state, strongest, largest work to failure, unusual Pareto member, key hierarchical
  design, most localised fracture, most progressive fracture, most flaw-tolerant, counter-example."""
from __future__ import annotations
import sys, os, json, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from experiments import db
from analysis.fracture_viz import progression_panel, before_after_panel, make_movie, load_traj, select_event_frames
from analysis.campaign_analysis import pareto_front, M
FD = os.path.join(ROOT, "final_designs"); PF = os.path.join(ROOT, "figures", "progression"); MV = os.path.join(ROOT, "movies")
os.makedirs(PF, exist_ok=True); os.makedirs(MV, exist_ok=True)


def pick(recs, key, best="max", filt=None):
    rs = [r for r in recs if (filt is None or filt(r)) and np.isfinite(M(r, key))]
    if not rs: return None
    return (max if best == "max" else min)(rs, key=lambda r: M(r, key))


def main(make_movies=True, extra_names=None, interpretations=None):
    recs = [r for r in db.all_records() if r.get("status") == "completed" and r.get("campaign") == "discovery" and r.get("trajectory")]
    by = {r["name"]: r for r in recs}
    sel = []
    def add(r, title, reason, movie=False):
        if r is None or any(s["run_id"] == r["run_id"] for s in sel): return
        sel.append({"run_id": r["run_id"], "name": r["name"], "family": r["family"], "title": title, "reason": reason, "movie": movie, "metrics": r["metrics"], "design": r["design"], "porosity": r.get("porosity"), "hierarchy_levels": r.get("hierarchy_levels")})
    add(by.get("S1_pristine_zz"), "Reference: pristine zigzag graphene", "reference state (intrinsic strength, abrupt rupture)", movie=True)
    add(by.get("S1_precrack_L20"), "Reference: precracked graphene (20 A slit)", "flaw-sensitivity reference; localised crack propagation", movie=True)
    porous = [r for r in recs if (r.get("porosity") or 0) > 0.08]
    add(pick(porous, "strength_Nm"), "Strongest porous architecture", "highest 2D strength among porous designs", movie=True)
    add(pick(porous, "work_to_failure_J_m2"), "Largest work to failure", "highest energy absorption (work-to-failure proxy)", movie=True)
    add(pick(porous, "damage_localization", "min", lambda r: M(r, "n_broken_total") >= 10), "Most delocalised / progressive fracture", "lowest damage localisation with substantial damage", movie=True)
    add(pick(porous, "damage_localization", "max", lambda r: M(r, "n_broken_total") >= 6), "Most localised fracture", "highest damage localisation", movie=False)
    hier = [r for r in recs if r["family"] == "nanomesh_hier"]
    add(pick(hier, "work_to_failure_J_m2"), "Key hierarchical design", "best-performing nested (two/three-level) mesh", movie=True)
    # unusual Pareto member: strength vs failure strain front, member with the largest failure strain that is not the strongest
    pts = [(M(r, "strength_Nm"), M(r, "failure_strain")) for r in porous]
    if pts:
        fr = pareto_front(pts, (True, True)); fr = sorted(fr, key=lambda i: pts[i][1])
        if fr: add(porous[fr[-1]], "Unusual Pareto design (max failure strain on the strength-ductility front)", "non-dominated design at the ductile end of the front")
    # flaw tolerant: highest strength among flaw-halo family / hole structures
    add(by.get("S2_hole20_rings2"), "Flaw-halo design (hole + pore rings)", "hole with surrounding pore rings vs bare hole (19.1 N/m) and uniform far-field pores (13.2 N/m): flaw-tolerance test")
    # counter-example: hierarchical design with the lowest strength among hierarchy ladder
    ladder = [r for r in recs if "hierarchy" in (r.get("tags") or [])]
    add(pick(ladder, "strength_Nm", "min"), "Counter-example within the hierarchy ladder", "weakest member of the matched-porosity hierarchy ladder")
    defaults_extra = {"S4_H2_D40_W8_veins_x": ("Hierarchy with load-parallel veins only", "discriminating design: veins along the load only (H-F)"),
                      "S7_H2_veinsX_W12_ellipseX": ("Holdout: load-aligned veins + load-aligned ellipses", "holdout design combining the two strongest orientation effects; strength under-predicted by the data-driven model (synergy)"),
                      "S7_slit_20deg": ("Holdout: slit array at 20 degrees (echelon coalescence)", "holdout that exposed a new mechanism: en-echelon slit coalescence through the ligaments; largest prediction miss")}
    for nm in list(defaults_extra) + list(extra_names or []):
        t, why = defaults_extra.get(nm, (nm, "added by analysis"))
        add(by.get(nm), t, why, movie=(nm in defaults_extra))
    ip = os.path.join(FD, "interpretations.json")
    if interpretations is None and os.path.exists(ip):
        interpretations = json.load(open(ip))
    # keep at most 10 designs: drop, in this order, categories whose mechanism is already covered by another selected design
    drop_order = ["Largest work to failure", "Hierarchy with load-parallel veins only", "Unusual Pareto design (max failure strain on the strength-ductility front)"]
    for ttl in drop_order:
        if len(sel) <= 10: break
        sel = [s for s in sel if s["title"] != ttl]
    sel = sel[:10]
    for s in sel:
        meta = progression_panel(s["run_id"], PF, "energy_rel"); s["panel"] = os.path.relpath(meta["file"], ROOT); s["frames"] = meta["frames"]
        s["before_after"] = os.path.relpath(before_after_panel(s["run_id"], PF, "energy_rel"), ROOT)
        s["caption"] = f"Fracture progression of {s['name']} ({s['family']}, N={by[s['name']]['n_atoms']}, porosity {s['porosity']:.2f}): event-selected frames (relaxed, pre-damage, first bond loss, peak load, major propagation, post-peak, final; colour = energy change per atom relative to the relaxed state, red circles = atoms that lost a bond) and the stress-strain curve with the displayed frames marked."
        s["interpretation"] = (interpretations or {}).get(s["name"], "")
        if make_movies and s.get("movie"):
            mv = os.path.join(MV, f"{s['name']}.mp4")
            if not os.path.exists(mv):
                make_movie(s["run_id"], mv, "energy_rel")
            s["movie"] = os.path.relpath(mv, ROOT)
        else:
            s["movie"] = None
    json.dump({"selected": sel, "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(FD, "top_structures.json"), "w"), indent=1, default=db._json_default)
    # copy structures of the selected designs
    for s in sel:
        rdir = os.path.join(ROOT, by[s["name"]]["run_dir"])
        for f in ["initial.extxyz", "relaxed.extxyz", "final.extxyz"]:
            shutil.copy(os.path.join(rdir, f), os.path.join(FD, f"{s['name']}_{f}"))
    print(json.dumps([(s["title"], s["name"]) for s in sel], indent=1))


if __name__ == "__main__":
    main(make_movies="--no-movies" not in sys.argv)
