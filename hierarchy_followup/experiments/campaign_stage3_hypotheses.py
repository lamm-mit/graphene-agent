"""Stage 3: competing mechanistic hypotheses formed from the Stage 1-2 observations (values quoted
from the experiment database at the time of writing).  Writes experiments/predictions/hypotheses.json."""
from __future__ import annotations
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import db

H = [
 {"id": "H-A", "title": "Net section x load-path straightness controls strength",
  "observation": "At matched porosity 0.20 the strength varies 6x (slit_90 4.0, ellipse_0 8.6, random Voronoi 8.2-9.7, fine meshes 11.7-13.3, coarse meshes 16.4-17.0, ellipse_90 19.0, slit_0 25.4 N/m). Dividing by the minimum load-bearing solid fraction across the loading direction (minSF) collapses the range to a 'net-section strength' of 12-32 N/m that is highest for straight, uniform-width ligaments parallel to the load (slit_0 31.8, square mesh 31, p32 22) and lowest for necked/curved paths (round-pore meshes 17-19, random Voronoi 12).",
  "mechanism": "sigma_max = minSF x sigma_0 / K, where sigma_0 ~ 39 N/m is the intrinsic strength and K is a geometric knockdown set by the curvature/necking of the load-bearing ligaments (K ~ 1.2 straight strips, ~2 necked round-pore ligaments, ~3 random networks); hierarchy and porosity act only through minSF and K.",
  "competing": "H-B: strength is controlled by the fraction of under-coordinated edge atoms (atomic-scale edge weakness), not by continuum stress concentration; at fixed minSF, strength decreases with edge-atom fraction and is insensitive to ligament shape.",
  "prediction_if_true": "A coarse square-pore mesh and a coarse round-pore mesh with the same minSF (pore width = square side) have square >= round (straight strips beat necked paths) despite more edge atoms in the square; strength of any new design is predicted within ~20% by minSF x 31/K.",
  "prediction_if_competing": "Round >= square at equal minSF (fewer edge atoms), and designs with equal edge-atom fraction but different ligament straightness (slit_0 vs jittered mesh, both fUC ~0.15) would have similar strength -- they differ 2x, so H-B is already disfavoured.",
  "experiments": "S4 pair 1: square vs circle coarse mesh at equal minSF and equal period; S4 pair 2: same at equal porosity; S5: vein-width sweep of nested meshes (rule of mixtures of straight veins and necked fine ligaments)."},
 {"id": "H-C", "title": "Fracture mode is set by the equivalence of parallel load paths",
  "observation": "Perfectly periodic straight-ligament structures fail in one avalanche (square mesh: 6 harmless corner events then all rows snap at 17.0 N/m; slit_0: stepwise with 3 events; clustered vacancies), whereas structures whose load-bearing paths are non-equivalent (fine round-pore meshes 15-22 events, jittered meshes 17-22, nested meshes 15-16, distributed vacancies 9) fail progressively with post-peak load retention.",
  "mechanism": "When all load-bearing ligaments are crystallographically equivalent they reach the instability at the same strain and snap together (no load transfer possible); necking, disorder or hierarchy make ligaments non-equivalent so they fail sequentially and the surviving ones carry the transferred load.",
  "competing": "H-C': the mode is set by the ligament shape alone (straight = abrupt, necked = progressive) irrespective of equivalence.",
  "prediction_if_true": "Introducing +-15% dispersion of the ligament widths into the square coarse mesh (same porosity) converts the single avalanche into a multi-step failure at nearly the same peak stress; a perfectly uniform fine mesh remains progressive only because of necking.",
  "prediction_if_competing": "The dispersed square mesh still fails in one avalanche.",
  "experiments": "S4 pair 3: square mesh with size dispersion 0.15 vs ordered; S4 pair 4: p32 round mesh with and without dispersion."},
 {"id": "H-D", "title": "Disorder trades strength for progressivity with an optimum for energy absorption",
  "observation": "Pore-position jitter 0/1/2/3 A gives 11.7/12.7/11.8/11.1 N/m but failure strains 0.13/0.21/0.24/0.22 and 6/17/22/21 damage steps; Voronoi regularity 0/0.6/1 gives 9.7/14.4/13.3 N/m and W 0.74/1.56/1.20 J/m2 (n25); random n50 8.2 N/m.",
  "mechanism": "Disorder acts through two opposing channels: it creates weak links (thin or misoriented ligaments) that lower the peak (extreme-value control), and it breaks the equivalence of load paths so that failure becomes sequential (longer tail, more work). The net effect on work to failure is positive at moderate disorder and negative when weak links dominate.",
  "competing": "H-D': disorder always lowers both strength and work (pure weak-link picture); or disorder always raises work (pure crack-path-disruption picture).",
  "prediction_if_true": "W(disorder) is non-monotonic with a maximum at intermediate disorder for both jittered meshes and Voronoi networks; seed-to-seed scatter of strength grows with disorder.",
  "prediction_if_competing": "W monotonically decreasing (weak link) or increasing (disruption).",
  "experiments": "S5: Voronoi regularity sweep 0/0.3/0.6/0.8/1.0; S6: 3 seeds each for jitter 2 A, size dispersion 0.15, Voronoi reg 0.0 and 0.6; lattice-registry replicates of the ordered mesh."},
 {"id": "H-E", "title": "Flaw tolerance beyond ~20 A: crack-tip lattice trapping caps the size effect",
  "observation": "Cracks of 10/20/30 A give 24.1/18.5/18.7 N/m (Griffith scaling would give 24.1/17.0/13.9); the 30 A crack starts breaking bonds at 7.9% strain and grows stably to 9.9% before running; a 20 A hole (19.1) equals a 20 A crack.",
  "mechanism": "In this discrete lattice the crack advances by discrete bond-breaking events with an energy barrier (lattice trapping); once the flaw exceeds a few nanometres the peak stress is set by the trapping strength of the tip bond configuration rather than by the flaw length.",
  "competing": "H-E': the plateau is a periodic-image artefact (30 A crack in a 100 A cell interacts with its images) and strength keeps decreasing with crack length in larger cells.",
  "prediction_if_true": "A 40 A crack in a 150 A cell and a 30 A crack in a 150 A cell both give 17-19 N/m; a 30 A hole gives ~18 N/m.",
  "prediction_if_competing": "Strength continues to fall (40 A: ~13 N/m in the large cell).",
  "experiments": "S4: precrack 30/40 A in 150 A cells; hole 30 A."},
 {"id": "H-F", "title": "Nested hierarchy does not add strength; the finest load-bearing level and the load-parallel veins control it",
  "observation": "Two-level meshes (12.7-14.3) lie between the fine (11.7-13.3) and coarse (16.4-17.0) single-scale limits at matched porosity; strength rises with vein width (W4/8/12: 13.2/12.8/13.7) and falls with domain size (D24/40/60: 14.3/12.8/12.7); the three-level mesh (13.0) equals the fine mesh. Fracture panels show the fine level failing first at pore edges, load-parallel veins carrying the residual load, and cracks channelling along columns next to load-perpendicular veins.",
  "mechanism": "The nested structure is a parallel arrangement of straight veins (strong, straight strips) and a necked fine mesh (weak); the fine level reaches its instability first, the veins then carry a fraction W/D of the net section; perpendicular veins act as compliant crack channels rather than crack arrestors.",
  "competing": "H-F': veins arrest cracks (compartmentalisation) so that nested meshes gain failure strain and work relative to single-scale meshes of equal strength.",
  "prediction_if_true": "Veins along the load only (vein_dirs='x') give higher strength than veins in both directions at equal porosity, and veins perpendicular only ('y') give no more than the fine mesh; the work to failure of nested meshes does not exceed that of the fine mesh with the same porosity.",
  "prediction_if_competing": "xy veins beat x-only veins (compartments) and nested meshes have larger work than single-scale meshes.",
  "experiments": "S5: vein direction x / y / xy at matched porosity; vein width sweep to W = 20 A; hierarchy x disorder and hierarchy x anisotropy interactions."},
]

if __name__ == "__main__":
    doc = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "based_on_records": len([r for r in db.all_records() if r.get("stage") in ("stage1_baselines", "stage2_reconnaissance")]), "hypotheses": H}
    os.makedirs(os.path.join(ROOT, "experiments", "predictions"), exist_ok=True)
    json.dump(doc, open(os.path.join(ROOT, "experiments", "predictions", "hypotheses.json"), "w"), indent=1)
    print("hypotheses written:", [h["id"] for h in H], "based on", doc["based_on_records"], "records")
