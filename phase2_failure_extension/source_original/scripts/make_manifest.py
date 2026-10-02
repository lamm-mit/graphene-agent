"""Write manifest.json: inventory of deliverables with SHA-256 checksums, environment and provenance."""
from __future__ import annotations
import sys, os, json, hashlib, time, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import db
from potentials.rebo2scr.reference.rebo2scr_parameters import Rebo2ScrParameters

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, ROOT)
            if rel in ("manifest.json",) or rel.endswith(".pyc"):
                continue
            files.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha256(p)})
    recs = db.all_records()
    manifest = {
        "project": "carbon_discovery -- generative atomistic discovery platform for 2D carbon architectures (screened REBO2)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": db.environment_record(),
        "code": db.code_version(),
        "force_field": {"name": "screened second-generation REBO (REBO2+S / Rebo2Scr)", "reference_implementation": "Atomistica 1.2.7 (commit 14a86f2c reviewed)",
                        "parameter_checksum_sha256": Rebo2ScrParameters().checksum(), "dihedral": False},
        "database": {"n_runs": len(recs), "n_completed": sum(1 for r in recs if r.get("status") == "completed"), "campaigns": sorted(set(str(r.get("campaign")) for r in recs))},
        "deliverables": {
            "application": ["app/backend/server.py", "app/frontend/index.html", "app/frontend/static/app.js", "scripts/launch_app.sh"],
            "force_field": ["potentials/rebo2scr/pytorch/torch_rebo2scr.py", "potentials/rebo2scr/reference/rebo2scr_parameters.py", "potentials/rebo2scr/pytorch/neighborlist.py", "potentials/rebo2scr/pytorch/ase_calculator.py"],
            "simulation": ["simulation/aqs/aqs.py", "simulation/minimization/fire.py", "simulation/system.py", "simulation/topology.py", "simulation/md/md.py"],
            "validation": sorted(glob.glob("validation/*.py")) + ["validation/results/validation_table.json"],
            "experiments": ["experiments/database/index.jsonl", "experiments/database/summary.csv", "experiments/predictions/", "experiments/holdouts/"],
            "figures": "figures/", "movies": "movies/", "final_designs": "final_designs/", "report": ["report/report.tex", "report/report.pdf", "report/references.bib"],
            "tests": "tests/",
        },
        "n_files": len(files), "files": files,
    }
    json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=1)
    print("manifest written:", len(files), "files")


if __name__ == "__main__":
    os.chdir(ROOT); main()
