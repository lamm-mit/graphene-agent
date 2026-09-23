"""Download the large data files of this project from the Hugging Face dataset and put them in place.

    python scripts/fetch_data_from_huggingface.py --what all            # trajectories + large SVG figures
    python scripts/fetch_data_from_huggingface.py --what trajectories   # 132 npz files -> carbon_discovery/trajectories/
    python scripts/fetch_data_from_huggingface.py --what svg            # 47 SVG files  -> their original locations

Requires `pip install huggingface_hub`.  The dataset id can be overridden with --repo.
Trajectory format (npz): positions, cells, peratom_energy, peratom_virial, coordination, eps_x, eps_y, sigma_xx, sigma_yy,
sigma_xy (eV/A^2; x 16.0218 = N/m), energy, n_bonds, n_broken_cum, bonds_<k>; reader: carbon_discovery/analysis/fracture_viz.load_traj.
"""
from __future__ import annotations
import argparse, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = "lamm-mit/model-builds-a-model-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["all", "trajectories", "svg"], default="all")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset id")
    ap.add_argument("--keep-cache", action="store_true", help="keep the downloaded snapshot next to the repository")
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    patterns = []
    if a.what in ("all", "trajectories"): patterns.append("trajectories/*.npz")
    if a.what in ("all", "svg"): patterns.append("large_svg_figures/**")
    patterns.append("SHA256SUMS.txt")
    local = snapshot_download(repo_id=a.repo, repo_type="dataset", allow_patterns=patterns)
    n = 0
    tdir = os.path.join(local, "trajectories")
    if os.path.isdir(tdir) and a.what in ("all", "trajectories"):
        dst = os.path.join(ROOT, "carbon_discovery", "trajectories"); os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(tdir)):
            if f.endswith(".npz"):
                shutil.copy2(os.path.join(tdir, f), os.path.join(dst, f)); n += 1
        print(f"trajectories: {n} files -> {dst}")
    m = 0
    sdir = os.path.join(local, "large_svg_figures")
    if os.path.isdir(sdir) and a.what in ("all", "svg"):
        for dp, _, fns in os.walk(sdir):
            for f in fns:
                rel = os.path.relpath(os.path.join(dp, f), sdir)      # e.g. carbon_discovery/figures/progression/x.svg
                dst = os.path.join(ROOT, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(dp, f), dst); m += 1
        print(f"large SVG figures: {m} files restored to their original locations")
    print("done. Verify with:  (cd <dataset snapshot> && shasum -a 256 -c SHA256SUMS.txt)  ->", local)


if __name__ == "__main__":
    main()
