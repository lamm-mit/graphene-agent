#!/bin/bash
# Package the repository (platform, phase-2 analysis, slides) into one ZIP archive (excluding caches). Run from anywhere.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/make_manifest.py
NAME=carbon_discovery_$(date +%Y%m%d)
cd ..
rm -f "$NAME.zip"
zip -r -q "$NAME.zip" carbon_discovery paper_analysis slides prompt README.md PROVENANCE.md LICENSE LICENSE-DATA.md CITATION.cff THIRD_PARTY_NOTICES.md -x "*/__pycache__/*" "*.pyc" "*/.pytest_cache/*" "*/report/*.aux" "*/report/*.fls" "*/report/*.fdb_latexmk" "*/report/*.out" "*/report/*.toc" "*/report/*.bbl" "*/report/*.blg" "*/report/*.log" "*/node_modules/*" "paper_analysis/figures/*.log"
ls -la "$NAME.zip"
