#!/bin/bash
# Regenerate all paper and SI figures from the experiment database, audit them for text overlaps, and export the macros.
# Requires the environment of ../requirements.txt; pdflatex and PyMuPDF are needed only for the TikZ workflow figure.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY check_overlaps.py > figures/audit.log 2>&1 || true      # the six data figures (with the collision resolver) + audit
grep -E "TOTAL conflicts|conflict\(s\)" figures/audit.log
$PY extra_figures.py > figures/extra.log 2>&1                 # workflow figure, design space, progression comparison, SI material, facts.tex
tail -1 figures/extra.log
$PY mechanism_figure.py > figures/mech.log 2>&1; grep -E "conflict\(s\)" figures/mech.log || true   # mechanism overview figure + macros + snippet
if command -v pdflatex >/dev/null 2>&1; then
  (cd figures && pdflatex -interaction=nonstopmode -halt-on-error fig1_experiment.tex >/dev/null 2>&1 \
     && gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -sOutputFile=fig1_experiment.png fig1_experiment.pdf \
     && $PY -c "import fitz; d=fitz.open('fig1_experiment.pdf'); open('fig1_experiment.svg','w').write(d[0].get_svg_image())" \
     && rm -f fig1_experiment.aux fig1_experiment.log) || echo "TikZ figure not rebuilt (pdflatex/gs/PyMuPDF missing?)"
else
  echo "pdflatex not found: the TikZ workflow figure was not rebuilt (its pdf/png/svg are in figures/)"
fi
echo "done: figures/ updated"
