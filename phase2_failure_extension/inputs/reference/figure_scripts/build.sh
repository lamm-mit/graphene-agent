#!/bin/bash
# Regenerate all figures (main text + supplementary) from the database and compile the manuscript.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY check_overlaps.py > figures/audit.log 2>&1 || true      # generates the six data figures (with the collision resolver) and audits them
grep -E "TOTAL conflicts|conflict\(s\)" figures/audit.log
$PY extra_figures.py > figures/extra.log 2>&1                 # Fig. 1 (experiment), Fig. 2 (design space), Fig. 6 (progression), SI material
tail -1 figures/extra.log
$PY mechanism_figure.py > figures/mech.log 2>&1; grep -E "conflict\(s\)" figures/mech.log   # mechanism overview figure (fig_mechanisms.*), macros and LaTeX snippet
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error fig1_experiment.tex >/dev/null 2>&1 && gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -sOutputFile=fig1_experiment.png fig1_experiment.pdf && $PY -c "import fitz; d=fitz.open('fig1_experiment.pdf'); open('fig1_experiment.svg','w').write(d[0].get_svg_image())" && rm -f fig1_experiment.aux fig1_experiment.log)   # Figure 1 (TikZ) as pdf/png/svg
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error fig_approach.tex >/dev/null 2>&1 && gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r220 -sOutputFile=fig_approach.png fig_approach.pdf && $PY -c "import fitz; d=fitz.open('fig_approach.pdf'); open('fig_approach.svg','w').write(d[0].get_svg_image())" && rm -f fig_approach.aux fig_approach.log)   # approach figure (TikZ + images) as pdf/png/svg
latexmk -pdf -interaction=nonstopmode -halt-on-error -silent main.tex >/dev/null 2>&1 || { echo "latexmk failed; see main.log"; grep -n "^!" main.log | head; exit 1; }
echo "built: $(pwd)/main.pdf ($(grep -o 'Output written on main.pdf ([0-9]* pages' main.log | grep -o '[0-9]* pages'))"
