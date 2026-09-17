#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

artifact_cache_dir="${TMPDIR:-/tmp}/neurocomputing-matplotlib"
mkdir -p "$artifact_cache_dir"
export MPLCONFIGDIR="$artifact_cache_dir"

python paper/verify_claims.py
python paper/make_tables.py
python paper/make_figs.py
python -m unittest discover -s tests -v

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
  printf '%s\n' 'Skipping PDF build: pdflatex and bibtex are required.'
fi
