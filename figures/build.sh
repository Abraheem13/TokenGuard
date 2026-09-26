#!/usr/bin/env bash
# Render every figure in figures/src to figures/pdf (and to figures/png when
# pdftoppm is available). Needs a TeX distribution with pdfLaTeX, TikZ and pgfplots.
#
#   bash figures/build.sh
set -euo pipefail
cd "$(dirname "$0")"
command -v pdflatex >/dev/null || { echo "pdflatex not found: install a TeX distribution"; exit 1; }
export SOURCE_DATE_EPOCH=0 FORCE_SOURCE_DATE=1
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
mkdir -p pdf png
for src in src/fig_*.tex; do
  name=$(basename "$src" .tex)
  cat > "$BUILD/$name.tex" <<TEX
\\documentclass[border=4pt]{standalone}
\\ifdefined\\pdftrailerid\\pdftrailerid{}\\fi
\\input{style.tex}
\\begin{document}
\\input{src/$name.tex}
\\end{document}
TEX
  TEXINPUTS=".:$PWD:" pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$BUILD" "$BUILD/$name.tex" > "$BUILD/$name.out" 2>&1 \
    || { tail -30 "$BUILD/$name.out"; echo "failed: $name"; exit 1; }
  cp "$BUILD/$name.pdf" "pdf/$name.pdf"
  if command -v pdftoppm >/dev/null; then
    pdftoppm -png -r 150 -singlefile "pdf/$name.pdf" "png/$name"
  fi
  echo "figures/pdf/$name.pdf"
done
