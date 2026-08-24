# Build the manuscript.  `make` produces main.pdf; `make figures` rebuilds the
# result figures from the frozen artifact first (figures/ needs the probe files).
.PHONY: all figures clean
all: main.pdf
main.pdf: main.tex refs.bib $(wildcard fig_*.pdf)
	latexmk -pdf -interaction=nonstopmode main.tex
figures:
	cd figures && python3 paperfigs.py && python3 curvefig.py && python3 shiftfig.py
	cp figures/fig_*.pdf .
clean:
	latexmk -C; rm -f main.bbl
