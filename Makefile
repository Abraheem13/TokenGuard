# Reproduction entry points. `GPQA_PASSWORD=<password> make all` regenerates every
# result table and figure from the committed probe streams and checks that nothing
# differs from the committed versions. Everything runs on a CPU; figures need
# pdfLaTeX. The password of the GPQA-Diamond archive is available on request from
# abraheemrashid@outlook.com (see README).
.PHONY: all install data results figures check test lint clean help

all: results figures check ## regenerate every table and figure, then verify them

install: ## install the package and the analysis dependencies
	python -m pip install -r requirements.txt
	python -m pip install -e .

data: ## extract the GPQA-Diamond probe files (needs GPQA_PASSWORD)
	bash scripts/unpack_gpqa.sh

results: ## regenerate every result file in experiments/ntc
	bash scripts/run_analyses.sh

figures: ## regenerate the data figures and render all figures to PDF and PNG
	python scripts/make_figures.py
	bash figures/build.sh

check: ## fail if any regenerated table or figure source differs from the committed one
	@git diff --exit-code --stat -- experiments figures/src \
	  && echo "All result files and figure sources match the committed versions."

test: ## run the unit tests
	python -m pytest

lint: ## static checks
	ruff check scripts src tests

clean: ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

help: ## list the targets
	@grep -E '^[a-z]+:.*## ' Makefile | sed 's/:.*## /\t/'
