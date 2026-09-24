# Reproduction entry points. Every target below runs on a CPU from the frozen
# probe streams in experiments/ntc; only `generate` needs a GPU.
.PHONY: help install test lint tables corpus clean

help:
	@grep -E '^[a-z]+:.*##' Makefile | sed 's/:.*##/ ->/'

install:          ## install the package and the analysis dependencies
	pip install -r requirements.txt && pip install -e .

test:             ## run the unit tests
	python -m pytest tests -q

lint:             ## static checks
	ruff check src scripts tests

tables:           ## regenerate every result file from the frozen probe streams
	bash scripts/run_analyses.sh

corpus:           ## regenerate the corpus inventory and track tables
	python scripts/ntc_data_inventory.py
	python scripts/ntc_corpus_tracks.py

clean:            ## remove caches (never touches experiments/)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
