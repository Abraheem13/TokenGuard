# TokenGuard — developer entry points
.PHONY: setup check data test lint clean

setup:            ## create venv + install everything
	bash setup_env.sh

check:            ## verify environment & GPU
	python scripts/day1_setup_check.py

data:             ## download + canonicalise RouterBench (Day 1 gate)
	python scripts/day1_download_data.py

test:             ## run the test suite
	python -m pytest tests/ -v

lint:             ## static checks
	ruff check src/ scripts/ tests/

clean:            ## remove caches (never touches data/ or experiments/)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
