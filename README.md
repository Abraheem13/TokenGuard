# TokenGuard

**A multi-timescale, online-learning LLM router — a practical adaptation of
Nested Learning (Behrouz et al., NeurIPS 2025) — shipped as an
OpenAI-compatible drop-in proxy.**

MSc Artificial Intelligence dissertation · Brunel University London
Author: Raja Abraheem Rashid · Supervisor: Dr. Yongmin Li

---

## What it does

TokenGuard sits between any OpenAI-SDK client and a pool of locally served
open-weight models (Qwen3 tiers via vLLM). For each request it predicts the
*cheapest model capable of answering correctly* and forwards the call. Its
defining feature is that the routing policy **keeps learning in production,
on three update frequencies**:

| Level | Component | Updates |
|-------|-----------|---------|
| L1 · FAST | LinUCB contextual-bandit head | every request, from reward `r = quality − λ·cost` |
| L2 · MID | EMA of the fast head | continuous decay (stabiliser) |
| L3 · SLOW | Contrastive query encoder + per-LLM embeddings (RouterDC-style, Qwen3-0.6B) | LoRA refresh every N≈500 requests from the replay buffer |

This instantiates Nested Learning's core principle — *update frequency
defines the optimisation level* (`update level ℓ iff t mod Cℓ = 0`) — for the
LLM-routing problem. A conventional frozen router is the degenerate
single-frequency case.

Research training/evaluation runs entirely on **RouterBench precomputed
outputs** (405K samples × 11 models with cost + correctness), so the project
requires **zero API budget**; local model serving is needed only for the live
demo.

## Quick start

```bash
git clone <repo-url> && cd tokenguard
make setup     # venv + pinned deps + editable install + env check
make data      # download RouterBench, canonicalise, print Day-1 gate report
make test      # run the test suite
```

## Repository layout

```
configs/            single-source-of-truth YAML configs
src/tokenguard/
  config.py         typed config loader (rejects unknown keys)
  utils/            seeding, logging
  data/             RouterBench loader + canonicaliser   (Day 1)
  eval/             AIQ, cost–quality frontier, runner   (Day 2)
  routers/          static, MF, BERT, contrastive, LinUCB, nested (Days 2–5)
  online/           stream simulator, replay buffer, shift experiment (Day 5)
  proxy/            FastAPI OpenAI-compatible gateway + telemetry (Day 6)
  dashboard/        Streamlit monitoring app             (Day 7)
scripts/            one runnable script per day, each ending in a GATE report
tests/              pytest suite (network-free; synthetic fixtures)
experiments/        results tables, figures, telemetry DB
docs/PROJECT_PLAN.md  the 7-day build plan with go/no-go gates
```

## Reproducibility

* Every script seeds Python/NumPy/Torch from `configs/default.yaml`.
* Every results file is prefixed with `experiment.tag`.
* The detected RouterBench schema is persisted alongside the processed
  parquet (`*.schema.json`) and reported in the dissertation.
* `make test` runs without network access.

## Key references

RouterBench (arXiv:2403.12031) · RouteLLM (arXiv:2406.18665, ICLR 2025) ·
RouterDC (NeurIPS 2024) · Nested Learning / HOPE (NeurIPS 2025) ·
LinUCB (Li et al., WWW 2010).

## License

MIT — see `LICENSE`.
