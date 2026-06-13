# TokenGuard — 7-Day Build Plan

> **Project:** TokenGuard v2 — a multi-timescale, online-learning LLM router
> (Nested-Learning-inspired), shipped as an OpenAI-compatible drop-in proxy.
> **Author:** Raja Abraheem Rashid · MSc AI, Brunel University London
> **Supervisor:** Dr. Yongmin Li

Each day ends with a **gate**: a concrete, verifiable artifact. Do not start
the next day until the gate passes. Modules are organised by *function*, not
by day — this table maps days to the modules they deliver.

| Day | Theme | New modules / files | Gate (must pass) |
|-----|-------|---------------------|------------------|
| **1** | Environment, repo, data | `setup_env.sh`, `requirements.txt`, `configs/default.yaml`, `src/tokenguard/config.py`, `utils/{seed,logging}.py`, `data/routerbench.py`, `scripts/day1_setup_check.py`, `scripts/day1_download_data.py`, tests | `make check` passes (env + GPU report); RouterBench downloaded, parsed into canonical parquet; dataset stats table printed; `make test` green |
| **2** | Evaluation harness + static baselines | `eval/metrics.py` (AIQ, cost–quality frontier, APGR), `eval/runner.py`, `routers/base.py`, `routers/static.py` (random / always-small / always-large / oracle), `scripts/day2_static_baselines.py` | Frontier plot with 4 static baselines saved to `experiments/figures/`; AIQ table in `experiments/results/` |
| **3** | Learned baselines (the "old way") | `routers/mf_router.py` (RouteLLM-style matrix factorisation), `routers/bert_router.py` (the rejected baseline — kept to beat it), `scripts/day3_learned_baselines.py` | MF router dominates random on AIQ; both baselines on the frontier plot |
| **4** | Modern offline router (SLOW level) | `routers/encoder.py` (Qwen3-0.6B pooled embeddings), `routers/contrastive_router.py` (RouterDC-style dual contrastive loss + per-LLM embeddings), `scripts/day4_train_contrastive.py` | Contrastive router ≥ MF on AIQ on held-out split — else fall back to LinUCB-on-frozen-embeddings (SAFE path) |
| **5** | Nested online loop (FAST + MID + SLOW) | `routers/linucb.py`, `online/replay_buffer.py`, `online/stream.py`, `online/shift.py` (sequential-domain stream), `routers/nested_router.py` (3-level orchestrator), `scripts/day5_online_experiments.py` | Online adaptation curve improves over stream; multi-timescale beats fast-only under shift; 4-way ablation table saved |
| **6** | Product: proxy + telemetry | `proxy/app.py` (FastAPI, `/v1/chat/completions`), `proxy/backends.py` (vLLM / Ollama), `proxy/telemetry.py` (SQLite, doubles as replay buffer) | `curl` an OpenAI-format request → routed response with `x-tokenguard-*` headers; rows visible in SQLite |
| **7** | Dashboard + final runs | `dashboard/app.py` (Streamlit), `scripts/day7_final_runs.py` (frozen seeds, all figures), README polish, repo tag `v1.0` | End-to-end demo; all dissertation figures regenerate from one command |

## Daily working protocol
1. `git pull` → create branch `dayN-<topic>`.
2. Add the day's files; run `make test` and the day's script.
3. Verify the gate; commit with message `dayN: <gate evidence>`; merge.
4. Record results (numbers, figure paths) in `docs/LAB_NOTEBOOK.md`.

## Risk ladder (cut in this order — never cut the Day-5 ablation)
1. Day 4 underperforms → SAFE path: LinUCB on frozen embeddings (still multi-timescale).
2. Day 5 online gain unclear → headline pivots to "modern router + rigorous study".
3. GPU constrained → research needs no big-model inference (RouterBench is precomputed); demo on 2 tiers.
