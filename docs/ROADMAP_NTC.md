# NTC — Nested Token-Budget Controller: 3-Week Roadmap

**Project:** Scaling TokenGuard → **NTC** (Nested Token-Budget Controller)
**Student:** Raja Abraheem Rashid · MSc AI · Brunel University London
**Supervisor:** Dr. Yongmin Li
**Dissertation topic:** Token-utilization optimization for LLMs
**Window:** 3 weeks (GPU available) + write-up

---

## 0. The pivot in one paragraph

TokenGuard's router (which model?) is kept and **promoted to the SLOW tier** of a
nested, multi-timescale controller that also decides **how many reasoning tokens
to spend** (budget control). A **FAST tier** reads a per-step *surprise/uncertainty*
signal during generation and halts when reasoning has converged; a **MEDIUM tier**
consolidates the realized surprise→budget mapping per query; the **SLOW tier** is
the contrastive router + budget prior, updated across the stream. This is a
genuine, equation-level instance of Nested Learning's Continuum Memory System
(token / query / stream frequencies) and Titans' surprise update — and it moves
the multi-timescale machinery onto a variable that *actually drifts* (the optimal
reasoning budget), unlike the memory-routing dead end.

**Headline deliverable:** a real-LLM demo — **match full-CoT accuracy using
~40-55% fewer reasoning tokens**, beating REFRAIN and MUR on their own metrics.

---

## 1. What we KEEP, ADD, and ARCHIVE

### KEEP (reuse — this is why routing stays in the story)
- `routers/` (contrastive, single_timescale, MF, kNN, cascade, static) — the SLOW tier
- `data/routerbench.py`, `eval/` (AIQ, runner, metrics) — evaluation backbone
- `encoders/`, `proxy/` — encoder + serving
- `online/linucb.py` — reused for the SW-UCB threshold bandit

### ADD (the NTC core)
- `budget/` — the nested budget controller (fast surprise + medium memory + slow prior)
- `llm/` — real-LLM harness (vLLM/transformers generation + token counting)
- `reasoning/` — GSM8K / MATH-500 / GPQA-Diamond loaders + answer extraction + scoring
- `baselines/` — REFRAIN, MUR, TALE, CAR, CoD, vanilla-CoT reimplementations
- `eval/pareto.py` — quality-token Pareto + matched-budget / matched-accuracy + stats

### ARCHIVE (keep for the dissertation's honest negative-result ablation, off main path)
- `online/nested_router.py`, `memory/`, `streams/` (recurrence/drift/new_model/price_shift)
  → moved to `archive/memory_routing/` (still importable for the ablation chapter,
  but out of the NTC critical path). **Not deleted** — the negative result is a
  real finding you will report.

### DELETE (genuine junk)
- root `fix_schema.sh`, `install_day2.sh` (one-off setup scripts, already applied)
- `scaffold_nestor.sh` (superseded by `scaffold_ntc.sh`)
- `scripts/day5_online.py.save`, stray `__pycache__`, `experiments/results/telemetry.db`

---

## 2. Target architecture

```
                    query
                      │
         ┌────────────▼─────────────┐
         │  SLOW tier (stream freq) │  TokenGuard contrastive router:
         │  router + budget prior   │  picks model m AND initial budget b0
         └────────────┬─────────────┘
                      │ (m, b0)
         ┌────────────▼─────────────┐
         │ MEDIUM tier (query freq) │  associative budget memory M:
         │  Titans surprise update  │  corrects b0 → b_t from realized surprise
         └────────────┬─────────────┘
                      │ b_t
         ┌────────────▼─────────────┐
         │  FAST tier (token freq)  │  per-step uncertainty u_t (entropy / logprob),
         │  surprise-driven halting │  momentum EMA; halt when converged ≤ τ;
         │  τ adapted by SW-UCB     │  τ tuned online by bandit → feeds MEDIUM
         └────────────┬─────────────┘
                      │
              answer  +  tokens_used   →  (accuracy, #tokens) logged
```

NL mapping (equation-level, for the methods chapter):
- **CMS frequency hierarchy** → token / query / stream tiers (NL Eq. 70-71)
- **Surprise = ∇ associative-memory loss + momentum** → step uncertainty EMA (Titans Eqs. 13-14)
- **Deep optimizer as associative memory** → slow-tier budget-prior update

---

## 3. Baselines to BEAT (named, on their own metrics)

| Baseline | arXiv | Metric / benchmark | Number to beat |
|----------|-------|--------------------|----------------|
| **REFRAIN** | 2510.10103 | Pass@1 & #tokens, Qwen3-8B, GSM8K/MATH-500/GPQA-Diamond/CSQA | ~40% token cut (GSM8K/MATH-500), ~55% (CSQA) at = accuracy |
| **MUR** | 2507.14958 | accuracy & compute, Qwen3-1.7B/4B/8B, MATH-500/GPQA-diamond | >50% backbone-token cut, +0.6-3.4% accuracy |
| **TALE** | 2412.18547 | accuracy / output-tokens / expense | ~67% token reduction, <5% accuracy loss |
| **CAR** | 2505.15154 | accuracy at matched tokens | +8.3% over TALE, +6.9% over CoD |
| **Chain-of-Draft** | 2502.18600 | accuracy & tokens | matches CoT at ~7.6-40% of tokens |
| **vanilla CoT** | — | accuracy & tokens | the quality ceiling to match |

**Our target:** match full-CoT accuracy at **5-10% relative fewer tokens than REFRAIN/MUR**, OR **+2-3% accuracy at matched budget**, shown with Pareto curves + seed-level significance.

---

## 4. Week-by-week plan

### WEEK 1 — Real-LLM harness + fast-tier surprise + baselines
**Goal:** reproduce the world's numbers before improving them (true comparison).
- [ ] `llm/` harness: vLLM (or transformers) generation for Qwen3-1.7B/4B/8B, with exact token counting and answer extraction.
- [ ] `reasoning/`: GSM8K, MATH-500, GPQA-Diamond loaders + scorers (exact-match / boxed-answer).
- [ ] Baselines in `baselines/`: vanilla-CoT, Chain-of-Draft, and reproduce **REFRAIN** & **MUR** halting; verify token-savings match their papers (±a few %).
- [ ] `budget/surprise.py`: per-step uncertainty (entropy + answer-logprob) with momentum EMA.
- **GATE W1:** vanilla-CoT and CoD numbers reproduced on GSM8K; REFRAIN/MUR halting reproduced within a few % of paper.

### WEEK 2 — The nested controller (the novelty)
**Goal:** build NTC and get a positive joint result.
- [ ] `budget/memory.py`: medium-tier associative budget memory (Titans surprise update).
- [ ] `budget/bandit.py`: SW-UCB threshold adapter (reuse `online/linucb.py`), reward → medium tier.
- [ ] `budget/controller.py`: wire SLOW router prior (b0) + MEDIUM memory (b_t) + FAST halting (τ).
- [ ] Run NTC on GSM8K/MATH-500/GPQA-Diamond; first Pareto curves vs baselines.
- **GATE W2:** NTC matches full-CoT accuracy at fewer tokens than REFRAIN/MUR on ≥1 benchmark (else pivot headline to joint cost-accuracy where the routing tier wins).

### WEEK 3 — Rigor, ablations, write-up
**Goal:** A+-level rigorous comparison.
- [ ] `eval/pareto.py`: three-way comparison — Pareto area, accuracy@matched-budget, tokens@matched-accuracy.
- [ ] 3-5 seeds; paired bootstrap (10k) + Wilcoxon + McNemar vs strongest baseline.
- [ ] Ablations: remove fast / medium / slow / momentum → show monotone degradation (proves each NL timescale earns its keep — the positive result the memory-routing work lacked).
- [ ] Cross-family check: Llama-3.1-8B-Instruct.
- [ ] Live demo + dissertation chapters + workshop-paper draft.
- **GATE W3:** every claim reproducible by one command; CIs + significance reported; demo runs.

---

## 5. Models & benchmarks (locked)
- **Open models:** Qwen3-1.7B / 4B / 8B (primary), Llama-3.1-8B-Instruct (cross-family).
- **Frontier routing target:** GPT-4o-mini or Claude Haiku (API) — strong tier for the router.
- **Benchmarks:** GSM8K, MATH-500, GPQA-Diamond (reasoning); RouterBench (routing tier); BBH optional.
- **Metrics:** accuracy@matched-tokens, tokens@matched-accuracy, Pareto-area, AIQ (routing), $ cost.

---

## 6. Honest pivot thresholds
- If NTC halting can't beat MUR/REFRAIN alone → headline the **joint cost-accuracy** metric (routing tier gives a win they structurally lack).
- If GPU/real-LLM runs too slow → fall back to **RouterBench++** (precompute each model at 3-5 reasoning-length settings; controller picks the cell; 2-D cost Pareto) — preserves the unified story at low compute.
- If a baseline won't reproduce → cite paper numbers AND your reproduction, and compare on the overlap.

---

## 7. Defensible novelty statement (for viva)
> "NTC is the first controller to unify model routing with within-chain,
> surprise-driven reasoning-budget control on nested timescales. The fast tier is
> a Titans-style surprise signal (Eqs. 13-14) halting generation; the tiers form a
> Continuum Memory System (NL Eq. 70-71) at token/query/stream frequencies. No
> prior router (RTR, SCOPE, R2-Router) uses an online surprise signal driving both
> halting and routing." 
