# No fixed halting signal generalizes

**Risk-controlled early exit for reasoning language models.**
Code, data and results for an MSc Artificial Intelligence dissertation, Brunel
University London (CS5500, 2025/26). Author: Abraheem Rashid. Supervisor:
Professor Yongmin Li.

Reasoning models keep generating after their answer has settled. Five halting
signals have been published to stop them early. This repository tests whether any
of them generalises, explains where they fail, and proposes a controller whose
tiers are ordered by what they require rather than by expected accuracy.

## Findings

| Finding | Evidence in this repository |
|---|---|
| **No fixed halting signal generalises.** The strongest, answer agreement, gains 8.3 accuracy points on GSM8K with Qwen3-8B and loses 23.5 on AIME-2024 with Qwen3-4B, under one tuning procedure. | `SLO_ATTAINMENT.md` |
| **The failure has a measurable cause.** Wrong answers repeat more readily as the answer space shrinks: stickiness rises from 0.130 on open-ended GSM8K to 0.765 on four-option GPQA-Diamond, and lost-correct risk rises with it (Spearman 0.71). | `PROP2_VALIDATION.md`, `PROP2_WITHIN.md` |
| **The fusion tier has the lowest worst-case regret**, 0.031 against 0.100 for the best fixed signal, and never loses more than 0.5 points in any of eleven settings, at a probing overhead of 1.4 to 7.1%. | `PRIMARY_STATS.md`, `DISSERTATION_NUMBERS.md` |
| **The risk certificate is conditional.** Within a domain it holds in 97% of cells; transferred to another domain the worst cell loses 50 points; required to hold in every domain it certifies only the null action. | `SHIFT_CERTIFICATE.md` |
| **What survives transfer is the fusion tier.** Asked for the cheapest rule that is safe on every other domain, the library picks a fusion setting in 58 of 60 cells, within a five-point budget in 95% of them at a 14.5% saving. | `FUSION_TRANSFER.md` |
| **DEER remains more accurate** under matched conditions, by 5.6 to 11.3 points, and the granularity explanation for the gap was tested and refuted. | `H2H_TABLE.md`, `CKPT_DENSITY.md` |

## What is here

```
src/tokenguard/llm/thinking.py        thinking-mode generation and answer probing
src/tokenguard/reasoning/datasets.py  benchmark loaders and the answer grader
src/tokenguard/baselines/halting.py   momentum and bandit halting baselines
scripts/ntc_*.py                      one analysis per result file (CPU only)
scripts/run_analyses.sh               regenerates every result file in one command
experiments/ntc/*.json                frozen probe streams: the data every analysis reads
experiments/ntc/*.md                  committed result files, one per analysis
external/DEER/                        the DEER authors' code, used for the matched comparison
tests/                                unit tests for the probe protocol and the graders
```

The corpus is 1,158 distinct questions, 10,178 reasoning traces and 108,266
graded trial answers across six benchmarks (GSM8K, MATH-500, GPQA-Diamond,
MMLU-Pro, AIME-2024, AIME-2025) and four models (Qwen3-1.7B/4B/8B and
DeepSeek-R1-Distill-Qwen-7B). Benchmarks are loaded from their public Hugging
Face releases; no dataset is redistributed here.

## Reproducing the results

```bash
pip install -r requirements.txt && pip install -e .
make test                 # unit tests
make tables               # every result file, from the frozen probe streams
make corpus               # the corpus inventory and the per-track tables
```

`make tables` runs `scripts/run_analyses.sh`, which holds the exact command for
every result file. Each analysis runs on a CPU and reads only committed files, so
the results regenerate without a GPU or network access. Each script writes one
markdown file and prints the same table to the terminal.

| Result | Script | Output |
|---|---|---|
| Corpus and experimental tracks | `ntc_corpus_tracks.py`, `ntc_data_inventory.py` | `CORPUS_TRACKS.md`, `DATA_INVENTORY.md` |
| Per-setting behaviour and attainment | `ntc_genseed_agg.py`, `ntc_slo_report.py` | `GENSEEDS_*.md`, `SLO_ATTAINMENT.md` |
| Primary comparison and minimax regret | `ntc_primary_stats.py` | `PRIMARY_STATS.md` |
| Paired tests, overhead, gate diagnostic | `ntc_dissertation_numbers.py` | `DISSERTATION_NUMBERS.md` |
| Error stickiness | `ntc_prop2_validate.py`, `ntc_prop2_within.py` | `PROP2_VALIDATION.md`, `PROP2_WITHIN.md` |
| Serving regimes | `ntc_cost_regimes.py` | `COST_REGIMES.md` |
| Matched comparison with DEER | `ntc_h2h_table.py` | `H2H_TABLE.md`, `DEER_OFFICIAL.md` |
| Checkpoint density sweep | `ntc_ckpt_density.py` | `CKPT_DENSITY.md` |
| Certificate under domain shift | `ntc_shift_certificate.py` | `SHIFT_CERTIFICATE.md` |
| Fusion tier under domain shift | `ntc_fusion_transfer.py` | `FUSION_TRANSFER.md` |
| Pricing the tail | `ntc_tail_price.py` | `TAIL_PRICE.md` |
| Joint tier (routing) | `ntc_joint_router.py` | `JOINT.md` |
| Grader reproducibility | `ntc_grader_check.py` | `GRADER_CHECK.md` |

Regenerating the probe streams themselves needs a GPU and vLLM:

```bash
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 \
  --limit 200 --max-think 8192 --batch 16 --seed 42
```

`run_queue*.sh` hold the exact SLURM jobs used for the runs in `experiments/ntc`.

## Method in one paragraph

Each trace is probed at paragraph-aligned checkpoints: at a 256-token mark or
before a reflective transition, at most ten per trace. At a checkpoint the cue
`</think>\n\nThe final answer is \boxed{` is appended and at most 24 greedy
tokens are read, which gives a trial answer and its confidence. Every halting
rule is a causal function of that one shared stream, so rules can be compared
without any of them enjoying an evidential advantage, and any rule can be scored
by exact replay. Costs always include the probes. The controller's fusion tier
halts when successive trial answers agree *and* smoothed confidence clears a
threshold; its selection tier chooses a rule per domain under a
Bonferroni-corrected lower confidence bound, with a null action that keeps the
admitted set non-empty.

## What this repository does not claim

The controller is not more accurate than DEER at DEER's own operating points; it
occupies a different regime, spending 1.4 to 7.1% of the chain on probes against
DEER's 66 to 79%. The certificate is valid only while calibration and deployment
items are exchangeable, which the shift experiment deliberately breaks. The
fusion tier is not shown to be better than answer agreement on average; its
advantage is in the worst case.

## Licence and citation

MIT, see `LICENSE`. The vendored `external/DEER` code is the property of its
authors and carries its own licence.

```bibtex
@mastersthesis{rashid2026halting,
  title  = {No Fixed Halting Signal Generalizes: Risk-Controlled Early Exit
            for Reasoning Language Models},
  author = {Rashid, Abraheem},
  school = {Brunel University London},
  year   = {2026}
}
```
