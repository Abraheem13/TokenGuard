# No Fixed Halting Signal Generalizes

**Risk-controlled early exit for reasoning language models.**

Code, data and results of the MSc Artificial Intelligence dissertation of Abraheem
Rashid, Brunel University London (CS5500, 2025/26), supervised by Professor
Yongmin Li. Every number, table and figure in the dissertation is regenerated
from the data in this repository by one command, on a CPU.

Reasoning models keep generating after their answer has settled, and several
halting signals have been proposed to stop them early. This work tests whether
any fixed signal generalises across benchmarks, model sizes and answer formats,
explains where the signals fail, and proposes the Nested Token-Budget Controller
(NTC), whose tiers are ordered by what they require rather than by expected
accuracy.

## Findings

| Finding | Evidence |
|---|---|
| **No fixed halting signal generalises.** Answer agreement, the strongest, gains 8.3 accuracy points on GSM8K with Qwen3-8B and loses 23.5 on AIME-2024 with Qwen3-4B under one tuning procedure. | `SLO_ATTAINMENT.md` |
| **The failure has a measurable cause.** Wrong answers repeat more readily as the answer space shrinks: error stickiness rises from 0.130 on open-ended GSM8K to 0.765 on four-option GPQA-Diamond, and the risk of losing a correct answer rises with it (Spearman 0.71). | `PROP2_VALIDATION.md`, `PROP2_WITHIN.md` |
| **The fusion tier has the lowest worst-case regret**, 0.031 against 0.100 for the best fixed signal, and never loses more than 0.5 points in any of eleven settings, at a probing overhead of 0.9 to 7.1%. | `PRIMARY_STATS.md`, `SLO_ATTAINMENT.md`, `DISSERTATION_NUMBERS.md` |
| **The risk certificate is conditional.** Within a domain it holds in 97% of cells; transferred to another domain the worst cell loses 50 points; required to hold in every domain it certifies only the null action. | `SHIFT_CERTIFICATE.md` |
| **What survives transfer is the fusion tier.** Asked for the cheapest rule that is safe on every other domain, the library picks a fusion setting in 58 of 60 cases, within a five-point budget in 95% of them at a 14.5% token saving. | `FUSION_TRANSFER.md` |
| **DEER remains more accurate** under matched conditions, by 5.6 to 11.1 points, and the granularity explanation for the gap was tested and refuted. | `H2H_TABLE.md`, `CKPT_DENSITY.md` |

All evidence files are in `experiments/ntc/`.

## Reproducing every result

```bash
git clone https://github.com/Abraheem13/TokenGuard.git
cd TokenGuard
python -m venv .venv && source .venv/bin/activate
make install
GPQA_PASSWORD=<password> make all
```

The eight GPQA-Diamond probe files are distributed in the password-protected
archive `experiments/ntc/gpqa_diamond.zip`, because the authors of GPQA ask that
its questions not be posted online in plain text. To obtain the password, email
abraheemrashid@outlook.com. Every other file is in plain text.

`make all` runs three steps:

1. `make results` extracts the GPQA-Diamond files (`scripts/unpack_gpqa.sh`) and
   regenerates every result file in `experiments/ntc/` from the probe streams
   (`scripts/run_analyses.sh`; about two minutes on a laptop CPU).
2. `make figures` writes the source of every data figure from those files
   (`scripts/make_figures.py`) and renders all 21 figures to `figures/pdf/` and
   `figures/png/` (`figures/build.sh`; needs pdfLaTeX with TikZ and pgfplots).
3. `make check` fails if any regenerated result file or figure source differs from
   the committed version.

No GPU, network access or model weights are needed. `make test` runs the unit
tests and `make lint` the static checks. Python 3.10 or later.

## Where each result comes from

Every script writes one result file in `experiments/ntc/`; `scripts/run_analyses.sh`
holds the exact command for each.

| Dissertation | Result file | Script |
|---|---|---|
| Tables 4.3, 4.4; Figure 4.1 | `CORPUS_TRACKS.md` | `ntc_corpus_tracks.py` |
| Table 5.1; Figures 5.1, 5.2, 5.5; Table 5.5 | `SLO_ATTAINMENT.md` (from `GENSEEDS_*.md`) | `ntc_slo_report.py`, `ntc_genseed_agg.py` |
| Tables 5.2, 5.3; Figure 5.3 | `PROP2_VALIDATION.md`, `PROP2_WITHIN.md` | `ntc_prop2_validate.py`, `ntc_prop2_within.py` |
| Table 5.4; Figure 5.4 | `PRIMARY_STATS.md`, `DISSERTATION_NUMBERS.md` | `ntc_primary_stats.py`, `ntc_dissertation_numbers.py` |
| Figures 5.6, 5.7 | `DISSERTATION_NUMBERS.md`, `DEER_OFFICIAL.md` | `ntc_dissertation_numbers.py` |
| Table 5.6 | `COST_REGIMES.md` | `ntc_cost_regimes.py` |
| Table 5.7; Figure 5.8 | `H2H_TABLE.md` | `ntc_h2h_table.py` |
| Figure 5.9 | `CKPT_DENSITY.md` | `ntc_ckpt_density.py` |
| Table 5.8; Figure 5.10 | `SHIFT_CERTIFICATE.md` | `ntc_shift_certificate.py` |
| Tables 5.9, 5.10; Figure 5.11 | `FUSION_TRANSFER.md` | `ntc_fusion_transfer.py` |
| Table 5.13; Figure 5.12 | `TAIL_PRICE.md` | `ntc_tail_price.py` |
| Section 4.5 (evaluation protocol) | `OPERATING_CURVES.md` | `ntc_operating_curves.py` |
| Section 5.10 (joint tier) | `JOINT.md` | `ntc_joint_router.py` |
| Section 3.2, Table 4.10 (grading) | `GRADER_CHECK.md` | `ntc_grader_check.py` |
| Data provenance | `DATA_INVENTORY.md`, `PROVENANCE.md` | `ntc_data_inventory.py`, `ntc_provenance.py` |

The diagrams (Figures 1.1, 2.1, 3.1 to 3.5 and 4.2) are in `figures/src/` with the
data figures.

## Repository layout

```
scripts/ntc_w1_stats.py        halting rules, overhead-inclusive replay, risk-controlled calibration
scripts/ntc_*.py               one analysis per result file (CPU)
scripts/run_analyses.sh        regenerates every result file
scripts/make_figures.py        writes every data figure from the result files
scripts/ntc_w1_thinking.py     generates a probe stream (GPU, vLLM)
src/tokenguard/llm/            thinking-mode generation and answer probing
src/tokenguard/reasoning/      benchmark loaders, answer extraction and the grader
experiments/ntc/*.json         probe streams, the grader's verdict cache
experiments/ntc/gpqa_diamond.zip  the GPQA-Diamond probe streams (password-protected)
experiments/ntc/*.md           result files
figures/                       figure sources, style, build script and rendered figures
external/DEER/                 the DEER authors' code (grader and matched comparison)
tests/                         unit tests
```

## Data

Each probe stream (`experiments/ntc/*.json`) records, for every question, the
model's thinking trace and, at up to ten paragraph-aligned checkpoints, a trial
answer forced by the cue `</think>\n\nThe final answer is \boxed{` with its
confidence, first-token entropy and token cost. Every halting rule is a causal
function of that one stream, so rules are compared on identical evidence and
scored by exact replay; costs always include the probes.

| Prefix | Track | Content |
|---|---|---|
| `w1_`, `w1sh_` | primary | six benchmarks, four models, several generation seeds; `w1sh_` is GPQA-Diamond with its options shuffled per item |
| `h2h2_` | head-to-head | MATH-500, GPQA-Diamond and AIME-2024 under DEER's conditions (greedy, 16k budget), with GPQA-Diamond in its source option order as in DEER's data |
| `dens` | density sweep | MATH-500 with two and four times as many checkpoints |

The corpus holds 1,158 distinct questions, 8,524 reasoning traces and 92,457
graded trial answers from GSM8K, MATH-500, GPQA-Diamond, MMLU-Pro, AIME-2024 and
AIME-2025, generated with Qwen3-1.7B, Qwen3-4B, Qwen3-8B and
DeepSeek-R1-Distill-Qwen-7B. `PROVENANCE.md` gives the settings of every probe
file and the command that regenerates it, which needs a GPU and
`pip install -e ".[generate]"`. The files contain the benchmark questions and the
models' outputs; the benchmarks remain under their own licences. The GPQA-Diamond
files are kept in the encrypted archive described above.

Answers are graded by the symbolic equivalence checker distributed with DEER.
Its 8,340 verdicts are cached in `experiments/ntc/grader_cache.json`, which every
analysis reads. `GRADER_CHECK.md` recomputes all of them: one long sequential run
reproduces 8,332 of them, and the other 8, which change only within a long
sequential run, match the stored verdict when recomputed in a fresh interpreter.

`DEER_OFFICIAL.md` records the results of running the DEER authors' code at its
default configuration; it is an input, not an output, of this repository.

## Names used in the code

| In the code and result files | In the dissertation |
|---|---|
| `NTC-v2`, `NTC-v2 (fusion)`, `NTC-Fuse` | NTC fusion tier |
| `NTC-full`, `NTC-full (selection)`, `NTC-Select` | NTC selection tier |
| `AGREE`, `Answer agreement` | answer agreement |
| `DEER`, `Confidence (DEER-λ)` | confidence threshold |
| `NTC-conf` | smoothed confidence |
| `EAT` | entropy (EAT) |
| `MUR-mom` | uncertainty momentum (MUR) |
| `REFRAIN-SWUCB` | bandit threshold (REFRAIN) |
| `NEVER-HALT`, `pi0` | the null action (full generation) |
| `vanilla` | full generation |
| calibration / warm-up split | calibration partition (40%) |

## Scope of the claims

The controller is not more accurate than DEER at DEER's own operating points; it
occupies a different regime, spending 0.9 to 7.1% of the chain on probes against
DEER's 66 to 79%. The certificate is valid only while calibration and deployment
items are exchangeable, which the shift experiment deliberately breaks. The
fusion tier is not shown to be better than answer agreement on average; its
advantage is in the worst case.

## Citation

```bibtex
@mastersthesis{rashid2026halting,
  title  = {No Fixed Halting Signal Generalizes: Risk-Controlled Early Exit
            for Reasoning Language Models},
  author = {Rashid, Abraheem},
  school = {Brunel University London},
  year   = {2026}
}
```

## Licence

MIT, see `LICENSE`. The code in `external/DEER` belongs to its authors and is
distributed under its own licence.
