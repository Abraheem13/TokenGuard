# GPU runbook — what is left to measure, and exactly how

Everything in the paper is reproducible on a laptop from the frozen probe files
in `experiments/ntc/`.  Two questions cannot be: they need new generations.
Both are queued and analysed by scripts already in the repository, so the loop
is submit, wait, run one analysis command, and the tables regenerate.

## Before you start

```bash
cd ~/TokenGuard && source .venv/bin/activate
python -c "import vllm, torch; print(vllm.__version__, torch.cuda.device_count())"
python scripts/ntc_ckpt_density.py     # should report 4 missing files
```

---

## Experiment A (required) — does checkpoint density explain the gap to DEER?

**The question.** Table 12 of the paper reports that DEER, run from its authors'
code, is 4–10 accuracy points ahead of our controller on MATH-500 under matched
conditions, and cheaper.  Section 6.6 attributes this to granularity: at 1x
density we buy 9.3 probes per item and pay 3.6% overhead, while DEER interrupts
at every reasoning transition and pays 73–79%.  That is a hypothesis, and it is
testable: probe more often and see whether the accuracy gap closes and at what
cost in overhead.

**Why it matters.** This is the single result that would change the paper's
standing.  If accuracy rises towards DEER's as density rises, the two systems
occupy one density–overhead trade-off curve and we can say so — which converts
"DEER is more accurate than us" into "DEER buys accuracy with 20x our
overhead, and here is the exchange rate".  If accuracy does not rise, the gap is
in the signal, and the paper should say that plainly instead.  Either outcome is
publishable; only silence is not.

**Run it.**

```bash
bash run_queue_density.sh --dry     # inspect the four sbatch files
bash run_queue_density.sh           # submit
squeue -u $USER
```

Four jobs: Qwen3-4B and Qwen3-8B, each at 2x (a probe every 128 thinking
tokens, at most 20) and 4x (every 64, at most 40) density.  MATH-500, n=500,
greedy, 16k budget, standard boxed prompt — identical to `run_queue_h2h2.sh`
in every respect except density, which is the point.  Budget 9–10 h per job on
one L40S; the 1x baseline already exists as `h2h2_math500_Qwen3-{4B,8B}.json`.

**Analyse it (no GPU).**

```bash
python scripts/ntc_ckpt_density.py          # writes experiments/ntc/CKPT_DENSITY.md
```

The table reports probes per item, vanilla accuracy and cost, controller
overhead, fixed-default agreement, and the calibrated NTC-Select operating
point at each density, with DEER's official numbers on the same convention in
the last row.

---

## Experiment B (optional) — a genuinely different backbone family

**The question.** The paper says "two model families", but
DeepSeek-R1-Distill-Qwen-7B is a Qwen backbone, so the claim is thinner than it
sounds.  A reviewer will notice.  `DeepSeek-R1-Distill-Llama-8B` is a reasoning
model on a Llama backbone using the same `<think>` protocol, so it drops into
the harness unchanged.

**Run it.**

```bash
CROSS_FAMILY=1 bash run_queue_density.sh    # adds 2 jobs (MATH-500, GPQA-D)
```

**Analyse it (no GPU).**

```bash
python scripts/ntc_w1_stats.py --probes experiments/ntc/w1_math500_DeepSeek-R1-Distill-Llama-8B.json
python scripts/ntc_genseed_agg.py --probes experiments/ntc/w1_math500_DeepSeek-R1-Distill-Llama-8B.json \
    --tag math500_R1-Llama-8B
python scripts/ntc_prop2_validate.py --probes experiments/ntc/w1_*.json --probes experiments/ntc/w1sh_*.json
```

If stickiness on the Llama backbone lands where Proposition 2 predicts from its
answer space, the mechanism is a property of reasoning models rather than of
Qwen, which is a stronger claim than the paper currently makes.

---

## After either experiment

```bash
python scripts/ntc_slo_report.py            # refresh SLO_ATTAINMENT.md
python scripts/ntc_final_freeze.py          # refresh RESULTS_FINAL/MASTER.md
cd paper && make figures && make            # rebuild the manuscript
```

Then update Section 6.6 (Experiment A) or Section 5.1 and Table 4 (Experiment
B) with the new numbers.  Keep the paper at or under 25 pages: TIST rejects
longer submissions automatically.
