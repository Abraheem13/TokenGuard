#!/usr/bin/env bash
#
# scaffold_ntc.sh — scaffold the NTC (Nested Token-Budget Controller) upgrade.
# Run ONCE from the tokenguard repo root:  bash scaffold_ntc.sh
#
# It (1) ADDS new packages budget/ llm/ reasoning/ baselines/ + scripts + tests,
#       (2) ARCHIVES the dead memory-routing code into archive/ (kept for the
#           honest negative-result ablation, NOT deleted),
#       (3) DELETES genuine junk (one-off setup scripts, caches, stray files).
# Every create is guarded (never overwrites your work). Safe to re-run.

set -euo pipefail
if [ ! -d "src/tokenguard" ]; then
  echo "ERROR: run from the tokenguard repo root (src/tokenguard not found)."; exit 1
fi

write() {
  local path="$1"
  if [ -e "$path" ]; then echo "  skip (exists): $path"
  else mkdir -p "$(dirname "$path")"; cat > "$path"; echo "  created: $path"; fi
}
move() {  # move $1 -> $2 if source exists and dest doesn't
  if [ -e "$1" ] && [ ! -e "$2" ]; then mkdir -p "$(dirname "$2")"; git mv "$1" "$2" 2>/dev/null || mv "$1" "$2"; echo "  archived: $1 -> $2"; fi
}
scrub() {  # delete junk if present
  if [ -e "$1" ]; then rm -rf "$1"; echo "  deleted: $1"; fi
}

echo "== NTC scaffold =="

# ---------- 1. DELETE junk ----------
echo "-- deleting junk --"
scrub fix_schema.sh
scrub install_day2.sh
scrub scripts/day5_online.py.save
scrub experiments/results/telemetry.db
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name "*.save" -delete 2>/dev/null || true

# ---------- 2. ARCHIVE the memory-routing dead end (keep for ablation) ----------
echo "-- archiving memory-routing experiments --"
move src/tokenguard/online/nested_router.py archive/memory_routing/nested_router.py
move src/tokenguard/memory                   archive/memory_routing/memory
move src/tokenguard/streams                  archive/memory_routing/streams
move src/tokenguard/routers/nestor_router.py archive/memory_routing/nestor_router.py
move scripts/nestor_week2_midmemory.py       archive/memory_routing/nestor_week2_midmemory.py
move scripts/nestor_week3_slow.py            archive/memory_routing/nestor_week3_slow.py
move scripts/nestor_week4_ablations.py       archive/memory_routing/nestor_week4_ablations.py
move tests/test_neural_memory.py             archive/memory_routing/tests/test_neural_memory.py
move tests/test_cms.py                       archive/memory_routing/tests/test_cms.py
move tests/test_surprise.py                  archive/memory_routing/tests/test_surprise.py
move tests/test_nestor.py                    archive/memory_routing/tests/test_nestor.py
move tests/test_streams.py                   archive/memory_routing/tests/test_streams.py
move tests/test_day5.py                      archive/memory_routing/tests/test_day5.py
write archive/memory_routing/README.md <<'PY'
# Archived: memory-routing experiments (honest negative result)

These modules implement the surprise-gated nested *memory* router. On real
RouterBench they did NOT beat a single-timescale contextual bandit across five
regimes (shuffled, recurrence, topic-drift, model-drift, price-shift), because a
cost-aware bandit already adapts to context and price. This is a reportable
negative result and is retained for the dissertation's ablation chapter. The
NTC project moves the multi-timescale idea onto reasoning-budget control, where
the controlled variable genuinely drifts.
PY

# ---------- 3. ADD: budget/ (the NTC core) ----------
echo "-- adding NTC packages --"
write src/tokenguard/budget/__init__.py <<'PY'
"""NTC — Nested Token-Budget Controller.

Three timescales (Nested Learning Continuum Memory System):
  FAST   (token freq)  surprise-driven halting           budget/surprise.py
  MEDIUM (query freq)  associative budget memory          budget/memory.py
  SLOW   (stream freq) router + budget prior (TokenGuard) budget/controller.py
The SW-UCB threshold adapter lives in budget/bandit.py.
"""
PY

write src/tokenguard/budget/surprise.py <<'PY'
"""FAST tier — per-step surprise / uncertainty signal driving halting.

surprise_t = predictive uncertainty of reasoning step t (entropy or negative
answer-log-likelihood), smoothed by a momentum EMA (Titans Eqs. 13-14 analogue:
S_t = eta * S_{t-1} + (1-eta) * u_t). Halt when the smoothed signal has
converged below threshold tau for `patience` steps.

TODO(week1): compute u_t from the model's step logprobs/entropy in llm/.
"""
from __future__ import annotations
import numpy as np


class SurpriseHalter:
    def __init__(self, tau: float = 0.15, eta: float = 0.7, patience: int = 2,
                 min_steps: int = 1, max_steps: int = 64):
        self.tau, self.eta, self.patience = tau, eta, patience
        self.min_steps, self.max_steps = min_steps, max_steps
        self.reset()

    def reset(self) -> None:
        self._ema = None
        self._below = 0
        self._t = 0

    def observe(self, uncertainty: float) -> bool:
        """Feed step uncertainty; return True if generation should HALT now."""
        self._t += 1
        self._ema = uncertainty if self._ema is None else (
            self.eta * self._ema + (1 - self.eta) * uncertainty)
        self._below = self._below + 1 if self._ema <= self.tau else 0
        if self._t < self.min_steps:
            return False
        if self._t >= self.max_steps:
            return True
        return self._below >= self.patience

    @property
    def smoothed(self) -> float:
        return float(self._ema if self._ema is not None else 0.0)
PY

write src/tokenguard/budget/memory.py <<'PY'
"""MEDIUM tier — associative budget memory (Titans surprise update).

Maps a query key (router embedding) -> a correction on the initial budget prior
b0, learned online from realized (key, optimal-budget) pairs. Titans update with
momentum + data-dependent forgetting. Corrects b0 -> b_t.

TODO(week2): integrate with the controller; tune lr/forget on a held-out split.
"""
from __future__ import annotations
import numpy as np


class BudgetMemory:
    def __init__(self, key_dim: int, lr: float = 0.1, momentum: float = 0.9,
                 forget: float = 0.01, seed: int = 42):
        self.w = np.zeros(key_dim)          # linear key -> budget correction
        self.S = np.zeros(key_dim)
        self.lr, self.momentum, self.forget = lr, momentum, forget

    def predict(self, key: np.ndarray) -> float:
        return float(self.w @ key)

    def update(self, key: np.ndarray, target_correction: float) -> float:
        pred = float(self.w @ key)
        surprise = target_correction - pred
        grad = -surprise * key
        self.S = self.momentum * self.S - self.lr * grad
        self.w = (1 - self.forget) * self.w + self.S
        return abs(surprise)
PY

write src/tokenguard/budget/bandit.py <<'PY'
"""Sliding-window UCB adapter for the halting threshold tau (REFRAIN-style).

Picks tau from a small grid to maximise reward = accuracy - mu * tokens, over a
sliding window so it tracks drift. Reward feeds the MEDIUM tier, closing the
nested loop.

TODO(week2): wire reward from realized (correct?, tokens) per query.
"""
from __future__ import annotations
import numpy as np


class SWUCBThreshold:
    def __init__(self, taus=(0.05, 0.1, 0.15, 0.2, 0.3), window: int = 200,
                 c: float = 1.0, seed: int = 42):
        self.taus = list(taus)
        self.window = window
        self.c = c
        self.hist = {i: [] for i in range(len(self.taus))}
        self._t = 0

    def select(self) -> tuple[int, float]:
        self._t += 1
        best, best_score = 0, -1e9
        for i in range(len(self.taus)):
            h = self.hist[i][-self.window:]
            if not h:
                return i, self.taus[i]            # try each arm once
            mean = float(np.mean(h))
            bonus = self.c * np.sqrt(np.log(self._t + 1) / len(h))
            score = mean + bonus
            if score > best_score:
                best, best_score = i, score
        return best, self.taus[best]

    def update(self, arm: int, reward: float) -> None:
        self.hist[arm].append(reward)
PY

write src/tokenguard/budget/controller.py <<'PY'
"""NTC controller — wires SLOW (router prior) + MEDIUM (memory) + FAST (halting).

decide(query_key) -> (model, budget_prior, halter, tau)
observe(query_key, correct, tokens) -> updates bandit + memory.

TODO(week2): integrate the contrastive router for the SLOW prior; connect to the
llm/ harness so halting actually stops generation.
"""
from __future__ import annotations
import numpy as np

from tokenguard.budget.surprise import SurpriseHalter
from tokenguard.budget.memory import BudgetMemory
from tokenguard.budget.bandit import SWUCBThreshold


class NestedBudgetController:
    def __init__(self, key_dim: int, base_budget: int = 32, mu: float = 0.01,
                 seed: int = 42):
        self.memory = BudgetMemory(key_dim, seed=seed)
        self.bandit = SWUCBThreshold(seed=seed)
        self.base_budget = base_budget
        self.mu = mu

    def decide(self, key: np.ndarray):
        b0 = self.base_budget
        b_t = max(1, int(round(b0 + self.memory.predict(key))))
        arm, tau = self.bandit.select()
        halter = SurpriseHalter(tau=tau, max_steps=b_t)
        return {"budget": b_t, "tau": tau, "arm": arm, "halter": halter}

    def observe(self, key: np.ndarray, arm: int, correct: bool, tokens: int,
                target_budget: int) -> None:
        reward = (1.0 if correct else 0.0) - self.mu * tokens
        self.bandit.update(arm, reward)
        self.memory.update(key, target_budget - self.base_budget)
PY

# ---------- ADD: llm/ (real-LLM harness) ----------
write src/tokenguard/llm/__init__.py <<'PY'
"""Real-LLM generation harness for token-budget experiments."""
PY

write src/tokenguard/llm/generate.py <<'PY'
"""Real-LLM generation with token counting + step uncertainty.

Wraps vLLM (preferred) or transformers to: generate a reasoning chain, expose
per-step logprobs/entropy (the FAST-tier surprise signal), count tokens, and
support early halting. Keep models small (Qwen3-1.7B/4B/8B) for GPU budget.

TODO(week1): implement generate(prompt, max_tokens, halter=None) -> dict with
keys {text, n_tokens, step_uncertainty[list], answer}.
"""
from __future__ import annotations


class LLMRunner:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", device: str = "cuda",
                 backend: str = "vllm"):
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self._llm = None

    def _lazy(self):
        raise NotImplementedError("implement vLLM/transformers load in week 1")

    def generate(self, prompt: str, max_tokens: int = 512, halter=None) -> dict:
        raise NotImplementedError("implement generation + token counting in week 1")
PY

# ---------- ADD: reasoning/ (benchmarks) ----------
write src/tokenguard/reasoning/__init__.py <<'PY'
"""Reasoning benchmarks: GSM8K, MATH-500, GPQA-Diamond loaders + scorers."""
PY

write src/tokenguard/reasoning/datasets.py <<'PY'
"""Load reasoning benchmarks and score answers (exact-match / boxed).

TODO(week1): load via `datasets`; implement extract_answer + is_correct per set.
"""
from __future__ import annotations


def load_benchmark(name: str, split: str = "test", limit: int | None = None):
    """name in {gsm8k, math500, gpqa_diamond}. Returns list of {question, answer}."""
    raise NotImplementedError("implement dataset loaders in week 1")


def extract_answer(text: str, benchmark: str) -> str:
    raise NotImplementedError("implement answer extraction in week 1")


def is_correct(pred: str, gold: str, benchmark: str) -> bool:
    raise NotImplementedError("implement scoring in week 1")
PY

# ---------- ADD: baselines/ ----------
write src/tokenguard/baselines/__init__.py <<'PY'
"""Reimplemented token-saving baselines for true head-to-head comparison."""
PY

write src/tokenguard/baselines/prompts.py <<'PY'
"""Prompt templates for baselines: vanilla CoT, Chain-of-Draft, budget-forced.

TODO(week1): fill in exact prompts from the respective papers for fair compare.
"""
COT = "Let's think step by step.\n"
CHAIN_OF_DRAFT = ("Think step by step, but keep each thinking step to at most "
                  "five words. Return the final answer after '####'.\n")
BUDGET_FORCED = "Solve concisely in at most {budget} tokens.\n"
PY

write src/tokenguard/baselines/halting.py <<'PY'
"""REFRAIN / MUR-style uncertainty halting baselines (for reproduction).

TODO(week1): implement DEER/HALT-style entropy halt (REFRAIN) and momentum-
uncertainty halt (MUR); verify token savings match the papers within a few %.
"""
from __future__ import annotations


def refrain_halt(step_uncertainties, tau=0.15):
    raise NotImplementedError("implement REFRAIN-style halting in week 1")


def mur_halt(step_uncertainties, momentum=0.9, tau=0.15):
    raise NotImplementedError("implement MUR momentum halting in week 1")
PY

# ---------- ADD: eval/pareto.py ----------
write src/tokenguard/eval/pareto.py <<'PY'
"""Quality-token Pareto + matched-budget / matched-accuracy + significance.

The three honest ways to show 'same quality, fewer tokens':
  1. pareto_area(points)                     area under accuracy-vs-tokens curve
  2. accuracy_at_budget(points, budget)      interpolate accuracy at common tokens
  3. tokens_at_accuracy(points, target_acc)  tokens to reach an accuracy
Plus paired bootstrap CIs for deltas.

TODO(week3): finalise interpolation + bootstrap/Wilcoxon helpers.
"""
from __future__ import annotations
import numpy as np


def pareto_area(tokens, accuracy) -> float:
    order = np.argsort(tokens)
    t, a = np.asarray(tokens)[order], np.asarray(accuracy)[order]
    return float(np.trapezoid(a, t) / (t.max() - t.min() + 1e-9))


def accuracy_at_budget(tokens, accuracy, budget) -> float:
    return float(np.interp(budget, np.sort(tokens),
                           np.asarray(accuracy)[np.argsort(tokens)]))


def tokens_at_accuracy(tokens, accuracy, target_acc) -> float:
    order = np.argsort(accuracy)
    return float(np.interp(target_acc, np.asarray(accuracy)[order],
                           np.asarray(tokens)[order]))


def bootstrap_delta(a, b, n=10000, seed=0):
    """Paired bootstrap 95% CI for mean(a)-mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    deltas = a[idx].mean(1) - b[idx].mean(1)
    return float(deltas.mean()), (float(np.percentile(deltas, 2.5)),
                                  float(np.percentile(deltas, 97.5)))
PY

# ---------- ADD: week scripts ----------
write scripts/ntc_week1_harness.py <<'PY'
#!/usr/bin/env python
"""Week 1 — real-LLM harness + reasoning benchmarks + reproduce baselines.
See docs/ROADMAP_NTC.md. GATE: reproduce vanilla-CoT/CoD on GSM8K and
REFRAIN/MUR halting within a few % of their papers."""
raise SystemExit("TODO(week1): implement per docs/ROADMAP_NTC.md")
PY
write scripts/ntc_week2_controller.py <<'PY'
#!/usr/bin/env python
"""Week 2 — nested budget controller (fast+medium+slow); first Pareto curves."""
raise SystemExit("TODO(week2): implement per docs/ROADMAP_NTC.md")
PY
write scripts/ntc_week3_eval.py <<'PY'
#!/usr/bin/env python
"""Week 3 — Pareto/matched-budget/matched-accuracy + seeds + ablations."""
raise SystemExit("TODO(week3): implement per docs/ROADMAP_NTC.md")
PY

# ---------- ADD: tests ----------
write tests/test_budget_surprise.py <<'PY'
from tokenguard.budget.surprise import SurpriseHalter


def test_halts_after_convergence():
    h = SurpriseHalter(tau=0.1, eta=0.5, patience=2, min_steps=1, max_steps=20)
    seq = [0.5, 0.2, 0.05, 0.02, 0.02, 0.01, 0.01]
    halted = [h.observe(u) for u in seq]
    assert any(halted) and halted[-1] is True


def test_does_not_halt_while_uncertain():
    h = SurpriseHalter(tau=0.05, eta=0.5, patience=2, min_steps=1, max_steps=20)
    halted = [h.observe(0.8) for _ in range(5)]
    assert not any(halted)


def test_respects_max_steps():
    h = SurpriseHalter(tau=0.0, max_steps=3)   # tau=0 never satisfied
    outs = [h.observe(1.0) for _ in range(3)]
    assert outs[-1] is True
PY

write tests/test_budget_memory.py <<'PY'
import numpy as np
from tokenguard.budget.memory import BudgetMemory


def test_memory_learns_correction():
    m = BudgetMemory(key_dim=4, lr=0.2, forget=0.0)
    k = np.ones(4) / 2
    for _ in range(50):
        m.update(k, target_correction=5.0)
    assert m.predict(k) > 1.0
PY

write tests/test_budget_bandit.py <<'PY'
from tokenguard.budget.bandit import SWUCBThreshold


def test_bandit_prefers_high_reward_arm():
    b = SWUCBThreshold(taus=(0.1, 0.2), window=50, c=0.1)
    for _ in range(60):
        arm, tau = b.select()
        b.update(arm, reward=1.0 if arm == 0 else 0.0)
    picks = [b.select()[0] for _ in range(20)]
    assert picks.count(0) > picks.count(1)
PY

write tests/test_pareto.py <<'PY'
import numpy as np
from tokenguard.eval.pareto import pareto_area, tokens_at_accuracy, bootstrap_delta


def test_pareto_area_monotone():
    a = pareto_area([10, 20, 30], [0.5, 0.7, 0.9])
    assert 0.5 <= a <= 0.9


def test_tokens_at_accuracy():
    assert tokens_at_accuracy([10, 20, 30], [0.5, 0.7, 0.9], 0.7) == 20


def test_bootstrap_delta_sign():
    d, ci = bootstrap_delta([1, 1, 1, 1], [0, 0, 0, 0])
    assert d > 0 and ci[0] <= d <= ci[1]
PY

# ---------- ADD: experiments dir + docs ----------
mkdir -p experiments/ntc && write experiments/ntc/.gitkeep <<'PY'
PY

echo ""
echo "NTC scaffold complete."
echo "Next:"
echo "  1) cp ROADMAP_NTC.md docs/ROADMAP_NTC.md"
echo "  2) python -m pytest tests/ -q"
echo "  3) git add -A && git commit -m 'scaffold NTC: budget controller, llm harness, baselines'"
