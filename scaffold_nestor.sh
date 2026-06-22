#!/usr/bin/env bash
set -euo pipefail
if [ ! -d "src/tokenguard" ]; then
  echo "ERROR: run from tokenguard repo root (src/tokenguard not found)."; exit 1
fi
write() {
  local path="$1"
  if [ -e "$path" ]; then echo "  skip (exists): $path"
  else mkdir -p "$(dirname "$path")"; cat > "$path"; echo "  created: $path"; fi
}
echo "Scaffolding NESTOR..."

write src/tokenguard/memory/__init__.py <<'PY'
"""NESTOR Continuum Memory System: L1 fast + L2 mid + L3 slow."""
PY

write src/tokenguard/memory/neural_memory.py <<'PY'
"""L2 Titans-style neural memory, test-time surprise updates (Eqs. 13-14)."""
from __future__ import annotations
import numpy as np


class NeuralRoutingMemory:
    def __init__(self, key_dim, n_models, lr=0.05, momentum=0.9, forget=0.01, seed=42):
        self.key_dim, self.n_models = key_dim, n_models
        self.lr, self.momentum, self.forget = lr, momentum, forget
        self.rng = np.random.default_rng(seed)
        self.M = np.zeros((n_models, key_dim))
        self.S = np.zeros_like(self.M)

    def predict(self, key):
        return self.M @ key

    def update(self, key, model, reward):
        pred = float(self.M[model] @ key)
        surprise = reward - pred
        grad = -surprise * key
        self.S[model] = self.momentum * self.S[model] - self.lr * grad
        self.M[model] = (1.0 - self.forget) * self.M[model] + self.S[model]
        return abs(surprise)
PY

write src/tokenguard/memory/surprise.py <<'PY'
"""Surprise-gated write controller."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WriteDecision:
    to_fast: bool = True
    to_mid: bool = False
    to_slow: bool = False
    surprise: float = 0.0


class SurpriseGate:
    def __init__(self, mid_threshold=0.3, slow_window=2000):
        self.mid_threshold = mid_threshold
        self.slow_window = slow_window

    def decide(self, surprise, recurrent):
        return WriteDecision(to_fast=True,
                             to_mid=(surprise > self.mid_threshold and recurrent),
                             to_slow=False, surprise=surprise)
PY

write src/tokenguard/memory/consolidation.py <<'PY'
"""L3 slow consolidation (CMS Eq. 71)."""
from __future__ import annotations


class SlowConsolidator:
    def __init__(self, every=4000, min_cal_err=0.30):
        self.every, self.min_cal_err, self._step = every, min_cal_err, 0

    def maybe_consolidate(self, base, replay):
        self._step += 1
        return False
PY

write src/tokenguard/memory/cms.py <<'PY'
"""Continuum Memory System orchestrator (L1+L2+L3)."""
from __future__ import annotations
import numpy as np
from tokenguard.memory.neural_memory import NeuralRoutingMemory
from tokenguard.memory.surprise import SurpriseGate
from tokenguard.memory.consolidation import SlowConsolidator


class ContinuumMemoryRouter:
    def __init__(self, key_dim, n_models, c2=1, c3=4000, seed=42):
        self.mid = NeuralRoutingMemory(key_dim, n_models, seed=seed)
        self.gate = SurpriseGate()
        self.slow = SlowConsolidator(every=c3)
        self.c2, self._step = c2, 0

    def predict(self, key, fast_pred):
        return 0.5 * fast_pred + 0.5 * self.mid.predict(key)

    def observe(self, key, model, reward, recurrent=False):
        self._step += 1
        pred = float(self.mid.predict(key)[model])
        surprise = abs(reward - pred)
        dec = self.gate.decide(surprise, recurrent)
        if dec.to_mid and self._step % self.c2 == 0:
            self.mid.update(key, model, reward)
        return surprise
PY

write src/tokenguard/encoders/__init__.py <<'PY'
"""Modern encoder wrappers (BERT excluded)."""
PY

write src/tokenguard/encoders/qwen3_embedding.py <<'PY'
"""Qwen3-Embedding-0.6B wrapper (modern BERT replacement)."""
from __future__ import annotations
import numpy as np


class Qwen3Embedding:
    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self, device="mps", cache_path=None):
        self.device, self.cache_path, self._model = device, cache_path, None

    def encode(self, texts):
        raise NotImplementedError("implement Qwen3-Embedding encode in week 1")
PY

write src/tokenguard/streams/__init__.py <<'PY'
"""Non-stationary stream generators for NESTOR experiments."""
PY

write src/tokenguard/streams/recurrence.py <<'PY'
"""Recurrence / personalisation stream (L2 should win here)."""
from __future__ import annotations


def recurrence_stream(bench, n_clusters=20, revisit_rate=0.5, seed=42):
    raise NotImplementedError("implement in week 1")
PY

write src/tokenguard/streams/drift.py <<'PY'
"""Topic-drift stream."""
from __future__ import annotations


def drift_stream(bench, n_phases=5, seed=42):
    raise NotImplementedError("implement in week 1")
PY

write src/tokenguard/streams/price_shift.py <<'PY'
"""Price-shift stream (L3 should win here)."""
from __future__ import annotations


def price_shift_stream(bench, shift_at=0.5, factor=2.0, models=None, seed=42):
    raise NotImplementedError("implement in week 1")
PY

write src/tokenguard/streams/new_model.py <<'PY'
"""New-model-arrival stream (mechanism designed to win)."""
from __future__ import annotations


def new_model_stream(bench, arrival_model, arrival_fraction=0.5, seed=42):
    raise NotImplementedError("implement in week 3")
PY

write src/tokenguard/routers/nestor_router.py <<'PY'
"""NESTOR top-level router wiring CMS to the decision rule."""
from __future__ import annotations
import numpy as np
from tokenguard.memory.cms import ContinuumMemoryRouter


class NestorRouter:
    def __init__(self, base, key_dim, lambda_cost=0.5, c2=1, c3=4000, seed=42):
        self.base, self.lambda_cost = base, lambda_cost
        self.cms = ContinuumMemoryRouter(key_dim, len(base.models_), c2=c2, c3=c3, seed=seed)

    def run_stream(self, stream, **kw):
        raise NotImplementedError("wire CMS into run_stream in week 3")
PY

write scripts/nestor_week1_baselines.py <<'PY'
#!/usr/bin/env python
"""Week 1 - encoder swap + single-timescale LinUCB baseline + NL mapping."""
raise SystemExit("TODO(week1): see docs/ROADMAP.md")
PY

write scripts/nestor_week2_midmemory.py <<'PY'
#!/usr/bin/env python
"""Week 2 - L2 neural memory + surprise gate + recurrence experiment."""
raise SystemExit("TODO(week2): see docs/ROADMAP.md")
PY

write scripts/nestor_week3_slow.py <<'PY'
#!/usr/bin/env python
"""Week 3 - L3 slow + price-shift + new-model + live demo."""
raise SystemExit("TODO(week3): see docs/ROADMAP.md")
PY

write scripts/nestor_week4_ablations.py <<'PY'
#!/usr/bin/env python
"""Week 4 - ablation grid, 5 seeds, bootstrap CIs, final figures."""
raise SystemExit("TODO(week4): see docs/ROADMAP.md")
PY

write scripts/day7_dashboard.py <<'PY'
#!/usr/bin/env python
"""Day 7 - Streamlit dashboard. Run: streamlit run scripts/day7_dashboard.py"""
raise SystemExit("TODO(day7): implement Streamlit dashboard")
PY

write tests/test_neural_memory.py <<'PY'
import numpy as np
from tokenguard.memory.neural_memory import NeuralRoutingMemory


def test_memory_predict_shape():
    m = NeuralRoutingMemory(key_dim=8, n_models=3)
    assert m.predict(np.ones(8)).shape == (3,)


def test_memory_update_reduces_error():
    m = NeuralRoutingMemory(key_dim=8, n_models=3, lr=0.1, forget=0.0)
    key = np.ones(8) / np.sqrt(8)
    for _ in range(50):
        m.update(key, model=1, reward=1.0)
    assert m.predict(key)[1] > m.predict(key)[0]
PY

write tests/test_cms.py <<'PY'
import numpy as np
from tokenguard.memory.cms import ContinuumMemoryRouter


def test_cms_observe_returns_surprise():
    cms = ContinuumMemoryRouter(key_dim=8, n_models=3)
    s = cms.observe(np.ones(8) / np.sqrt(8), model=0, reward=1.0, recurrent=True)
    assert s >= 0.0


def test_cms_predict_blend_shape():
    cms = ContinuumMemoryRouter(key_dim=8, n_models=3)
    out = cms.predict(np.ones(8), fast_pred=np.zeros(3))
    assert out.shape == (3,)
PY

write tests/test_surprise.py <<'PY'
from tokenguard.memory.surprise import SurpriseGate


def test_gate_writes_mid_on_high_surprise_recurrent():
    g = SurpriseGate(mid_threshold=0.3)
    assert g.decide(surprise=0.9, recurrent=True).to_mid is True


def test_gate_skips_mid_on_low_surprise():
    g = SurpriseGate(mid_threshold=0.3)
    assert g.decide(surprise=0.1, recurrent=True).to_mid is False
PY

write tests/test_streams.py <<'PY'
import pytest


@pytest.mark.xfail(reason="streams implemented later", strict=False)
def test_recurrence_stream_importable():
    from tokenguard.streams.recurrence import recurrence_stream
    recurrence_stream(bench=None)
PY

mkdir -p experiments/nestor && touch experiments/nestor/.gitkeep
echo ""
echo "NESTOR scaffold complete."
