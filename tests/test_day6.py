"""Day 6 tests — telemetry store, router service, and the OpenAI-compatible proxy."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from tokenguard.proxy.router_service import RouterService
from tokenguard.proxy.telemetry import Record, TelemetryStore
from tokenguard.proxy.app import create_app


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class _FakeBase:
    class _Enc:
        def encode(self, t):
            v = np.ones((len(t), 4), np.float32) * max(len(t[0]), 1)
            return v / np.linalg.norm(v)

    encoder = _Enc()
    P_ = np.eye(4, dtype=np.float32)
    E_ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)
    a_ = np.full(3, 4.0, np.float32)
    b_ = np.zeros(3, np.float32)
    models_ = ["cheap", "mid", "strong"]


@pytest.fixture
def tmp_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


@pytest.fixture
def service():
    return RouterService(_FakeBase(),
                         {"cheap": 0.0001, "mid": 0.001, "strong": 0.003},
                         lambda_cost=0.5)


# --------------------------------------------------------------------------- #
# telemetry
# --------------------------------------------------------------------------- #
def test_telemetry_log_and_count(tmp_db):
    s = TelemetryStore(tmp_db)
    assert s.count() == 0
    s.log(Record(request_id="r1", query_chars=10, chosen_model="cheap",
                 cost_usd=0.0001, latency_ms=50.0))
    assert s.count() == 1


def test_telemetry_late_reward(tmp_db):
    s = TelemetryStore(tmp_db)
    s.log(Record(request_id="r1", query_chars=10, chosen_model="cheap",
                 cost_usd=0.0001, latency_ms=50.0))
    s.set_reward("r1", 0.9)
    assert s.recent(1)[0]["reward"] == pytest.approx(0.9)


def test_telemetry_summary(tmp_db):
    s = TelemetryStore(tmp_db)
    s.log(Record(request_id="a", query_chars=1, chosen_model="cheap",
                 cost_usd=0.001, latency_ms=10.0))
    s.log(Record(request_id="b", query_chars=1, chosen_model="strong",
                 cost_usd=0.003, latency_ms=20.0))
    summ = s.summary()
    assert summ["n"] == 2
    assert summ["by_model"] == {"cheap": 1, "strong": 1}
    assert summ["avg_cost"] == pytest.approx(0.002)


# --------------------------------------------------------------------------- #
# router service
# --------------------------------------------------------------------------- #
def test_router_service_returns_valid_model(service):
    d = service.route("hello there")
    assert d.chosen_model in {"cheap", "mid", "strong"}
    assert 0.0 <= d.predicted_quality <= 1.0
    assert d.est_cost_usd > 0
    assert set(d.scores) == {"cheap", "mid", "strong"}


def test_router_service_unfitted_raises():
    class Unfitted:
        P_ = None
    with pytest.raises(RuntimeError):
        RouterService(Unfitted(), {})


# --------------------------------------------------------------------------- #
# proxy endpoints
# --------------------------------------------------------------------------- #
def _client(service, tmp_db):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    return TestClient(create_app(service, TelemetryStore(tmp_db), demo_mode=True))


def test_healthz(service, tmp_db):
    c = _client(service, tmp_db)
    j = c.get("/healthz").json()
    assert j["status"] == "ok"
    assert j["models"] == ["cheap", "mid", "strong"]


def test_chat_completions_openai_shape(service, tmp_db):
    c = _client(service, tmp_db)
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "what is 2+2"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion"
    assert j["choices"][0]["message"]["role"] == "assistant"
    assert j["model"] in {"cheap", "mid", "strong"}
    assert "tokenguard" in j


def test_chat_completions_logs_telemetry(service, tmp_db):
    store = TelemetryStore(tmp_db)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    c = TestClient(create_app(service, store, demo_mode=True))
    c.post("/v1/chat/completions",
           json={"messages": [{"role": "user", "content": "hi"}]})
    assert store.count() == 1


def test_chat_completions_no_user_message(service, tmp_db):
    c = _client(service, tmp_db)
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "system", "content": "be nice"}]})
    assert r.status_code == 400