"""Halting rules, overhead-inclusive replay, calibration and the paired tests."""

import math

import numpy as np

import ntc_primary_stats as PS
import ntc_w1_stats as S


def probe(answer, conf, tokens, entropy=0.1, cost=5):
    return {"answer": answer, "confidence": conf, "ckpt_tokens": tokens,
            "first_entropy": entropy, "n_probe_tokens": cost}


STREAM = [probe("C", 0.40, 256), probe("B", 0.80, 512), probe("B", 0.92, 768),
          probe("B", 0.97, 1024)]


def test_confidence_threshold_halts_at_first_confident_probe():
    assert S.deer_policy(STREAM, lam=0.9) == 2
    assert S.deer_policy(STREAM, lam=0.99) is None


def test_agreement_needs_m_consecutive_equal_answers():
    assert S.agree_policy(STREAM, m=2, bm="mmlu_pro") == 2
    assert S.agree_policy(STREAM, m=3, bm="mmlu_pro") == 3
    assert S.agree_policy(STREAM, m=4, bm="mmlu_pro") is None


def test_fusion_requires_agreement_and_smoothed_confidence():
    # agreement alone (m=2) halts at probe 2; a gate at theta=0.8 delays it to probe 3
    assert S.ntc_v2_policy(STREAM, m=2, theta=0.5, bm="mmlu_pro") == 2
    assert S.ntc_v2_policy(STREAM, m=2, theta=0.8, bm="mmlu_pro") == 3


def test_never_halt_is_the_null_action():
    assert S.never_halt_policy(STREAM) is None


def test_replay_charges_every_probe_paid():
    trace = {"probes": STREAM, "gold": "B", "natural_correct": True, "n_total_tokens": 2000}
    ok, tok = S.per_item([trace], "mmlu_pro", S.agree_policy, {"m": 2})
    assert bool(ok[0]) and tok[0] == 768 + 3 * 5
    ok, tok = S.per_item([trace], "mmlu_pro", S.never_halt_policy, {})
    assert bool(ok[0]) and tok[0] == 2000 + 4 * 5


def test_calibration_returns_the_null_action_when_nothing_is_safe():
    rng = np.random.default_rng(0)
    warm = []
    for _ in range(60):
        wrong = [probe("B", 0.99, 256 * (k + 1)) for k in range(4)]
        warm.append({"probes": wrong, "gold": "A", "natural_correct": bool(rng.random() < 0.9),
                     "n_total_tokens": 2000})
    _, (fam, _) = S.calibrate(warm, "mmlu_pro", eps=0.01)
    assert fam == "NEVER-HALT"


def test_t_quantile_approaches_the_normal_quantile():
    assert abs(S._t_quantile(0.975, 10_000) - 1.95996) < 1e-3
    assert S._t_quantile(0.975, 10) > S._t_quantile(0.975, 100)


def test_exact_sign_test_and_mcnemar():
    p, pos, n = PS.sign_test([1.0] * 10)
    assert (pos, n) == (10, 10) and math.isclose(p, 2 / 1024)
    b, c, p = S.mcnemar_exact(np.array([True, True, False]), np.array([False, False, False]))
    assert (b, c) == (2, 0) and math.isclose(p, 0.5)
