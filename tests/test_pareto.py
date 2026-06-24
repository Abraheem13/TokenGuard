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
