"""Tests for reasoning benchmark answer extraction + scoring (no downloads)."""

from tokenguard.reasoning.datasets import _extract_boxed, extract_answer, is_correct


def test_gsm8k_hash_extract():
    assert extract_answer("reasoning #### 42", "gsm8k") == "42"
    assert is_correct("#### 42", "42", "gsm8k")


def test_gsm8k_last_number_fallback():
    assert extract_answer("the total is 18 apples", "gsm8k") == "18"


def test_gsm8k_comma_number():
    assert extract_answer("#### 1,234", "gsm8k") == "1234"


def test_math_boxed_nested():
    assert _extract_boxed("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert is_correct("\\boxed{7}", "7", "math500")


def test_gpqa_letter():
    assert extract_answer("The answer is C.", "gpqa_diamond") == "C"
    assert is_correct("answer: A", "A", "gpqa")


def test_unknown_benchmark_raises():
    import pytest
    with pytest.raises(ValueError):
        from tokenguard.reasoning.datasets import load_benchmark
        load_benchmark("nonsense")


def test_answer_space_intervention_keeps_the_answer_and_nests_distractors():
    from tokenguard.reasoning.datasets import restrict_options

    opts = [f"option {j}" for j in range(10)]
    for item_seed in range(25):
        answer = item_seed % 10
        kept = {}
        for k in (2, 4, 10):
            sub, ai = restrict_options(opts, answer, k, item_seed)
            assert len(sub) == k and len(set(sub)) == k
            assert sub[ai] == opts[answer]
            assert restrict_options(opts, answer, k, item_seed) == (sub, ai)
            kept[k] = set(sub)
        assert kept[2] < kept[4] < kept[10] == set(opts)
    # the position of the correct option is not fixed
    positions = {restrict_options(opts, 0, 4, s)[1] for s in range(40)}
    assert len(positions) > 1

