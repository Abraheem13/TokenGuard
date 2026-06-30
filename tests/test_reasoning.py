"""Tests for reasoning benchmark answer extraction + scoring (no downloads)."""

from tokenguard.reasoning.datasets import extract_answer, is_correct, _extract_boxed


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
