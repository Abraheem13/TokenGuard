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
