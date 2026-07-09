"""Reasoning benchmarks: GSM8K, MATH-500, GPQA-Diamond — loaders + scoring."""

from __future__ import annotations

import re
from pathlib import Path


def load_benchmark(name: str, split: str = "test", limit: int | None = None):
    name = name.lower()
    if name == "gsm8k":
        return _load_gsm8k(split, limit)
    if name in ("math500", "math-500"):
        return _load_math500(limit)
    if name in ("gpqa_diamond", "gpqa"):
        return _load_gpqa_diamond(limit)
    if name in ("aime24", "aime-24", "aime2024"):
        return _load_aime24(limit)
    raise ValueError(f"unknown benchmark {name!r}")


def _load_gsm8k(split, limit):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-{i}", "question": ex["question"], "answer": gold})
    return out


def _load_math500(limit):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        out.append({"id": f"math500-{i}", "question": ex["problem"],
                    "answer": ex["answer"]})
    return out


def _load_gpqa_diamond(limit):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        correct = ex["Correct Answer"]
        incorrect = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"],
                     ex["Incorrect Answer 3"]]
        options = [correct] + incorrect
        letters = ["A", "B", "C", "D"]
        q = ex["Question"] + "\n" + "\n".join(
            f"{letters[j]}) {opt}" for j, opt in enumerate(options))
        out.append({"id": f"gpqa-{i}", "question": q, "answer": "A",
                    "options": options})
    return out


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str, benchmark: str) -> str:
    benchmark = benchmark.lower()
    if benchmark == "gsm8k":
        if "####" in text:
            tail = text.split("####")[-1]
            m = _NUM.findall(tail)
            if m:
                return m[0].replace(",", "")
        nums = _NUM.findall(text)
        return nums[-1].replace(",", "") if nums else ""
    if benchmark in ("math500", "math-500", "aime24", "aime-24", "aime2024"):
        return _extract_boxed(text)
    if benchmark in ("gpqa_diamond", "gpqa"):
        m = re.findall(r"\b([A-D])\b", text.upper())
        return m[-1] if m else ""
    return text.strip()


def _extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed")
    if idx == -1:
        nums = _NUM.findall(text)
        return nums[-1] if nums else text.strip()
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return text[i + 1:].strip()


def _norm(s: str) -> str:
    s = s.strip().replace(" ", "").replace(",", "").replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "").rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


# --- DEER grader adoption (sympy equivalence for math/aime) --------------
_DEER_GRADER = "unset"

def _get_deer_grader():
    global _DEER_GRADER
    if _DEER_GRADER == "unset":
        try:
            import sys as _sys
            _deer = Path(__file__).resolve().parents[3] / "external" / "DEER"
            (_deer / "utils" / "__init__.py").touch(exist_ok=True)
            _sys.path.insert(0, str(_deer))
            from utils.grader import math_equal as _me
            _DEER_GRADER = _me
        except Exception:
            _DEER_GRADER = None
    return _DEER_GRADER


from functools import lru_cache as _lru

@_lru(maxsize=200000)
def _graded_equal(p: str, g: str) -> bool:
    me = _get_deer_grader()
    if me is None:
        return False
    try:
        return bool(me(p, g, timeout=True))
    except Exception:
        return False


def is_correct(pred: str, gold: str, benchmark: str) -> bool:
    p = extract_answer(pred, benchmark)
    if _norm(p) == _norm(gold):
        return True
    # sympy equivalence (DEER grader) for open-math benchmarks only
    if benchmark in ("math500", "math-500", "aime24", "aime-24", "aime2024",
                     "gsm8k") and p and gold:
        return _graded_equal(p, gold)
    return False


def _load_aime24(limit):
    """AIME 2024 (30 problems; integer answers 0-999). avg@k is obtained by
    running the harness k times with different --seed values and aggregating
    with scripts/ntc_genseed_agg.py."""
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        get = lambda ex: (ex["problem"], str(ex["answer"]).strip())
    except Exception:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        get = lambda ex: (ex["Problem"], str(ex["Answer"]).strip())
    out = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        q, a = get(ex)
        out.append({"id": f"aime24-{i}", "question": q, "answer": a})
    return out
