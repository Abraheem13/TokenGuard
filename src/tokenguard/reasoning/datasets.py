"""Benchmark loaders, answer extraction and the answer grader.

Six benchmarks are loaded from their public Hugging Face releases: GSM8K,
MATH-500, GPQA-Diamond, MMLU-Pro, AIME-2024 and AIME-2025. Open-ended answers
are graded by the symbolic equivalence checker distributed with DEER
(external/DEER/utils/grader.py); its verdicts are cached on disk in
experiments/ntc/grader_cache.json so that every reported number is
reproducible without re-running the grader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_benchmark(name: str, split: str = "test", limit: int | None = None,
                   shuffle_options: bool = True):
    """Load a benchmark as a list of {id, question, answer} dicts.

    GPQA-Diamond lists the correct option first; with shuffle_options (the
    default) its four options are shuffled deterministically per item.
    """
    name = name.lower()
    if name == "gsm8k":
        return _load_gsm8k(split, limit)
    if name in ("math500", "math-500"):
        return _load_math500(limit)
    if name in ("gpqa_diamond", "gpqa"):
        return _load_gpqa_diamond(limit, shuffle_options)
    if name in ("aime24", "aime-24", "aime2024"):
        return _load_aime24(limit)
    if name in ("aime25", "aime-25", "aime2025"):
        return _load_aime25(limit)
    if name in ("mmlu_pro", "mmlu-pro", "mmlupro"):
        return _load_mmlu_pro(limit)
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


def _load_gpqa_diamond(limit, shuffle_options=True):
    import random

    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        correct = ex["Correct Answer"]
        incorrect = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"],
                     ex["Incorrect Answer 3"]]
        all4 = [correct] + incorrect
        order = [0, 1, 2, 3]
        if shuffle_options:
            random.Random(1234 + i).shuffle(order)
        options = [all4[j] for j in order]
        letters = ["A", "B", "C", "D"]
        ans_letter = letters[order.index(0)]
        q = ex["Question"] + "\n" + "\n".join(
            f"{letters[j]}) {opt}" for j, opt in enumerate(options))
        out.append({"id": f"gpqa-{i}", "question": q, "answer": ans_letter,
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
    if benchmark in ("math500", "math-500", "aime24", "aime-24", "aime2024", "aime25", "aime-25", "aime2025"):
        return _extract_boxed(text)
    if benchmark in ("gpqa_diamond", "mmlu_pro", "mmlu-pro", "mmlupro", "gpqa"):
        _hi = "J" if benchmark in ("mmlu_pro", "mmlu-pro", "mmlupro") else "D"
        m = re.findall(r"\b([A-" + _hi + r"])\b", text.upper())
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


# ------------------------------------------------------------------ grader
_DEER_GRADER = "unset"


def _get_deer_grader():
    """DEER's math_equal, imported on first use; None if it cannot be loaded."""
    global _DEER_GRADER
    if _DEER_GRADER == "unset":
        try:
            import sys as _sys
            # Python >= 3.13 removed typing.io, which the antlr4 runtime used by
            # the grader's LaTeX parser still imports. Without this shim the
            # import fails and every symbolic comparison silently returns False.
            if "typing.io" not in _sys.modules:
                import types as _types
                import typing as _typing
                _m = _types.ModuleType("typing.io")
                _m.TextIO, _m.IO, _m.BinaryIO = (_typing.TextIO, _typing.IO,
                                                 _typing.BinaryIO)
                _sys.modules["typing.io"] = _m
            _deer = Path(__file__).resolve().parents[3] / "external" / "DEER"
            (_deer / "utils" / "__init__.py").touch(exist_ok=True)
            _sys.path.insert(0, str(_deer))
            from utils.grader import math_equal as _me
            _DEER_GRADER = _me
        except Exception:
            _DEER_GRADER = None
    return _DEER_GRADER


import signal as _signal
from functools import lru_cache as _lru


def _alarm_handler(signum, frame):
    raise TimeoutError("grader timeout")


# Verdicts persisted on disk: every analysis reads them, so a result does not
# depend on re-running the grader.
_GC_PATH = Path(__file__).resolve().parents[3] / "experiments" / "ntc" / "grader_cache.json"
_GC = None
_GC_DIRTY = False


def _gc_load():
    global _GC
    if _GC is None:
        try:
            _GC = json.loads(_GC_PATH.read_text())
        except Exception:
            _GC = {}
        import atexit
        atexit.register(_gc_flush)
    return _GC


def _gc_flush():
    global _GC_DIRTY
    if _GC_DIRTY and _GC is not None:
        try:
            _GC_PATH.parent.mkdir(parents=True, exist_ok=True)
            _GC_PATH.write_text(json.dumps(_GC))
            _GC_DIRTY = False
        except Exception:
            pass


@_lru(maxsize=200000)
def _graded_equal(p: str, g: str) -> bool:
    gc = _gc_load()
    key = p + "\x1f" + g
    if key in gc:
        return bool(gc[key])
    v = _graded_equal_compute(p, g)
    global _GC_DIRTY
    gc[key] = bool(v)
    _GC_DIRTY = True
    return v


def _graded_equal_compute(p: str, g: str) -> bool:
    me = _get_deer_grader()
    if me is None:
        return False
    old_h = _signal.signal(_signal.SIGALRM, _alarm_handler)
    _signal.alarm(5)
    try:
        return bool(me(p, g, timeout=False))
    except Exception:
        return False
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_h)


def is_correct(pred: str, gold: str, benchmark: str) -> bool:
    """Whether a model output answers `gold`, by normalised string match and,
    on the open-answer benchmarks, by the symbolic grader."""
    p = extract_answer(pred, benchmark)
    if _norm(p) == _norm(gold):
        return True
    if benchmark in ("math500", "math-500", "aime24", "aime-24", "aime2024",
                     "gsm8k", "aime25", "aime-25", "aime2025") and p and gold:
        return _graded_equal(p, gold)
    return False


def _load_aime24(limit):
    """AIME 2024: 30 problems with integer answers from 0 to 999."""
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


def _load_aime25(limit):
    """AIME 2025: 30 problems, from DEER's local copy if present, else Hugging Face."""
    import json as _json
    local = Path(__file__).resolve().parents[3] / "external" / "DEER" / "data" / "aime25" / "test.jsonl"
    out = []
    if local.exists():
        for i, line in enumerate(local.read_text().splitlines()):
            if limit and i >= limit:
                break
            ex = _json.loads(line)
            q = ex.get("problem") or ex.get("question") or ex.get("Problem")
            a = ex.get("answer") or ex.get("expected_answer") or ex.get("Answer")
            out.append({"id": f"aime25-{i}", "question": str(q),
                        "answer": str(a).strip()})
        return out
    from datasets import load_dataset
    try:
        ds = load_dataset("math-ai/aime25", split="test")
    except Exception:
        ds = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        q = ex.get("problem") or ex.get("question")
        a = ex.get("answer") or ex.get("expected_answer")
        out.append({"id": f"aime25-{i}", "question": str(q),
                    "answer": str(a).strip()})
    return out


def _load_mmlu_pro(limit):
    """MMLU-Pro: up to ten options per question; the items are a seeded random sample.

    Options keep their source order; the position of the correct option already
    varies across items.
    """
    import random as _random

    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = list(range(len(ds)))
    _random.Random(4321).shuffle(idx)
    letters = "ABCDEFGHIJ"
    out = []
    for i in idx:
        if limit and len(out) >= limit:
            break
        ex = ds[i]
        opts = [o for o in ex["options"] if o not in (None, "N/A", "")]
        ai = ex.get("answer_index")
        if ai is None or ai >= len(opts):
            continue
        q = ex["question"] + "\n" + "\n".join(
            f"{letters[j]}) {o}" for j, o in enumerate(opts))
        out.append({"id": f"mmlupro-{i}", "question": q,
                    "answer": letters[ai], "options": opts})
    return out
