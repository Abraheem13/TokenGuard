"""Stream construction for the online experiments.

Two orderings of the test split into a request stream:

* ``shuffled_stream`` — i.i.d. ordering (queries arrive in random order). This
  is the stationary baseline: online adaptation should help modestly, mostly by
  refining the warm-started policy.

* ``shift_stream`` — *distribution shift*: queries are grouped by task family
  and the families arrive in sequential blocks (e.g. all maths, then all code,
  then all knowledge). This is the regime that separates a multi-timescale
  online router from a static one: as the active task family changes, a frozen
  router keeps applying a stale policy, while the nested router's FAST head
  re-adapts and its SLOW level consolidates — sustaining the frontier.

Both return a new ``RouterBench`` whose row order *is* the stream order.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench


def shuffled_stream(bench: RouterBench, seed: int = 42) -> RouterBench:
    """Random i.i.d. ordering of the rows."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bench.df))
    return RouterBench(bench.df.iloc[order].reset_index(drop=True))


def _task_family(eval_name: str) -> str:
    """Coarse family for a RouterBench task name (for block ordering)."""
    e = eval_name.lower()
    if e.startswith("mmlu"):
        return "knowledge"
    if "math" in e or e.startswith("grade-school") or "gsm" in e:
        return "math"
    if "mbpp" in e or "code" in e or "humaneval" in e:
        return "code"
    if "chinese" in e or "translation" in e:
        return "multilingual"
    if "hellaswag" in e or "winogrande" in e or "arc" in e:
        return "commonsense"
    return "other"


def shift_stream(bench: RouterBench, seed: int = 42, granularity: str = "task") -> RouterBench:
    """Block-sequential ordering that induces distribution shift.

    ``granularity``:
      * ``"task"`` (default) — each distinct ``eval_name`` is its own block.
        This exposes the *full* natural shift in RouterBench: with 86 tasks,
        the active task (and thus the best-suited model) changes 86 times, so a
        static policy tuned to the global average is repeatedly mismatched.
      * ``"family"`` — coarse families (math/code/knowledge/...) as blocks, a
        milder shift used for the ablation's "weak shift" condition.

    Within each block rows are shuffled; blocks arrive in a fixed order, so the
    input distribution changes abruptly at block boundaries.
    """
    rng = np.random.default_rng(seed)
    if granularity == "task":
        keys = bench.df["eval_name"].to_numpy()
        # deterministic block order: sort task names, but interleave so that
        # consecutive blocks are dissimilar (max shift). We simply sort here;
        # the abrupt per-task change is what drives the effect.
        block_order = sorted(set(keys))
    else:
        fams = bench.df["eval_name"].map(_task_family)
        keys = fams.to_numpy()
        block_order = ["math", "code", "knowledge", "commonsense",
                       "multilingual", "other"]

    pieces = []
    for block in block_order:
        idx = np.flatnonzero((keys == block))
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        pieces.append(idx)
    if not pieces:
        return shuffled_stream(bench, seed)
    order = np.concatenate(pieces)
    return RouterBench(bench.df.iloc[order].reset_index(drop=True))


def family_boundaries(bench_stream: RouterBench, granularity: str = "task") -> list[tuple[str, int]]:
    """Return (block_label, start_index) markers for plotting shift boundaries.

    With ``granularity="task"`` the markers are per-task; to keep the figure
    readable we only label a marker when the block changes, and the caller may
    thin them. With ``"family"`` the coarse families are used.
    """
    if granularity == "task":
        keys = bench_stream.df["eval_name"].to_numpy()
    else:
        keys = bench_stream.df["eval_name"].map(_task_family).to_numpy()
    marks, prev = [], None
    for i, f in enumerate(keys):
        if f != prev:
            marks.append((str(f), i))
            prev = f
    return marks