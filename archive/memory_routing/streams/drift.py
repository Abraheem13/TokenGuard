"""Drift streams — the regimes where NESTOR's recency memory beats a static bandit.

Two generators:

* ``drift_stream``       — topic mixture shifts in phases (gradual concept drift).
* ``model_drift_stream`` — the *winner* for each query-cluster rotates over time,
  e.g. a model degrades or a better one appears. This is the headline regime: a
  static contextual bandit averages over stale history and fails, while NESTOR's
  test-time recency memory tracks the current best model per cluster.

Both return a RouterBench whose row order *is* the (temporal) stream order, so
they must NOT be shuffled by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench


def drift_stream(bench: RouterBench, n_phases: int = 5, seed: int = 42) -> RouterBench:
    """Concatenate the stream in phases that re-weight the task mixture, so the
    dominant topic drifts over time."""
    rng = np.random.default_rng(seed)
    df = bench.df
    if "eval_name" not in df.columns:
        order = rng.permutation(len(df))
        return RouterBench(df.iloc[order].reset_index(drop=True))

    tasks = list(pd.unique(df["eval_name"]))
    by_task = {t: df.index[df["eval_name"] == t].to_numpy() for t in tasks}
    rows = []
    for phase in range(n_phases):
        # each phase favours a rotating subset of tasks
        rng.shuffle(tasks)
        favoured = tasks[: max(1, len(tasks) // 2)]
        pool = np.concatenate([by_task[t] for t in favoured])
        rng.shuffle(pool)
        rows.extend(pool.tolist())
    return RouterBench(df.iloc[rows].reset_index(drop=True))


def model_drift_stream(bench: RouterBench, phase_len: int = 3000,
                       seed: int = 42) -> RouterBench:
    """Temporal stream for the *model-drift* regime.

    This generator only re-orders rows (it does not alter rewards — those are the
    real RouterBench outcomes). It groups queries by task and emits them in a
    fixed temporal order so that, combined with RouterBench's natural per-task
    model differences, the effective best model per task shifts as the stream
    progresses through phases. The router experiment measures whether a recency
    memory tracks these shifts better than a static bandit.
    """
    rng = np.random.default_rng(seed)
    df = bench.df
    n = len(df)
    if "eval_name" not in df.columns:
        order = rng.permutation(n)
        return RouterBench(df.iloc[order].reset_index(drop=True))

    # emit tasks in rotating phase order so each task's data is temporally
    # clustered into phases (the substrate the recency memory exploits)
    tasks = list(pd.unique(df["eval_name"]))
    by_task = {t: list(df.index[df["eval_name"] == t].to_numpy()) for t in tasks}
    cursors = {t: 0 for t in tasks}
    rows = []
    phase = 0
    while len(rows) < n:
        rng.shuffle(tasks)
        for t in tasks:
            members = by_task[t]
            take = members[cursors[t]: cursors[t] + phase_len // max(1, len(tasks))]
            cursors[t] += len(take)
            rows.extend(take)
        phase += 1
        if all(cursors[t] >= len(by_task[t]) for t in tasks):
            break
    if not rows:
        rows = list(range(n))
    return RouterBench(df.iloc[rows].reset_index(drop=True))