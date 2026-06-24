"""Recurrence / personalisation stream — the regime where L2 memory should win.

Real routing traffic is not i.i.d.: the same users / topics recur, so a memory
of "which model won for this kind of query last time" is valuable. We simulate
this by clustering queries (by their eval_name / task, a proxy for topic) and
emitting a stream in *bursts* that revisit clusters, so a test-time memory can
accumulate and exploit per-cluster routing knowledge — something a single
context-free pass cannot.

This is the experiment that tests NESTOR's central claim: multi-timescale memory
beats a single-timescale bandit *when the stream has recurring structure*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench


def recurrence_stream(bench: RouterBench, n_clusters: int = 20,
                      burst_size: int = 50, revisits: int = 8,
                      seed: int = 42) -> RouterBench:
    """Return a RouterBench whose row order recurs in topic bursts.

    Parameters
    ----------
    n_clusters
        How many recurring topic groups to form (uses ``eval_name`` if present,
        otherwise a random partition).
    burst_size
        Queries emitted per cluster visit.
    revisits
        How many times each cluster is revisited across the stream (the source
        of recurrence the memory exploits).
    """
    rng = np.random.default_rng(seed)
    df = bench.df

    # group rows into recurring clusters: prefer the natural task label
    if "eval_name" in df.columns:
        labels = df["eval_name"].to_numpy()
        uniq = list(pd.unique(labels))
        rng.shuffle(uniq)
        uniq = uniq[:n_clusters] if len(uniq) > n_clusters else uniq
        groups = {u: np.where(labels == u)[0].tolist() for u in uniq}
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        groups = {i: list(chunk) for i, chunk in enumerate(np.array_split(idx, n_clusters))}

    cluster_keys = list(groups.keys())
    # build a visit schedule: each cluster appears `revisits` times, interleaved
    schedule = []
    for _ in range(revisits):
        order = cluster_keys[:]
        rng.shuffle(order)
        schedule.extend(order)

    rows: list[int] = []
    # a per-cluster cursor so revisits draw *fresh* (but same-topic) queries,
    # cycling if a cluster is exhausted — this is what makes the pattern learnable
    cursors = {k: 0 for k in groups}
    for k in schedule:
        members = groups[k]
        if not members:
            continue
        for _ in range(burst_size):
            c = cursors[k] % len(members)
            rows.append(members[c])
            cursors[k] += 1

    if not rows:  # degenerate fallback
        rows = list(range(len(df)))

    stream_df = df.iloc[rows].reset_index(drop=True)
    return RouterBench(stream_df)


def cluster_key_fn(bench: RouterBench):
    """Return a function row_index -> cluster id, for memory keying / recurrence
    detection in the router (queries in the same task share a key neighbourhood)."""
    if "eval_name" in bench.df.columns:
        labels = bench.df["eval_name"].to_numpy()
        uniq = {u: i for i, u in enumerate(pd.unique(labels))}
        return lambda i: uniq.get(labels[i], -1)
    return lambda i: 0