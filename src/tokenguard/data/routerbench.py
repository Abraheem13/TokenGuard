"""RouterBench dataset loading and canonicalisation.

RouterBench (Hu et al., 2024, arXiv:2403.12031) ships ~405K samples spanning
11 LLMs and 8 task datasets, with **precomputed** model responses, per-call
cost, and a correctness/performance score for every (sample, model) pair.
This is the property that makes the whole dissertation feasible on a zero
API budget: routers are trained and evaluated by *table lookup*, never by
calling the underlying models.

Wire format on the Hugging Face Hub (``withmartian/routerbench``): a pickled
pandas DataFrame. Per-model columns are encoded with a separator, e.g.::

    "gpt-4-1106-preview|total_cost", "gpt-4-1106-preview|performance"

Because the exact column inventory can differ between dataset revisions, this
loader **detects the schema at runtime** rather than hard-coding it, and
fails loudly with a column dump if expectations are violated. The detected
schema is also written next to the processed file so the dissertation's data
section can report it exactly.

Canonical processed format (saved as parquet):

* index column ``sample_id`` (str)
* ``prompt`` (str), ``eval_name`` (str — the source task/benchmark)
* for each model M: ``perf::M`` (float in [0, 1]) and ``cost::M`` (float, USD)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.data")

SEP_CANDIDATES = ("|",)  # observed separator in published RouterBench pickles
PERF_KEYS = ("performance", "perf", "score", "accuracy")
COST_KEYS = ("total_cost", "cost")


# --------------------------------------------------------------------------- #
# Schema detection                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RouterBenchSchema:
    """What we discovered about the raw frame at runtime."""

    models: tuple[str, ...]
    perf_suffix: str
    cost_suffix: str
    prompt_col: str
    eval_col: str
    sample_id_col: str | None
    sep: str

    def perf_col(self, model: str) -> str:
        return f"{model}{self.sep}{self.perf_suffix}"

    def cost_col(self, model: str) -> str:
        return f"{model}{self.sep}{self.cost_suffix}"


def detect_schema(df: pd.DataFrame) -> RouterBenchSchema:
    """Infer the model list and column conventions from the raw frame.

    Raises ``ValueError`` with a full column dump if detection fails, so any
    upstream dataset revision change is caught on Day 1, not on Day 5.
    """
    cols = list(df.columns)

    sep = next((s for s in SEP_CANDIDATES if any(s in c for c in cols)), None)
    if sep is None:
        raise ValueError(
            "Could not find a model/metric separator in columns. "
            f"Columns were:\n{cols}"
        )

    suffixes: dict[str, set[str]] = {}
    for col in cols:
        if sep not in col:
            continue
        model, suffix = col.rsplit(sep, 1)
        suffixes.setdefault(suffix, set()).add(model)

    perf_suffix = next((k for k in PERF_KEYS if k in suffixes), None)
    cost_suffix = next((k for k in COST_KEYS if k in suffixes), None)
    if perf_suffix is None or cost_suffix is None:
        raise ValueError(
            "Could not identify performance/cost column suffixes. "
            f"Suffixes found: {sorted(suffixes)}\nColumns: {cols}"
        )

    # Models must expose BOTH metrics; anything else is metadata, not a model.
    models = tuple(sorted(suffixes[perf_suffix] & suffixes[cost_suffix]))
    if len(models) < 2:
        raise ValueError(
            f"Detected fewer than 2 routable models: {models}. Columns: {cols}"
        )

    prompt_col = next(
        (c for c in ("prompt", "question", "input", "query") if c in cols), None
    )
    eval_col = next(
        (c for c in ("eval_name", "dataset", "task", "benchmark") if c in cols), None
    )
    if prompt_col is None or eval_col is None:
        raise ValueError(
            f"Missing prompt/eval columns. Looked for prompt in "
            f"('prompt','question','input','query') and eval in "
            f"('eval_name','dataset','task','benchmark').\nColumns: {cols}"
        )
    sample_id_col = next((c for c in ("sample_id", "id") if c in cols), None)

    logger.info(
        "Detected schema: %d models, perf='%s', cost='%s', prompt='%s', eval='%s'",
        len(models), perf_suffix, cost_suffix, prompt_col, eval_col,
    )
    return RouterBenchSchema(
        models=models,
        perf_suffix=perf_suffix,
        cost_suffix=cost_suffix,
        prompt_col=prompt_col,
        eval_col=eval_col,
        sample_id_col=sample_id_col,
        sep=sep,
    )


# --------------------------------------------------------------------------- #
# Canonicalisation                                                            #
# --------------------------------------------------------------------------- #
def canonicalise(df: pd.DataFrame, schema: RouterBenchSchema) -> pd.DataFrame:
    """Project the raw frame onto the canonical column layout.

    Rows where *any* model is missing either metric are dropped (and counted),
    so every downstream router sees a complete (sample x model) reward table —
    a requirement for fair frontier comparisons.
    """
    out = pd.DataFrame(
        {
            "sample_id": (
                df[schema.sample_id_col].astype(str)
                if schema.sample_id_col
                else df.index.astype(str)
            ),
            "prompt": df[schema.prompt_col].astype(str),
            "eval_name": df[schema.eval_col].astype(str),
        }
    )
    for m in schema.models:
        out[f"perf::{m}"] = pd.to_numeric(df[schema.perf_col(m)], errors="coerce")
        out[f"cost::{m}"] = pd.to_numeric(df[schema.cost_col(m)], errors="coerce")

    before = len(out)
    out = out.dropna().reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        logger.warning(
            "Dropped %d/%d rows with missing perf/cost values (%.2f%%)",
            dropped, before, 100 * dropped / before,
        )

    # Sanity: performance should live in [0, 1] for frontier maths.
    perf_cols = [c for c in out.columns if c.startswith("perf::")]
    pmin, pmax = out[perf_cols].min().min(), out[perf_cols].max().max()
    if pmin < -1e-9 or pmax > 1 + 1e-9:
        logger.warning(
            "Performance outside [0,1] (min=%.4f, max=%.4f) — Day-2 metrics "
            "will min-max normalise per task.", pmin, pmax,
        )
    return out


# --------------------------------------------------------------------------- #
# Public dataset object                                                       #
# --------------------------------------------------------------------------- #
class RouterBench:
    """Canonical RouterBench wrapper used by every router and experiment."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = tuple(
            sorted(c.removeprefix("perf::") for c in df.columns if c.startswith("perf::"))
        )
        self.tasks = tuple(sorted(df["eval_name"].unique()))

    # -- construction ------------------------------------------------------ #
    @classmethod
    def download(
        cls,
        repo_id: str,
        filename: str,
        raw_dir: str | Path,
        processed_dir: str | Path,
        force: bool = False,
    ) -> "RouterBench":
        """Download the raw pickle from HF Hub, canonicalise, cache parquet."""
        raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = processed_dir / f"{Path(filename).stem}.canonical.parquet"

        if parquet_path.exists() and not force:
            logger.info("Loading cached canonical parquet: %s", parquet_path)
            return cls(pd.read_parquet(parquet_path))

        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s/%s from the Hugging Face Hub...", repo_id, filename)
        local = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=raw_dir
        )
        raw = pd.read_pickle(local)
        schema = detect_schema(raw)
        canonical = canonicalise(raw, schema)
        canonical.to_parquet(parquet_path, index=False)
        with open(parquet_path.with_suffix(".schema.json"), "w") as fh:
            json.dump(
                {
                    "models": schema.models,
                    "perf_suffix": schema.perf_suffix,
                    "cost_suffix": schema.cost_suffix,
                    "prompt_col": schema.prompt_col,
                    "eval_col": schema.eval_col,
                    "n_rows": len(canonical),
                },
                fh,
                indent=2,
            )
        logger.info("Saved canonical parquet: %s (%d rows)", parquet_path, len(canonical))
        return cls(canonical)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "RouterBench":
        return cls(pd.read_parquet(path))

    # -- accessors ---------------------------------------------------------- #
    def perf_matrix(self) -> np.ndarray:
        """(n_samples, n_models) performance matrix, model order = self.models."""
        return self.df[[f"perf::{m}" for m in self.models]].to_numpy(dtype=float)

    def cost_matrix(self) -> np.ndarray:
        """(n_samples, n_models) cost matrix, model order = self.models."""
        return self.df[[f"cost::{m}" for m in self.models]].to_numpy(dtype=float)

    # -- splits -------------------------------------------------------------- #
    def split_random(self, test_size: float, seed: int) -> tuple["RouterBench", "RouterBench"]:
        """Task-stratified random split (default research split)."""
        rng = np.random.default_rng(seed)
        test_mask = np.zeros(len(self.df), dtype=bool)
        for task in self.tasks:
            idx = np.flatnonzero((self.df["eval_name"] == task).to_numpy())
            n_test = max(1, int(round(test_size * len(idx))))
            test_mask[rng.choice(idx, size=n_test, replace=False)] = True
        return (
            RouterBench(self.df[~test_mask].reset_index(drop=True)),
            RouterBench(self.df[test_mask].reset_index(drop=True)),
        )

    def split_leave_one_task_out(self, held_out_task: str) -> tuple["RouterBench", "RouterBench"]:
        """Out-of-distribution split for the Day-5 shift experiment."""
        if held_out_task not in self.tasks:
            raise ValueError(f"Unknown task '{held_out_task}'. Tasks: {self.tasks}")
        mask = (self.df["eval_name"] == held_out_task).to_numpy()
        return (
            RouterBench(self.df[~mask].reset_index(drop=True)),
            RouterBench(self.df[mask].reset_index(drop=True)),
        )

    # -- reporting ----------------------------------------------------------- #
    def summary(self) -> pd.DataFrame:
        """Per-model mean performance / mean cost — Table 1 of the dissertation."""
        rows = []
        perf, cost = self.perf_matrix(), self.cost_matrix()
        for j, m in enumerate(self.models):
            rows.append(
                {
                    "model": m,
                    "mean_perf": perf[:, j].mean(),
                    "mean_cost_usd": cost[:, j].mean(),
                    "perf_per_dollar": perf[:, j].mean() / max(cost[:, j].mean(), 1e-12),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("mean_perf", ascending=False)
            .reset_index(drop=True)
        )

    def oracle_stats(self) -> dict[str, float]:
        """Upper bound: best model per sample (quality-first oracle)."""
        perf, cost = self.perf_matrix(), self.cost_matrix()
        best = perf.argmax(axis=1)
        rows = np.arange(len(best))
        return {
            "oracle_perf": float(perf[rows, best].mean()),
            "oracle_cost": float(cost[rows, best].mean()),
        }
