#!/usr/bin/env bash
# TokenGuard — schema fix for the real RouterBench layout.
# Run from the repo root:  bash fix_schema.sh
set -euo pipefail
if [[ ! -d src/tokenguard/data || ! -d tests ]]; then
  echo "ERROR: run from the tokenguard repo root." >&2; exit 1
fi
echo "==> Applying schema fix..."

cat > src/tokenguard/data/routerbench.py << 'TG_FIX_EOF'
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
COST_KEYS = ("total_cost", "cost")
# Some revisions store performance under a suffix instead of the bare column;
# we try the bare column first (the layout published for routerbench_0shot),
# then fall back to these suffixes.
PERF_SUFFIX_KEYS = ("performance", "perf", "score", "accuracy")
# Suffixes that are never a routable metric (free-text generations, metadata).
IGNORE_SUFFIXES = ("model_response", "response", "output", "generation")
# Bare columns that are metadata rather than models.
NON_MODEL_BARE = (
    "sample_id", "id", "prompt", "question", "input", "query",
    "eval_name", "dataset", "task", "benchmark",
    "oracle_model_to_route_to", "oracle",
)


# --------------------------------------------------------------------------- #
# Schema detection                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RouterBenchSchema:
    """What we discovered about the raw frame at runtime.

    ``perf_layout`` records how performance is stored:
      * ``"bare"``    — performance lives in the bare model-name column
                        (the published ``routerbench_0shot`` layout),
      * ``"suffix"``  — performance lives in ``model<sep><perf_suffix>``.
    Cost is always suffixed (``model<sep><cost_suffix>``).
    """

    models: tuple[str, ...]
    perf_layout: str          # "bare" | "suffix"
    perf_suffix: str | None   # set only when perf_layout == "suffix"
    cost_suffix: str
    prompt_col: str
    eval_col: str
    sample_id_col: str | None
    sep: str

    def perf_col(self, model: str) -> str:
        if self.perf_layout == "bare":
            return model
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

    # Group suffixed columns by their suffix → set of model stems.
    suffixes: dict[str, set[str]] = {}
    for col in cols:
        if sep not in col:
            continue
        model, suffix = col.rsplit(sep, 1)
        suffixes.setdefault(suffix, set()).add(model)

    cost_suffix = next((k for k in COST_KEYS if k in suffixes), None)
    if cost_suffix is None:
        raise ValueError(
            "Could not identify a cost suffix. "
            f"Suffixes found: {sorted(suffixes)}\nColumns: {cols}"
        )
    cost_models = suffixes[cost_suffix]

    # Performance layout detection.
    perf_suffix = next((k for k in PERF_SUFFIX_KEYS if k in suffixes), None)
    if perf_suffix is not None:
        # Suffixed-performance layout (older/synthetic format).
        perf_layout = "suffix"
        models = tuple(sorted(suffixes[perf_suffix] & cost_models))
    else:
        # Bare-column performance layout (published routerbench_0shot):
        # a model is a bare column that (a) has a matching <model>|total_cost
        # and (b) is numeric.
        bare_cols = [
            c for c in cols
            if sep not in c and c not in NON_MODEL_BARE
        ]
        models = tuple(
            sorted(
                c for c in bare_cols
                if c in cost_models
                and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
            )
        )
        perf_layout = "bare"

    if len(models) < 2:
        raise ValueError(
            "Detected fewer than 2 routable models. "
            f"perf_layout={perf_layout!r}, cost_suffix={cost_suffix!r}, "
            f"models={models}\nColumns: {cols}"
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
        "Detected schema: %d models, perf_layout='%s', cost_suffix='%s', "
        "prompt='%s', eval='%s'",
        len(models), perf_layout, cost_suffix, prompt_col, eval_col,
    )
    return RouterBenchSchema(
        models=models,
        perf_layout=perf_layout,
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
                    "models": list(schema.models),
                    "perf_layout": schema.perf_layout,
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
TG_FIX_EOF
echo "   wrote src/tokenguard/data/routerbench.py"

cat > tests/test_day1.py << 'TG_FIX_EOF'
"""Day 1 tests.

The loader is tested against a synthetic fixture that mimics the published
RouterBench wire format (``model|metric`` columns), so the parsing logic is
verified *before* the real download — and continues to be tested in CI
without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.utils.seed import set_global_seed

MODELS = ["gpt-4-1106-preview", "claude-instant-v1", "mistral-8x7b-chat"]
TASKS = ["mmlu", "gsm8k", "hellaswag"]


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """Synthetic raw frame in RouterBench wire format (240 rows)."""
    rng = np.random.default_rng(0)
    n = 240
    data = {
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "prompt": [f"Question number {i}?" for i in range(n)],
        "eval_name": [TASKS[i % len(TASKS)] for i in range(n)],
    }
    for j, m in enumerate(MODELS):
        # Bigger models: higher performance, higher cost (realistic ordering).
        data[f"{m}|performance"] = rng.binomial(1, 0.55 + 0.15 * j, size=n).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.0001, 0.001, size=n) * (j + 1) ** 2
    return pd.DataFrame(data)


@pytest.fixture()
def bench(raw_frame: pd.DataFrame) -> RouterBench:
    schema = detect_schema(raw_frame)
    return RouterBench(canonicalise(raw_frame, schema))


# --------------------------- schema detection ------------------------------ #
def test_detect_schema_finds_models_and_columns(raw_frame: pd.DataFrame) -> None:
    schema = detect_schema(raw_frame)
    assert set(schema.models) == set(MODELS)
    assert schema.perf_suffix == "performance"
    assert schema.cost_suffix == "total_cost"
    assert schema.prompt_col == "prompt"
    assert schema.eval_col == "eval_name"


def test_detect_schema_fails_loudly_without_metric_columns() -> None:
    bad = pd.DataFrame({"prompt": ["x"], "eval_name": ["mmlu"]})
    with pytest.raises(ValueError):
        detect_schema(bad)


def test_metadata_columns_are_not_mistaken_for_models(raw_frame: pd.DataFrame) -> None:
    # A column with the separator but only one metric must not create a model.
    raw_frame["oracle|performance"] = 1.0  # no matching oracle|total_cost
    schema = detect_schema(raw_frame)
    assert "oracle" not in schema.models


def test_real_bare_column_layout_is_parsed() -> None:
    """Lock the published routerbench_0shot layout.

    Performance is stored in the *bare* model-name column; cost is suffixed
    with ``|total_cost``; ``|model_response`` holds free text; and
    ``oracle_model_to_route_to`` is metadata. This is the exact shape the
    real download exposes, so we encode it as a regression guard.
    """
    rng = np.random.default_rng(0)
    n = 120
    models = ["gpt-4-1106-preview", "claude-v2", "mistralai/mistral-7b-chat"]
    data = {
        "sample_id": [f"s{i}" for i in range(n)],
        "prompt": [f"Q{i}" for i in range(n)],
        "eval_name": [TASKS[i % len(TASKS)] for i in range(n)],
    }
    for m in models:
        data[m] = rng.binomial(1, 0.6, size=n).astype(float)        # bare perf
        data[f"{m}|model_response"] = [f"resp {i}" for i in range(n)]  # text
        data[f"{m}|total_cost"] = rng.uniform(1e-5, 1e-3, size=n)    # cost
    data["oracle_model_to_route_to"] = rng.integers(0, len(models), size=n)
    raw = pd.DataFrame(data)

    schema = detect_schema(raw)
    assert schema.perf_layout == "bare"
    assert set(schema.models) == set(models)
    assert "oracle_model_to_route_to" not in schema.models

    bench = RouterBench(canonicalise(raw, schema))
    assert not bench.df.isna().any().any()
    assert bench.perf_matrix().shape == (n, len(models))
    # bare perf column must be read as the performance value, not the text col
    assert set(np.unique(bench.perf_matrix())).issubset({0.0, 1.0})


# --------------------------- canonicalisation ------------------------------ #
def test_canonical_frame_has_no_nans_and_correct_shape(bench: RouterBench) -> None:
    assert not bench.df.isna().any().any()
    assert bench.perf_matrix().shape == (len(bench.df), len(MODELS))
    assert bench.cost_matrix().shape == (len(bench.df), len(MODELS))


def test_rows_with_missing_values_are_dropped(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, f"{MODELS[0]}|performance"] = np.nan
    schema = detect_schema(raw_frame)
    canonical = canonicalise(raw_frame, schema)
    assert len(canonical) == len(raw_frame) - 1


# --------------------------------- splits ---------------------------------- #
def test_random_split_is_stratified_and_disjoint(bench: RouterBench) -> None:
    train, test = bench.split_random(test_size=0.2, seed=42)
    assert len(train.df) + len(test.df) == len(bench.df)
    assert set(train.df.sample_id).isdisjoint(set(test.df.sample_id))
    # every task appears on both sides (stratification)
    assert set(train.tasks) == set(TASKS) and set(test.tasks) == set(TASKS)


def test_random_split_is_reproducible(bench: RouterBench) -> None:
    _, t1 = bench.split_random(0.2, seed=42)
    _, t2 = bench.split_random(0.2, seed=42)
    assert list(t1.df.sample_id) == list(t2.df.sample_id)


def test_leave_one_task_out_split(bench: RouterBench) -> None:
    train, test = bench.split_leave_one_task_out("gsm8k")
    assert set(test.df.eval_name) == {"gsm8k"}
    assert "gsm8k" not in set(train.df.eval_name)


def test_leave_one_task_out_rejects_unknown_task(bench: RouterBench) -> None:
    with pytest.raises(ValueError):
        bench.split_leave_one_task_out("not-a-task")


# ------------------------------ summaries ---------------------------------- #
def test_oracle_dominates_every_single_model(bench: RouterBench) -> None:
    oracle_perf = bench.oracle_stats()["oracle_perf"]
    assert oracle_perf >= bench.summary()["mean_perf"].max() - 1e-12


# ------------------------------- config ------------------------------------ #
def test_load_default_config_from_repo_root() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.experiment.seed == 42
    assert cfg.data.hf_repo_id == "withmartian/routerbench"
    assert 0.0 < cfg.data.test_size < 1.0
    assert cfg.router.slow_update_every > 0


def test_config_rejects_unknown_keys(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  sedd: 1\n")  # typo must fail loudly
    with pytest.raises(KeyError):
        load_config("configs/default.yaml", override_path=bad)


# -------------------------------- seeding ---------------------------------- #
def test_global_seed_makes_numpy_deterministic() -> None:
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)
TG_FIX_EOF
echo "   wrote tests/test_day1.py"

echo ""
echo "==> Done. Now run:"
echo "    make test                       # expect 26 passed"
echo "    python scripts/day1_download_data.py --force"
echo "    python scripts/day2_static_baselines.py"