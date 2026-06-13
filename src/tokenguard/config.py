"""Typed configuration loading for TokenGuard.

Design decisions (documented for the dissertation's reproducibility section):

* A single YAML file (``configs/default.yaml``) is the source of truth.
* An optional override YAML is deep-merged on top, so experiments change only
  the keys they need; everything else stays pinned.
* The merged config is exposed as a read-only dataclass tree, so typos in key
  names fail loudly at load time instead of silently at experiment time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ENV_VAR = "TOKENGUARD_CONFIG"


# --------------------------------------------------------------------------- #
# Dataclass schema                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentConfig:
    tag: str = "dev"
    seed: int = 42
    results_dir: str = "experiments/results"
    figures_dir: str = "experiments/figures"


@dataclass(frozen=True)
class DataConfig:
    hf_repo_id: str = "withmartian/routerbench"
    hf_filename: str = "routerbench_0shot.pkl"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    split: str = "random"
    test_size: float = 0.2


@dataclass(frozen=True)
class RouterConfig:
    lambda_cost: float = 0.5
    encoder_model: str = "Qwen/Qwen3-0.6B"
    embedding_dim: int = 1024
    linucb_alpha: float = 1.0
    ema_beta: float = 0.99
    slow_update_every: int = 500


@dataclass(frozen=True)
class ProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8800
    telemetry_db: str = "experiments/telemetry.sqlite"
    pool: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_section(cls, raw: dict[str, Any]):
    """Instantiate a dataclass section, rejecting unknown keys loudly."""
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise KeyError(
            f"Unknown config keys for {cls.__name__}: {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )
    if "pool" in raw and isinstance(raw["pool"], list):
        raw = {**raw, "pool": tuple(tuple(sorted(d.items())) for d in raw["pool"])}
    return cls(**raw)


def load_config(
    path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> Config:
    """Load the default config, optionally deep-merged with an override file.

    Resolution order for the base config:
    1. explicit ``path`` argument,
    2. ``TOKENGUARD_CONFIG`` environment variable,
    3. ``configs/default.yaml`` relative to the working directory.
    """
    base_path = Path(path or os.environ.get(ENV_VAR, DEFAULT_CONFIG_PATH))
    if not base_path.exists():
        raise FileNotFoundError(
            f"Config not found at '{base_path}'. Run from the repo root, or set "
            f"{ENV_VAR}, or pass --config explicitly."
        )
    with open(base_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if override_path is not None:
        with open(override_path) as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})

    return Config(
        experiment=_build_section(ExperimentConfig, raw.get("experiment", {})),
        data=_build_section(DataConfig, raw.get("data", {})),
        router=_build_section(RouterConfig, raw.get("router", {})),
        proxy=_build_section(ProxyConfig, raw.get("proxy", {})),
    )
