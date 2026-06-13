"""Routers: base interface (Day 2), static baselines (Day 2), learned
baselines (Day 3), contrastive router (Day 4), nested online router (Day 5)."""

from tokenguard.routers.base import Router
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter

__all__ = ["Router", "ConstantRouter", "OracleRouter", "RandomMixRouter"]
