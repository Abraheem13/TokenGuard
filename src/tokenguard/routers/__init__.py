"""Routers: base interface, static baselines (Day 2), learned baselines
(Day 3: MF, BERT, kNN, cascade), contrastive router (Day 4), nested online
router (Day 5)."""

from tokenguard.routers.base import Router
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter
from tokenguard.routers.mf_router import MatrixFactorizationRouter
from tokenguard.routers.knn_router import KNNRouter
from tokenguard.routers.cascade_router import CascadeRouter

__all__ = [
    "Router", "ConstantRouter", "OracleRouter", "RandomMixRouter",
    "MatrixFactorizationRouter", "KNNRouter", "CascadeRouter",
]

# BERT router is imported lazily (pulls in torch/transformers). Import via:
#   from tokenguard.routers.bert_router import BertClassifierRouter
