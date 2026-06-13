"""Deterministic seeding across all libraries used in the project.

Every experiment script calls ``set_global_seed(cfg.experiment.seed)`` as its
first action. Torch is imported lazily so Day-1/2 code (pure pandas/numpy)
does not require it.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch + CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Trade a little speed for reproducibility in dissertation runs.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed yet (fine for Days 1-2 data work)
