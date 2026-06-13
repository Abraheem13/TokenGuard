"""Project-wide logger: console (rich if available) + per-run file log.

Every experiment writes a timestamped log under experiments/results so each
figure/table in the dissertation can be traced to the exact run that made it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)

    try:
        from rich.logging import RichHandler

        console = RichHandler(rich_tracebacks=True, show_path=False)
        console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except ImportError:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fh = logging.FileHandler(log_dir / f"{name}-{stamp}.log")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)
    return logger
