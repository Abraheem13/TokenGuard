"""Evaluation runner: fit routers, trace frontiers, save tables and the
frontier figure. Every later day re-uses this runner unchanged, so all
dissertation figures share one code path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (cluster, CI)
import matplotlib.pyplot as plt
import pandas as pd

from tokenguard.data.routerbench import RouterBench
from tokenguard.eval import metrics
from tokenguard.routers.base import Router
from tokenguard.routers.static import ConstantRouter
from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.eval")


class EvalRunner:
    """Fit on ``train``, evaluate frontiers on ``test``, persist artifacts."""

    def __init__(
        self,
        train: RouterBench,
        test: RouterBench,
        results_dir: str | Path,
        figures_dir: str | Path,
        tag: str,
    ):
        self.train, self.test = train, test
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.tag = tag
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.frontiers: dict[str, pd.DataFrame] = {}

        # Shared references for AIQ window and APGR endpoints: the constant
        # weak/strong routers evaluated on THIS test split.
        weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
        strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
        self.weak_point = (float(weak["cost"]), float(weak["quality"]))
        self.strong_point = (float(strong["cost"]), float(strong["quality"]))
        self.c_lo, self.c_hi = self.weak_point[0], self.strong_point[0]
        logger.info(
            "References — weak: q=%.4f @ $%.6f | strong: q=%.4f @ $%.6f",
            self.weak_point[1], self.weak_point[0],
            self.strong_point[1], self.strong_point[0],
        )

    # ------------------------------------------------------------------ #
    def run(self, routers: list[Router]) -> pd.DataFrame:
        """Fit + frontier every router; return the summary table."""
        rows = []
        for router in routers:
            router.fit(self.train)
            frontier = router.frontier(self.test)
            self.frontiers[router.name] = frontier
            frontier.to_csv(
                self.results_dir / f"{self.tag}-frontier-{router.name}.csv",
                index=False,
            )
            if len(frontier) > 1:  # frontier-capable router
                rows.append(
                    metrics.summarise_router(
                        router.name, frontier, self.c_lo, self.c_hi,
                        self.weak_point, self.strong_point,
                    )
                )
            else:  # single-point router: report the point, AIQ/APGR n/a
                pt = frontier.iloc[0]
                rows.append(
                    {
                        "router": router.name,
                        "n_points": 1,
                        "min_cost": pt["cost"],
                        "max_quality": pt["quality"],
                        "aiq": float("nan"),
                        "apgr": float("nan"),
                    }
                )
            logger.info("Evaluated %-18s (%d frontier points)", router.name, len(frontier))

        summary = pd.DataFrame(rows).sort_values("aiq", ascending=False, na_position="last")
        summary.to_csv(self.results_dir / f"{self.tag}-summary.csv", index=False)
        return summary.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def plot(self, title: str) -> Path:
        """Cost-quality frontier figure (dissertation Figure 2)."""
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
        palette = plt.cm.tab10.colors

        for i, (name, frontier) in enumerate(self.frontiers.items()):
            color = palette[i % len(palette)]
            if len(frontier) == 1:
                ax.scatter(
                    frontier["cost"], frontier["quality"],
                    marker="*", s=180, color=color, zorder=5, label=name,
                )
            else:
                style = dict(linestyle="--", linewidth=1.4) if name == "oracle" else {}
                ax.plot(
                    frontier["cost"], frontier["quality"],
                    marker="o", markersize=3.5, color=color, label=name, **style,
                )

        ax.set_xlabel("Mean cost per query (USD)")
        ax.set_ylabel("Mean quality (RouterBench performance)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        out = self.figures_dir / f"{self.tag}-frontier.png"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved frontier figure: %s", out)
        return out
