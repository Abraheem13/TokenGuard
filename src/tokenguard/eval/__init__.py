"""Evaluation: cost-quality metrics and the experiment runner (Day 2).

Note: ``EvalRunner`` is intentionally NOT imported here. It depends on
``routers`` (for reference baselines), while ``routers.base`` depends on this
package's ``metrics`` — eagerly importing the runner would create a circular
import. Import the runner explicitly where needed:

    from tokenguard.eval.runner import EvalRunner
"""

from tokenguard.eval.metrics import (
    DEFAULT_LAMBDAS,
    aiq,
    apgr,
    evaluate_choices,
    pareto_front,
    quality_at_cost,
    summarise_router,
)

__all__ = [
    "DEFAULT_LAMBDAS", "aiq", "apgr", "evaluate_choices", "pareto_front",
    "quality_at_cost", "summarise_router",
]