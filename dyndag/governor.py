"""
governor.py — governance & failure containment (Layer 7 + cost).

A self-constructing graph will happily construct forever, so bounds are not
optional. The governor holds every cap in one place and is consulted before each
tick and before every patch is applied:

  * max_generation   — depth bound (unrolling limit)
  * max_nodes        — breadth/size bound
  * max_total_cost   — spend bound (the 'do not burn the local-model budget' cap)
  * retry_budget     — GLOBAL retry count across the whole run, not per-call.
                       This is the defense against retry storms, where an LLM
                       loop, the SDK, the engine, and the provider each retry and
                       amplify one failure into an outage.

When a bound trips the run terminates deterministically with a reason, rather
than degrading unpredictably.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import TerminationReason
from .graph import GraphProjection


@dataclass
class Budget:
    max_generation: int = 12
    max_nodes: int = 200
    max_total_cost: float = 500.0
    retry_budget: int = 8


class CostGovernor:
    def __init__(self, budget: Budget):
        self.budget = budget
        self.total_cost = 0.0
        self.retries_used = 0

    def charge(self, cost: float) -> None:
        self.total_cost += cost

    def spend_retry(self) -> bool:
        """Consume one unit of the global retry budget. False => budget exhausted."""
        if self.retries_used >= self.budget.retry_budget:
            return False
        self.retries_used += 1
        return True

    def check(self, graph: GraphProjection, generation: int) -> TerminationReason | None:
        """Return a termination reason if any bound is breached, else None."""
        if generation > self.budget.max_generation:
            return TerminationReason.GENERATION_CAP
        if len(graph.nodes) > self.budget.max_nodes:
            return TerminationReason.BUDGET_EXHAUSTED
        if self.total_cost > self.budget.max_total_cost:
            return TerminationReason.BUDGET_EXHAUSTED
        return None

    def would_exceed_nodes(self, graph: GraphProjection, adding: int) -> bool:
        return len(graph.nodes) + adding > self.budget.max_nodes

    def report(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "retries_used": self.retries_used,
            "retry_budget": self.budget.retry_budget,
        }
