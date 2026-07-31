"""
steering.py — goal-conditioned construction (Layer 3).

Three knobs turn a reactive graph into a goal-seeking one:

  GoalPredicate     — the terminal test. A GOAL-signalled node is only *accepted*
                      as terminal if this predicate holds; otherwise the search
                      continues. This is what stops a premature "done".

  value_function    — scores a node for beam-style pruning of the frontier so
                      compute concentrates on promising branches.

  ConstraintFilter  — arc-consistency in miniature: before a proposed node is
                      instantiated, drop the ones that cannot contribute to the
                      target (or would breach a budget). Killing branches at
                      CONSTRUCTION time is cheaper than generate-then-verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .graph import GraphProjection, NodeRecord
from .nodes import NodeSpec


@dataclass
class CoverageGoal:
    """Concrete goal predicate for the demo: the terminal brief must cover every
    required subtopic AND clear the target score. The predicate is evaluated on
    the terminal candidate's output, not on a vibe."""
    subtopics: list[str]
    target_score: float

    def satisfied(self, terminal_output: dict) -> bool:
        if not isinstance(terminal_output, dict) or terminal_output.get("kind") != "brief":
            return False
        covered = set(terminal_output.get("covered", []))
        score_ok = terminal_output.get("final_score", 0.0) >= self.target_score
        return set(self.subtopics).issubset(covered) and score_ok


def value_function(rec: NodeRecord) -> float:
    """A node's promise. Critic scores dominate when present; otherwise deeper
    (more-informed) nodes are mildly preferred. Used only for pruning decisions."""
    if rec.score is not None:
        return rec.score
    return 0.5 + 0.01 * rec.generation


class ConstraintFilter:
    """
    Applied to every emitted patch before instantiation. Returns the surviving
    NodeSpecs and a list of (spec, reason) for the pruned ones (which are
    recorded as NODE_PRUNED events for audit). The predicate is pluggable; the
    default keeps everything (the demo's patches are already tight), but the hook
    is where "must satisfy schema S / must not exceed budget B / must plausibly
    advance goal G" lives.
    """
    def __init__(self, predicate: Optional[Callable[[NodeSpec], tuple[bool, str]]] = None):
        self._predicate = predicate or (lambda spec: (True, ""))

    def apply(self, specs: list[NodeSpec]) -> tuple[list[NodeSpec], list[tuple[NodeSpec, str]]]:
        kept, pruned = [], []
        for spec in specs:
            ok, reason = self._predicate(spec)
            (kept if ok else pruned).append(spec if ok else (spec, reason))
        return kept, pruned


def beam_prune(graph: GraphProjection, candidate_ids: list[str], width: int) -> tuple[list[str], list[str]]:
    """Keep the top-`width` frontier candidates by value; return (kept, dropped).
    A no-op when width >= len(candidates). This is the lever that makes a wide
    speculative expansion affordable."""
    if width <= 0 or width >= len(candidate_ids):
        return candidate_ids, []
    ranked = sorted(candidate_ids, key=lambda n: value_function(graph.nodes[n]), reverse=True)
    return ranked[:width], ranked[width:]
