"""
phi.py — φ-nodes: merging competing branches (SSA φ-functions made concrete).

In SSA, a φ-function at a control-flow merge selects which definition of a variable to use
based on which predecessor executed. Here the "competing definitions" are the outputs of
sibling branches that computed the same logical result K different ways — K solver samples
(self-consistency), or two refinement lineages — and a φ-node selects or COMBINES them by a
pluggable merge policy. The selector (which branch won, and why) is recorded on the node's
output (`output["phi"]`), so lineage shows the merge decision — the φ made auditable.

A φ-merge is a PURE, deterministic function of its inputs, read in id-sorted order with stable
tie-breaks, so the projection's structural hash is stable and replay reconstructs it exactly
(INV-5.2). Degenerate input (no candidates) is fail-closed — a MalformedResultError, handled
as a failed attempt (INV-4.2), never a silent pass.

Why competing branches need this: the SYNTHESIZER aggregates ONE approved branch (arity-1). A
φ-node is the principled generalization — arity-N with an explicit selection rule — which is
exactly what self-consistency (majority-vote over K samples) and best-of-N (argmax over critic
scores) require.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from .core import Signal
from .executors import ExecutionResult, Executor, MalformedResultError

# A candidate is (producer_node_id, output_dict, score_or_None). A MergePolicy maps the
# candidate list -> (chosen_output_dict, basis_dict); `basis` records WHY (scores / vote
# tallies) and MUST include "selected_from" (the winning producer id) for lineage.
Candidate = tuple
MergePolicy = Callable[[list], tuple]


def argmax_score(candidates: list) -> tuple:
    """Best-of-N: choose the candidate with the highest critic score. Ties break to the
    lowest producer id (candidates arrive id-sorted, and max keeps the first max). A None
    score sorts lowest, so unscored candidates only win when all are unscored."""
    best = max(candidates, key=lambda c: (c[2] if c[2] is not None else float("-inf")))
    basis = {"policy": "argmax_score", "scores": {c[0]: c[2] for c in candidates},
             "selected_from": best[0]}
    return best[1], basis


def majority_vote(key: str = "final_answer") -> MergePolicy:
    """Self-consistency: choose the value of `key` that appears most often across candidates.
    Ties break to the tied value with the highest summed score, then (via id-sorted scan) the
    lowest producer id. Returns a representative candidate's full output carrying the winner."""
    def _vote(candidates: list) -> tuple:
        counts = Counter(str(c[1].get(key)) for c in candidates)

        def _rank(val: str) -> tuple:
            summed = sum((c[2] or 0.0) for c in candidates if str(c[1].get(key)) == val)
            return counts[val], summed
        winner = max(counts, key=_rank)
        rep = next(c for c in candidates if str(c[1].get(key)) == winner)   # id-sorted first
        basis = {"policy": "majority_vote", "key": key, "votes": dict(counts),
                 "winner": winner, "selected_from": rep[0]}
        return rep[1], basis
    return _vote


def confidence_weighted_vote(key: str = "final_answer") -> MergePolicy:
    """Verifier-weighted self-consistency (the F1 merge): sum each candidate's score (its
    critic's confidence) per answer value and pick the answer with the greatest TOTAL confidence
    — consensus AND verification, not a blind headcount. A high-confidence lineage can outweigh
    several low-confidence ones; unscored candidates contribute weight 1.0, so this degrades to
    plain majority when no scores are present. Ties -> most votes, then lowest producer id."""
    def _vote(candidates: list) -> tuple:
        weight: dict = {}
        votes: dict = {}
        for _cid, out, score in candidates:
            val = str(out.get(key))
            weight[val] = weight.get(val, 0.0) + (score if score is not None else 1.0)
            votes[val] = votes.get(val, 0) + 1
        winner = max(weight, key=lambda v: (weight[v], votes[v]))
        rep = next(c for c in candidates if str(c[1].get(key)) == winner)   # id-sorted first
        basis = {"policy": "confidence_weighted_vote", "key": key,
                 "weights": {k: round(w, 4) for k, w in weight.items()},
                 "votes": votes, "winner": winner, "selected_from": rep[0]}
        return rep[1], basis
    return _vote


def union_covered(candidates: list) -> tuple:
    """Combine (not select): union `covered` and merge `sections` across competing drafts,
    keeping the highest-scoring draft's other scalars. The brief task's fan-in merge."""
    best = max(candidates, key=lambda c: (c[2] if c[2] is not None else float("-inf")))
    covered: list = []
    sections: dict = {}
    for _cid, out, _s in candidates:
        for st in out.get("covered", []) or []:
            if st not in covered:
                covered.append(st)
        sections.update(out.get("sections", {}) or {})
    merged = {**best[1], "covered": covered, "sections": sections}
    basis = {"policy": "union_covered", "selected_from": best[0],
             "merged_from": [c[0] for c in candidates]}
    return merged, basis


class PhiExecutor(Executor):
    """A φ-merge node. Reads its competing predecessors (id-sorted for stable ties), applies a
    merge policy, and emits the chosen/combined output with a `phi` record of the selector
    (candidates + basis + which branch won). `signal` is a constructor arg: Signal.GOAL when
    the φ-node is terminal (e.g. GSM8K self-consistency), Signal.CONTINUE for a mid-graph
    best-of-N merge. Pure & deterministic ⇒ replay-stable; empty candidates ⇒ fail-closed.

    A candidate's score is read from its output dict (`output["score"]`), so a producer that
    should be rankable by argmax_score emits its critic score there; majority_vote needs none."""

    def __init__(self, policy: MergePolicy, signal: Signal = Signal.GOAL,
                 kind: Optional[str] = None):
        self.policy = policy
        self.signal = signal
        self.kind = kind          # force the output "kind" (e.g. "solution"); else keep chosen's

    def run(self, inputs, params, ctx) -> ExecutionResult:
        candidates = [(nid, out, out.get("score"))
                      for nid, out in sorted(inputs.items()) if isinstance(out, dict)]
        if not candidates:
            raise MalformedResultError("phi-merge has no candidate inputs")
        chosen, basis = self.policy(candidates)
        output = dict(chosen)
        if self.kind is not None:
            output["kind"] = self.kind
        output["phi"] = {"candidates": [c[0] for c in candidates], **basis}
        return ExecutionResult(output=output, signal=self.signal)
