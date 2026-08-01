"""
activities.py — recorded activities (Temporal-style durability at the executor seam).

A real executor is several *nondeterministic* steps: retrieve, call a model, parse.
The engine already records a node's final ExecutionResult (NODE_COMPLETED), so replay
never re-runs an executor — but a node that is RE-executed (the failure-path retry, or
crash recovery) would re-run every sub-step, re-spending money and re-firing side
effects. A recorded activity closes that gap:

  * the first time a step runs, its result is recorded as an ACTIVITY_RECORDED event;
  * on any later execution of the same node, the recorded result is REPLAYED instead
    of the step being called again — so the call is exactly-once (INV-5.1 extended
    below node granularity, and the mechanism for exactly-once external side effects).

The recorder respects the scheduler's execute/commit boundary (see scheduler.run):
during the PURE execute phase it only *reads* previously-recorded results and *buffers*
new ones; the serial commit phase is what writes the ACTIVITY_RECORDED events. So the
execute phase still performs no shared-state writes and stays parallelizable.

Keys are chosen by the executor and are stable across a node's re-executions (e.g.
`f"draft_rev{revision}"`), which is what makes replay match. Recorded results must be
JSON-serializable — they round-trip through the event log.
"""

from __future__ import annotations

from typing import Any, Callable


class ActivityRecorder:
    """Per-execute helper handed to an executor via Context.

    `prior` holds results already recorded for this node (from folding the log);
    `recorded` accumulates the *new* (key, result) pairs the commit phase must persist.
    """

    def __init__(self, prior: dict[str, Any] | None = None):
        self._prior: dict[str, Any] = dict(prior or {})
        self._new: dict[str, Any] = {}
        self.recorded: list[tuple[str, Any]] = []

    def record(self, key: str, fn: Callable[[], Any]) -> Any:
        """Return the recorded result for `key` if one exists (replay, no call);
        otherwise call `fn()` once, buffer its result for commit, and return it."""
        if key in self._prior:
            return self._prior[key]          # replay a prior execution's step
        if key in self._new:
            return self._new[key]            # already ran this pass (idempotent within run)
        result = fn()
        self._new[key] = result
        self.recorded.append((key, result))
        return result
