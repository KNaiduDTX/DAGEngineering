"""
test_invariants.py — small, dependency-free checks of the load-bearing guarantees.

Run:  python test_invariants.py
"""

from dyndag import (Budget, GraphProjection, NodeType, Provenance, Scheduler,
                    SQLiteEventStore, default_registry)
from dyndag.core import Event, EventType, now_iso
from dyndag.graph import SingleAssignmentError
from dyndag.steering import beam_prune


GOAL = {"topic": "t", "subtopics": ["a", "b"], "target_score": 0.80}


def _fresh():
    return Scheduler(SQLiteEventStore(":memory:"), default_registry(), budget=Budget())


def test_single_assignment_write_once():
    """A second NODE_COMPLETED for the same node must raise (SSA: write once)."""
    g = GraphProjection()
    prov = Provenance("r", "r:b0", 0, now_iso(), None, ()).as_dict()
    g.apply(Event("r", EventType.NODE_CREATED, {"node_type": "scout", "input_ids": [],
                                                "provenance": prov}, node_id="s.g0"))
    g.apply(Event("r", EventType.NODE_STARTED, {"tier": "local", "cost": 1.0}, node_id="s.g0"))
    g.apply(Event("r", EventType.NODE_COMPLETED, {"output": 1, "score": None}, node_id="s.g0"))
    try:
        g.apply(Event("r", EventType.NODE_COMPLETED, {"output": 2}, node_id="s.g0"))
    except SingleAssignmentError:
        return
    raise AssertionError("second completion was not rejected")


def test_cross_run_determinism():
    """Two independent runs of the deterministic fleet yield identical structure
    and identical terminal output."""
    r1 = _fresh().run(dict(GOAL))
    r2 = _fresh().run(dict(GOAL))
    assert r1.graph.structural_hash() == r2.graph.structural_hash(), "structure differs"
    assert r1.terminal_output == r2.terminal_output, "terminal output differs"
    assert r1.reason == r2.reason == r1.reason.GOAL_SATISFIED


def test_beam_prune_keeps_top_k():
    g = GraphProjection()
    prov = Provenance("r", "r:b0", 0, now_iso(), None, ()).as_dict()
    for i, sc in enumerate([0.9, 0.1, 0.5]):
        g.apply(Event("r", EventType.NODE_CREATED,
                      {"node_type": "engineer", "input_ids": [], "provenance": prov},
                      node_id=f"n{i}.g0"))
        g.apply(Event("r", EventType.NODE_STARTED, {"tier": "local", "cost": 1}, node_id=f"n{i}.g0"))
        g.apply(Event("r", EventType.NODE_COMPLETED, {"output": None, "score": sc}, node_id=f"n{i}.g0"))
        g.nodes[f"n{i}.g0"].state = g.nodes[f"n{i}.g0"].state.CREATED  # make them frontier-eligible
    kept, dropped = beam_prune(g, ["n0.g0", "n1.g0", "n2.g0"], width=2)
    assert kept == ["n0.g0", "n2.g0"], f"expected top-2 by score, got {kept}"
    assert dropped == ["n1.g0"]


def test_budget_terminates_runaway():
    """A pathological generation cap forces a bounded, deterministic termination
    reason rather than an infinite unroll."""
    s = Scheduler(SQLiteEventStore(":memory:"), default_registry(),
                  budget=Budget(max_generation=2))  # too small to reach the goal
    r = s.run(dict(GOAL))
    assert r.reason.value in ("generation_cap", "no_progress"), r.reason.value


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)} invariant tests passed.")


if __name__ == "__main__":
    main()
