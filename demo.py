"""
demo.py — end-to-end proof that the loop closes.

Run:  python demo.py

It exercises, in one run:
  * subgraph emission twice (initial decomposition + a critic-driven refinement)
  * the critic-as-controller loop converging in a bounded number of generations
  * SSA versioning (engineer.g3 -> engineer.g5) giving iteration WITHOUT a cycle
  * generation-monotonic acyclicity (guard rejects a back-edge)
  * provenance-at-write (store rejects a lineage-less creation)
  * event-sourced replay reconstructing an identical graph
  * cost routing (local vs cloud) + the governor's bounds
  * the blackboard / lineage projections
"""

from dyndag import (Budget, GraphProjection, NodeType, Provenance, Scheduler,
                    SQLiteEventStore, default_registry, lineage_report,
                    provenance_gaps, render_blackboard, verify_replay)
from dyndag.core import Event, EventType
from dyndag.store import SQLiteEventStore as Store  # alias for the guard demo


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    goal = {
        "topic": "stateless semantic enrichment",
        "subtopics": ["ingestion", "reasoning", "provenance"],
        "target_score": 0.80,
    }

    rule("RUN")
    store = SQLiteEventStore(":memory:")
    sched = Scheduler(store=store, registry=default_registry(), budget=Budget())
    result = sched.run(goal)

    print(f"run_id            : {result.run_id}")
    print(f"termination       : {result.reason.value}")
    print(f"generations (ticks): {result.ticks}")
    print(f"events in log     : {store.count(result.run_id)}")
    print(f"nodes constructed : {len(result.graph.nodes)}")
    if result.terminal_output:
        print(f"terminal brief    : score={result.terminal_output['final_score']} "
              f"covered={result.terminal_output['covered']}")

    rule("CONSTRUCTION TRACE  (how the DAG unrolled on the fly)")
    for e in store.read(result.run_id):
        if e.type == EventType.NODE_CREATED:
            p = e.payload["provenance"]
            print(f"  +node  {e.node_id:<20} type={e.payload['node_type']:<12} "
                  f"gen={p['generation']} parents={p['parent_ids']}")
        elif e.type == EventType.SUBGRAPH_EMITTED:
            print(f"   \u21b3 {e.node_id} EMITTED subgraph: {e.payload['added_nodes']}")
        elif e.type == EventType.NODE_COMPLETED and e.payload.get("signal") in (
            "needs_refinement", "done", "goal"):
            print(f"   =signal {e.node_id:<20} -> {e.payload['signal']} "
                  f"(score={e.payload.get('score')})")

    rule("BLACKBOARD  (Layer 4 — the DAG as a Kanban board)")
    print(render_blackboard(result.graph, wip_limit=4))

    rule("LINEAGE + PROVENANCE  (Layers 1,5 — no NULL provenance by construction)")
    print(lineage_report(result.graph))

    rule("COST + GOVERNANCE  (Layer 5/7 — local vs cloud routing)")
    per_tier = {}
    for rec in result.graph.nodes.values():
        if rec.tier:
            per_tier[rec.tier] = per_tier.get(rec.tier, 0.0) + rec.cost
    print(f"  cost by tier    : {per_tier}")
    print(f"  governor        : {result.governor.report()}")

    rule("DURABILITY  (Layer 5 — replay reconstructs an identical graph)")
    ok, live_h, rep_h = verify_replay(store, result.graph, result.run_id)
    print(f"  live  hash      : {live_h}")
    print(f"  replay hash     : {rep_h}")
    print(f"  identical       : {ok}")

    rule("GUARD 1 — acyclicity (Layer 6): a back-edge is rejected")
    g = result.graph
    # try to add an edge from a late node back to an early one (would form a cycle)
    late = "reviewer.g6"
    early = "goal.g0"
    try:
        g.assert_can_add_edge(late, early)
        print("  ERROR: guard failed to reject the back-edge")
    except Exception as exc:
        print(f"  rejected as expected: {type(exc).__name__}: {exc}")

    rule("GUARD 2 — provenance-at-write (Layer 5): lineage-less creation rejected")
    bad_store = Store(":memory:")
    try:
        bad_store.append(Event(
            run_id="run-x", type=EventType.NODE_CREATED, node_id="orphan.g0",
            payload={"node_type": "scout", "input_ids": [], "provenance": None},
        ))
        print("  ERROR: store accepted a node with no provenance")
    except Exception as exc:
        print(f"  rejected as expected: {type(exc).__name__}: {exc}")

    rule("SUMMARY")
    gaps = provenance_gaps(result.graph)
    checks = {
        "reached_goal": result.reason.value == "goal_satisfied",
        "refinement_happened": any(n.startswith("engineer.g") and n != "engineer.g3"
                                   for n in result.graph.nodes),
        "replay_identical": ok,
        "zero_provenance_gaps": not gaps,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "a core invariant failed"
    print("\nAll core invariants hold.")


if __name__ == "__main__":
    main()
