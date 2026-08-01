"""
test_invariants.py — small, dependency-free checks of the load-bearing guarantees.

Run:  python test_invariants.py
"""

import random
import threading

from dyndag import (Budget, ExecutionResult, Executor, GraphProjection, NodeSpec,
                    NodeType, Provenance, Scheduler, Signal, SQLiteEventStore,
                    SubgraphPatch, TerminationReason, default_registry, verify_replay)
from dyndag.core import Event, EventType, now_iso
from dyndag.graph import SingleAssignmentError
from dyndag.steering import beam_prune


GOAL = {"topic": "t", "subtopics": ["a", "b"], "target_score": 0.80}
ONE = {"topic": "t", "subtopics": ["a"], "target_score": 0.80}  # smallest goal


def _fresh():
    return Scheduler(SQLiteEventStore(":memory:"), default_registry(), budget=Budget())


# --------------------------------------------------------------------------- #
# Failure injection — the deterministic mock fleet never raises, so these      #
# wrappers drive the NODE_FAILED -> retry -> reopen path that the substrate    #
# must survive (durability under failure is the whole point of event sourcing).#
# --------------------------------------------------------------------------- #
class _Flaky(Executor):
    """Wraps an executor and raises on its first `fail_times` invocations, then
    delegates to the real one. `fail_times` large => always fails."""
    def __init__(self, inner: Executor, fail_times: int):
        self.inner = inner
        self.fail_times = fail_times

    def run(self, inputs, params, ctx):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("injected transient failure")
        return self.inner.run(inputs, params, ctx)


class _RandomFlaky(Executor):
    """Raises with probability `p` on each call (seeded RNG => reproducible)."""
    def __init__(self, inner: Executor, rng: random.Random, p: float):
        self.inner, self.rng, self.p = inner, rng, p

    def run(self, inputs, params, ctx):
        if self.rng.random() < self.p:
            raise RuntimeError("injected random failure")
        return self.inner.run(inputs, params, ctx)


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


def test_reopen_survives_replay_when_run_caps_after_failure():
    """The adversarial durability case: a node fails and is REOPENED, then the run
    hits a cap before the retry runs — leaving it CREATED in the live graph. If the
    reopen is an out-of-band state mutation instead of a logged fact, replay folds
    the log back to FAILED and the projections diverge (INV-5.2 broken). This test
    fails on a scheduler that reopens without an event."""
    reg = default_registry()
    reg[NodeType.SCOUT] = _Flaky(reg[NodeType.SCOUT], fail_times=99)  # always fails
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, reg, budget=Budget(max_generation=2, retry_budget=1))
    r = s.run(dict(ONE))
    assert s.governor.retries_used >= 1, "the reopen path was never exercised"
    ok, live_h, rep_h = verify_replay(store, r.graph, r.run_id)
    assert ok, f"replay diverged after a stranded reopen: live={live_h} replay={rep_h}"


def test_success_after_retry_reaches_goal_and_replays():
    """The happy failure case: a transient error that succeeds on retry must not
    perturb the outcome — the goal is still reached and the log still replays
    bit-identically."""
    reg = default_registry()
    reg[NodeType.SCOUT] = _Flaky(reg[NodeType.SCOUT], fail_times=1)  # one transient error
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, reg, budget=Budget(retry_budget=4))
    r = s.run(dict(GOAL))
    assert s.governor.retries_used >= 1
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged after a successful retry"


def test_malformed_executor_output_is_fail_closed():
    """A real model will eventually return garbage. The scheduler must treat a
    malformed ExecutionResult as a failed attempt and NEVER fold it into the log
    or graph — and it must not crash the orchestrator doing so."""
    class _Garbage(Executor):
        def run(self, inputs, params, ctx):
            return "not an ExecutionResult"

    reg = default_registry()
    reg[NodeType.SCOUT] = _Garbage()
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, reg, budget=Budget(retry_budget=2))
    r = s.run(dict(ONE))                       # must return, not raise
    completed = [e for e in store.read(r.run_id)
                 if e.type == EventType.NODE_COMPLETED and (e.node_id or "").startswith("scout")]
    assert not completed, "scheduler folded unvalidated output into the log"
    assert s.governor.retries_used >= 1, "malformed output was not handled as a failed attempt"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_replay_equivalence_under_random_failures():
    """Property sweep: across many seeded random failure schedules, replaying the
    log ALWAYS reconstructs the live projection (INV-5.2) and the run ALWAYS ends
    with one of the four reasons (INV-7.1). Stdlib RNG keeps it deterministic and
    dependency-free; a Hypothesis port is a later upgrade."""
    four = {TerminationReason.GOAL_SATISFIED, TerminationReason.GENERATION_CAP,
            TerminationReason.BUDGET_EXHAUSTED, TerminationReason.NO_PROGRESS}
    seen = set()
    for seed in range(40):
        rng = random.Random(seed)
        reg = default_registry()
        for nt in list(reg):
            reg[nt] = _RandomFlaky(reg[nt], rng, p=0.35)
        store = SQLiteEventStore(":memory:")
        s = Scheduler(store, reg, budget=Budget(retry_budget=6, max_generation=12))
        r = s.run(dict(GOAL))
        ok, live_h, rep_h = verify_replay(store, r.graph, r.run_id)
        assert ok, f"seed {seed}: replay diverged live={live_h} replay={rep_h}"
        assert r.reason in four, f"seed {seed}: undefined termination {r.reason}"
        seen.add(r.reason)
    assert seen, "sweep executed no runs"


def test_cost_cap_enforced_within_a_wide_tick():
    """Finding C: max_total_cost must bound spend as it accrues *within* a tick,
    not only at the next top-of-loop check. With a wide scout frontier and a cap
    below the full batch cost, admission halts mid-batch — total overshoot is
    bounded by one node's cost, not the whole batch."""
    goal = {"topic": "t", "subtopics": [f"s{i}" for i in range(20)], "target_score": 0.80}
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, default_registry(), budget=Budget(max_total_cost=10.0))
    r = s.run(goal)
    assert r.reason == TerminationReason.BUDGET_EXHAUSTED, r.reason
    # 20 scouts @1 (after orchestrator @5) would reach 25; mid-tick enforcement
    # stops admission shortly after 10, bounded by the most expensive single node.
    assert s.governor.total_cost <= 10.0 + 5.0, s.governor.total_cost
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_batch_commit_ordering_is_deterministic():
    """Finding D: the execute/commit split must keep a deterministic, single-writer
    commit. Two runs produce an identical event sequence, and every NODE_STARTED in
    the scout batch precedes every NODE_COMPLETED in it (the whole batch is admitted
    before any of it commits — the ordering Phase 2's asyncio.gather relies on)."""
    def seq():
        store = SQLiteEventStore(":memory:")
        r = Scheduler(store, default_registry(), budget=Budget()).run(dict(GOAL))
        return [(e.type.value, e.node_id) for e in store.read(r.run_id)]
    s1, s2 = seq(), seq()
    assert s1 == s2, "event sequence is not deterministic across runs"
    starts = [i for i, (t, n) in enumerate(s1)
              if t == "node_started" and (n or "").startswith("scout")]
    comps = [i for i, (t, n) in enumerate(s1)
             if t == "node_completed" and (n or "").startswith("scout")]
    assert starts and comps and max(starts) < min(comps), \
        "a scout committed before the batch finished admitting (commit not batched)"


def test_recorded_activity_is_replayed_on_retry():
    """A nondeterministic sub-step wrapped in ctx.record must run EXACTLY ONCE across
    a failed attempt + retry: on the retry the recorded result is replayed, not
    re-called. Proves idempotency (corpus items #1 and #7) offline — no model needed,
    the 'model call' is a counter. This is the Phase 1 executor-seam durability."""
    class _FlakyRecorder(Executor):
        def __init__(self):
            self.model_calls = 0
            self.runs = 0

        def _call_model(self):
            self.model_calls += 1                 # the expensive/nondeterministic step
            return {"material": "result"}

        def run(self, inputs, params, ctx):
            self.runs += 1
            data = ctx.record("scout_fetch", self._call_model)   # recorded activity
            if self.runs == 1:
                raise RuntimeError("parse failure AFTER the recorded call, first attempt")
            return ExecutionResult(output={"kind": "material", **data}, signal=Signal.CONTINUE)

    eng = _FlakyRecorder()
    reg = default_registry()
    reg[NodeType.SCOUT] = eng
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget(retry_budget=4)).run(dict(ONE))
    assert eng.runs == 2, f"expected one failed attempt + one retry, got runs={eng.runs}"
    assert eng.model_calls == 1, f"recorded step re-ran: model_calls={eng.model_calls} (idempotency broken)"
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    ok, live_h, rep_h = verify_replay(store, r.graph, r.run_id)
    assert ok, f"replay diverged with recorded activities present: {live_h} vs {rep_h}"


def test_synthesizer_uses_reviewer_score_not_draft_selfreport():
    """The terminal score is the REVIEWER's independent judgment, not the draft's
    self-reported quality — the decoupling that lets a real critic drive the loop.
    An approving reviewer scores 0.95 over a revision-0 draft (self-report 0.60);
    the brief must carry 0.95."""
    class _ApprovingReviewer(Executor):
        def run(self, inputs, params, ctx):
            draft_id = next(k for k, v in inputs.items()
                            if isinstance(v, dict) and v.get("kind") == "draft")
            synth = NodeSpec(key="synthesizer", node_type=NodeType.SYNTHESIZER,
                             input_keys=(draft_id, ctx.self_id),
                             params={"subtopics": params["subtopics"],
                                     "target_score": params["target_score"]})
            return ExecutionResult(output={"verdict": "approve", "score": 0.95},
                                   signal=Signal.DONE, score=0.95,
                                   patch=SubgraphPatch(added_nodes=[synth]))

    reg = default_registry()
    reg[NodeType.REVIEWER] = _ApprovingReviewer()
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(dict(ONE))
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert r.terminal_output["final_score"] == 0.95, r.terminal_output["final_score"]
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_real_cost_recorded_from_usage():
    """Executor-reported token usage becomes real $ via the ModelRouter's tier pricing,
    recorded in NODE_COMPLETED and replayed. The governor's unit budget is unchanged
    (pre-call gate); this is the post-call actual (Layer 5 cost accounting)."""
    from dyndag import cost_report

    class _CloudReviewer(Executor):   # REVIEWER routes to the cloud tier (5/25 per MTok)
        def run(self, inputs, params, ctx):
            draft_id = next(k for k, v in inputs.items()
                            if isinstance(v, dict) and v.get("kind") == "draft")
            synth = NodeSpec(key="synthesizer", node_type=NodeType.SYNTHESIZER,
                             input_keys=(draft_id, ctx.self_id),
                             params={"subtopics": params["subtopics"],
                                     "target_score": params["target_score"]})
            return ExecutionResult(output={"verdict": "approve", "score": 0.9},
                                   signal=Signal.DONE, score=0.9,
                                   usage={"input_tokens": 1000, "output_tokens": 500},
                                   patch=SubgraphPatch(added_nodes=[synth]))

    reg = default_registry()
    reg[NodeType.REVIEWER] = _CloudReviewer()
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(dict(ONE))
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    reviewer = next(rec for nid, rec in r.graph.nodes.items() if nid.startswith("reviewer"))
    assert reviewer.usage == {"input_tokens": 1000, "output_tokens": 500}
    assert abs(reviewer.cost_usd - 0.0175) < 1e-9, reviewer.cost_usd   # (1000*5 + 500*25)/1e6
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged with cost recorded"
    rep = cost_report(r.graph)
    assert "1000 in" in rep and "500 out" in rep, rep


def test_concurrent_execute_matches_serial():
    """max_workers>1 parallelizes the pure execute phase WITHOUT changing semantics:
    identical structure, terminal output, and termination vs serial. Commit stays serial
    and in admitted order, so the run is deterministic under concurrency (Phase 2)."""
    serial = Scheduler(SQLiteEventStore(":memory:"), default_registry(),
                       budget=Budget(), max_workers=1).run(dict(GOAL))
    conc = Scheduler(SQLiteEventStore(":memory:"), default_registry(),
                     budget=Budget(), max_workers=4).run(dict(GOAL))
    assert serial.graph.structural_hash() == conc.graph.structural_hash(), "structure differs"
    assert serial.terminal_output == conc.terminal_output, "terminal output differs"
    assert serial.reason == conc.reason == TerminationReason.GOAL_SATISFIED


def test_batch_executes_concurrently():
    """The admitted batch runs in parallel under max_workers: three scouts each waiting
    on a shared 3-party barrier all clear it — only possible if they execute
    concurrently (a serial run would block one scout on the barrier forever)."""
    barrier = threading.Barrier(3, timeout=15)

    class _BarrierScout(Executor):
        def run(self, inputs, params, ctx):
            barrier.wait()   # releases only when all three scouts are in-flight together
            return ExecutionResult(output={"subtopic": params["subtopic"], "facts": []},
                                   signal=Signal.CONTINUE)

    reg = default_registry()
    reg[NodeType.SCOUT] = _BarrierScout()
    store = SQLiteEventStore(":memory:")
    goal = {"topic": "t", "subtopics": ["a", "b", "c"], "target_score": 0.80}
    r = Scheduler(store, reg, budget=Budget(), max_workers=3).run(goal)
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged under concurrent execution"


def test_retrieval_scout_ranks_corpus_and_records():
    """A real retrieval SCOUT ranks a corpus by relevance to its subtopic (deterministic
    lexical overlap), returns the top hits as facts, and records the retrieval as an
    activity — offline, no model. The relevant doc is retrieved; the unrelated one isn't."""
    from dyndag import RetrievalScoutExecutor

    reg = default_registry()
    reg[NodeType.SCOUT] = RetrievalScoutExecutor(top_k=2)
    corpus = [
        {"id": "d1", "text": "ingestion pipelines parse and load raw records"},
        {"id": "d2", "text": "provenance tracks lineage and audit of every write"},
        {"id": "d3", "text": "unrelated cooking recipes and gardening tips"},
    ]
    goal = {"topic": "data platform", "subtopics": ["ingestion"],
            "target_score": 0.80, "corpus": corpus}
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(goal)
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    scout = next(rec for nid, rec in r.graph.nodes.items() if nid.startswith("scout"))
    assert "d1" in scout.output["sources"], scout.output["sources"]      # relevant, retrieved
    assert "d3" not in scout.output["sources"], scout.output["sources"]  # irrelevant, not
    assert any(k.startswith("retrieve_") for k in r.graph.activities.get(scout.node_id, {})), \
        "retrieval was not recorded as an activity"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)} invariant tests passed.")


if __name__ == "__main__":
    main()
