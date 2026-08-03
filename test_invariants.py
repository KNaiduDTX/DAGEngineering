"""
test_invariants.py — small, dependency-free checks of the load-bearing guarantees.

Run:  python test_invariants.py
"""

import asyncio
import json
import random
import re
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


def test_terminal_error_is_not_retried():
    """A TerminalExecutionError (e.g. a model refusal / content filter / 4xx) fails the
    node WITHOUT spending the global retry budget — unlike a retryable error, which
    reopens and retries. This is the terminal half of the error taxonomy."""
    from dyndag import TerminalExecutionError

    class _RefusingReviewer(Executor):
        def run(self, inputs, params, ctx):
            raise TerminalExecutionError("model refused (content_filter)")

    reg = default_registry()
    reg[NodeType.REVIEWER] = _RefusingReviewer()
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, reg, budget=Budget(retry_budget=8))
    r = s.run(dict(ONE))
    assert s.governor.retries_used == 0, f"terminal error spent {s.governor.retries_used} retries"
    assert r.reason != TerminationReason.GOAL_SATISFIED, r.reason
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged after a terminal failure"


def _or_completion(content, cost=None, finish="stop", refusal=None):
    """Build an OpenAI-compatible completion dict for the fake transport."""
    msg = {"content": content}
    if refusal is not None:
        msg["refusal"] = refusal
    usage = {"prompt_tokens": 120, "completion_tokens": 60}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}


def _fleet_transport(cost):
    """A fake transport serving valid plan/draft/review for any task at a fixed per-call
    cost (the orchestrator post-fills required subtopics, so plan can return no scouts)."""
    def _t(model, messages, response_format):
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        if name == "plan":
            content = json.dumps({"scouts": [], "engineer_brief": "b", "review_rubric": "r"})
        elif name == "draft":
            content = json.dumps({"covered": ["x"], "sections": [{"subtopic": "x", "text": "t"}]})
        else:  # review
            content = json.dumps({"score": 0.95, "critique": "ok"})
        return _or_completion(content, cost=cost)
    return _t


def test_openrouter_executors_run_and_record_provider_cost():
    """The OpenRouter (OpenAI-compatible) ENGINEER + REVIEWER run end-to-end through an
    injected fake transport (no network, no key), and the dollar cost recorded is the
    provider's own `usage.cost` — not a local tier estimate (per-model pricing gotcha)."""
    from dyndag import openrouter_registry

    def transport(model, messages, response_format):
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        if name == "plan":
            content = json.dumps({"scouts": [{"subtopic": "a", "focus": "f"}],
                                  "engineer_brief": "b", "review_rubric": "r"})
        elif name == "draft":
            content = json.dumps({"covered": ["a"], "sections": [{"subtopic": "a", "text": "real"}]})
        else:  # review
            content = json.dumps({"score": 0.95, "critique": "solid"})
        return _or_completion(content, cost=0.0012)

    reg = openrouter_registry(transport=transport)
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(dict(ONE))
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    eng = next(rec for nid, rec in r.graph.nodes.items() if nid.startswith("engineer"))
    assert eng.cost_usd == 0.0012, eng.cost_usd            # provider-reported, not tier pricing
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_openrouter_refusal_is_terminal():
    """A content-filter refusal from the provider becomes a TerminalExecutionError: the
    node fails and is NOT retried (retrying a refusal just refuses again)."""
    from dyndag import openrouter_registry

    def transport(model, messages, response_format):
        return _or_completion(None, finish="content_filter", refusal="disallowed content")

    reg = openrouter_registry(transport=transport)   # ENGINEER will refuse first
    store = SQLiteEventStore(":memory:")
    s = Scheduler(store, reg, budget=Budget(retry_budget=8))
    r = s.run(dict(ONE))
    assert s.governor.retries_used == 0, f"refusal spent {s.governor.retries_used} retries"
    assert r.reason != TerminationReason.GOAL_SATISFIED, r.reason
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_openrouter_repair_loop_and_cost_extraction():
    """The validate->repair loop re-prompts on invalid JSON then succeeds, and provider
    cost is extracted — exercised directly on the OpenAI-compatible `chat`."""
    from dyndag.executors_openrouter import openrouter_chat
    from dyndag.schemas import REVIEW_SCHEMA

    calls = {"n": 0}

    def transport(model, messages, response_format):
        calls["n"] += 1
        content = "not json at all" if calls["n"] == 1 else json.dumps({"score": 0.9, "critique": "ok"})
        return _or_completion(content, cost=0.0005)

    chat = openrouter_chat(transport=transport, repair_attempts=2)
    out = chat([{"role": "user", "content": "x"}], REVIEW_SCHEMA, "review", ("score", "critique"))
    assert calls["n"] == 2, f"expected one repair, got {calls['n']} calls"
    assert out["content"]["score"] == 0.9
    assert out["usage"]["cost"] == 0.0005


def test_ollama_chat_path_offline():
    """The refactor makes the native-Ollama backend offline-testable too: an injected
    single-call primitive drives the shared repair loop + the shared EngineerRole (no
    `ollama` package, no server). Proves the two providers share one role/repair path."""
    from dyndag import EngineerRole, ollama_chat

    calls = {"n": 0}

    def fake_call(messages, schema, name):
        calls["n"] += 1
        content = "garbage" if calls["n"] == 1 else json.dumps(
            {"covered": ["a"], "sections": [{"subtopic": "a", "text": "t"}]})
        return content, {"input_tokens": 10, "output_tokens": 5}

    chat = ollama_chat(call=fake_call, repair_attempts=2)
    reg = default_registry()
    reg[NodeType.ENGINEER] = EngineerRole(chat)     # mock REVIEWER converges the loop
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(dict(ONE))
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert calls["n"] >= 2, f"repair loop not exercised (calls={calls['n']})"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_openrouter_orchestrator_plans_and_guarantees_coverage():
    """A real ORCHESTRATOR emits a FLAT plan (no recursive patch); the executor
    assembles the subgraph and POST-FILLS required subtopics the model omitted, so the
    goal stays reachable. Here the model plans only 'a' and forgets 'b'; 'b' is added,
    the run reaches the goal, and the plan is a recorded activity."""
    from dyndag import openrouter_registry

    def transport(model, messages, response_format):
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        if name == "plan":                        # model omits required subtopic "b"
            content = json.dumps({"scouts": [{"subtopic": "a", "focus": "deep"}],
                                  "engineer_brief": "cover both", "review_rubric": "be strict"})
        elif name == "draft":
            content = json.dumps({"covered": ["a", "b"], "sections": [{"subtopic": "a", "text": "x"}]})
        else:                                     # review
            content = json.dumps({"score": 0.95, "critique": "ok"})
        return _or_completion(content, cost=0.001)

    reg = openrouter_registry(transport=transport)   # ORCHESTRATOR + ENGINEER + REVIEWER real
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(
        {"topic": "t", "subtopics": ["a", "b"], "target_score": 0.80})
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    scout_keys = {nid.split(".")[0] for nid in r.graph.nodes if nid.startswith("scout")}
    assert scout_keys == {"scout_a", "scout_b"}, scout_keys   # "b" was post-filled
    orch = next(rec for nid, rec in r.graph.nodes.items() if nid.startswith("orchestrator"))
    assert "plan" in r.graph.activities.get(orch.node_id, {}), "plan not recorded as an activity"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_eval_harness_ranks_policies_by_cost():
    """The cost-routing harness runs a task under two policies (fake cloud transports at
    different per-call cost), both succeed, and the report + savings_pct show the cheaper
    policy costs proportionally less — the mechanics of the cost-routing claim, offline."""
    from dyndag import (EvalTask, cost_routing_report, openrouter_registry, run_eval,
                        savings_pct)

    task = EvalTask("t", {"topic": "t", "subtopics": ["a", "b"], "target_score": 0.80})
    policies = {
        "cheap": lambda: openrouter_registry(transport=_fleet_transport(0.001)),
        "dear": lambda: openrouter_registry(transport=_fleet_transport(0.010)),
    }
    results = run_eval([task], policies)
    assert results["cheap"]["t"].success and results["dear"]["t"].success
    assert results["cheap"]["t"].cost_usd < results["dear"]["t"].cost_usd
    assert results["cheap"]["t"].cost_usd > 0                      # provider cost flowed through
    pct = savings_pct(results, "cheap", "dear")
    assert pct is not None and 85.0 <= pct <= 95.0, pct            # 10x cheaper -> ~90%
    rep = cost_routing_report(results)
    assert "cheap" in rep and "less than" in rep, rep


def test_eval_runs_mock_registry_offline():
    """The harness runs any registry: the all-mock fleet reaches the goal at zero
    recorded dollar cost (mocks report no usage)."""
    from dyndag import EvalTask, run_eval

    task = EvalTask("m", {"topic": "t", "subtopics": ["a"], "target_score": 0.80})
    results = run_eval([task], {"mock": lambda: default_registry()})
    tr = results["mock"]["m"]
    assert tr.success and tr.reason == "goal_satisfied", (tr.success, tr.reason)
    assert tr.cost_usd == 0.0, tr.cost_usd


def test_async_executor_runs_and_replays():
    """An `async def run` executor works at the seam: the scheduler awaits it via
    asyncio.gather, records its awaited step via ctx.record_async, and the run replays
    identically. (Phase 2 — literal asyncio, not a thread pool.)"""
    class _AsyncScout(Executor):
        async def run(self, inputs, params, ctx):
            async def fetch():
                await asyncio.sleep(0)                 # a real async model/IO call
                return {"subtopic": params["subtopic"], "facts": []}
            mat = await ctx.record_async("fetch", fetch)
            return ExecutionResult(output=mat, signal=Signal.CONTINUE)

    reg = default_registry()
    reg[NodeType.SCOUT] = _AsyncScout()
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, reg, budget=Budget()).run(dict(GOAL))
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    scout = next(rec for nid, rec in r.graph.nodes.items() if nid.startswith("scout"))
    assert "fetch" in r.graph.activities.get(scout.node_id, {}), "async activity not recorded"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged with an async executor"


def test_async_batch_runs_concurrently():
    """Under max_workers>1 the async batch truly interleaves via asyncio.gather: three
    async scouts are all in-flight at once — a serialized loop (or semaphore=1) could
    not produce that. Single-threaded, so the shared counter needs no lock."""
    state = {"live": 0, "peak": 0}

    class _AsyncScout(Executor):
        async def run(self, inputs, params, ctx):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            await asyncio.sleep(0.02)
            state["live"] -= 1
            return ExecutionResult(output={"subtopic": params["subtopic"], "facts": []},
                                   signal=Signal.CONTINUE)

    reg = default_registry()
    reg[NodeType.SCOUT] = _AsyncScout()
    store = SQLiteEventStore(":memory:")
    goal = {"topic": "t", "subtopics": ["a", "b", "c"], "target_score": 0.80}
    r = Scheduler(store, reg, budget=Budget(), max_workers=3).run(goal)
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert state["peak"] >= 2, f"async batch did not overlap (peak in-flight={state['peak']})"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_workers_produce_same_result_as_batch():
    """Distributed-worker execution (workers PULL the frontier; single-writer commit)
    produces the SAME result as the batch scheduler — identical structure, terminal
    output, and termination — and the log still replays. Only the event interleaving
    differs, and structural_hash is state-based, so results match (Phase 2)."""
    batch = Scheduler(SQLiteEventStore(":memory:"), default_registry(),
                      budget=Budget()).run(dict(GOAL))
    wstore = SQLiteEventStore(":memory:")
    workers = Scheduler(wstore, default_registry(),
                        budget=Budget()).run_workers(dict(GOAL), n_workers=4)
    assert workers.reason == batch.reason == TerminationReason.GOAL_SATISFIED
    assert workers.terminal_output == batch.terminal_output, "worker result differs from batch"
    assert workers.graph.structural_hash() == batch.graph.structural_hash(), "structure differs"
    ok, _, _ = verify_replay(wstore, workers.graph, workers.run_id)
    assert ok, "worker-mode log did not replay"


def test_workers_pull_frontier_concurrently():
    """N workers pull the frontier and execute concurrently: three async scouts are all
    in-flight at once under n_workers=3 — the pull model's real parallelism."""
    state = {"live": 0, "peak": 0}

    class _AsyncScout(Executor):
        async def run(self, inputs, params, ctx):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            await asyncio.sleep(0.02)
            state["live"] -= 1
            return ExecutionResult(output={"subtopic": params["subtopic"], "facts": []},
                                   signal=Signal.CONTINUE)

    reg = default_registry()
    reg[NodeType.SCOUT] = _AsyncScout()
    store = SQLiteEventStore(":memory:")
    goal = {"topic": "t", "subtopics": ["a", "b", "c"], "target_score": 0.80}
    r = Scheduler(store, reg, budget=Budget()).run_workers(goal, n_workers=3)
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert state["peak"] >= 2, f"workers did not overlap (peak in-flight={state['peak']})"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_backpressure_bounds_in_flight():
    """Backpressure: with max_inflight below the worker count, no more than max_inflight
    nodes are admitted-and-running at once — even with 8 ready scouts and 8 idle workers.
    The coordinator paces admission to commit instead of flooding the queue."""
    state = {"live": 0, "peak": 0}

    class _AsyncScout(Executor):
        async def run(self, inputs, params, ctx):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            await asyncio.sleep(0.02)
            state["live"] -= 1
            return ExecutionResult(output={"subtopic": params["subtopic"], "facts": []},
                                   signal=Signal.CONTINUE)

    reg = default_registry()
    reg[NodeType.SCOUT] = _AsyncScout()
    store = SQLiteEventStore(":memory:")
    goal = {"topic": "t", "subtopics": [f"s{i}" for i in range(8)], "target_score": 0.80}
    r = Scheduler(store, reg, budget=Budget()).run_workers(goal, n_workers=8, max_inflight=3)
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert state["peak"] <= 3, f"backpressure exceeded: peak in-flight={state['peak']} > 3"
    assert state["peak"] >= 2, f"no concurrency under the cap (peak={state['peak']})"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


# --------------------------------------------------------------------------- #
# GSM8K harness — single-pass vs SSA-generational, offline with a fake `chat`   #
# --------------------------------------------------------------------------- #
def _arith(q):
    m = re.search(r"(\d+)\s+(plus|minus|times)\s+(\d+)", q)
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return {"plus": a + b, "minus": a - b, "times": a * b}[op]


def _gsm_chat(first_wrong=False):
    """A fake `chat` (the ChatJson seam): solves 'Compute A op B.' problems. With
    first_wrong, the first pass (no critique yet) returns a wrong answer and a refinement
    returns the right one; the checker verifies against its OWN recomputation (never gold)."""
    def chat(messages, schema, name, required):
        payload = json.loads(messages[-1]["content"])
        if name == "solution":
            correct = _arith(payload["problem"])
            wrong = first_wrong and payload.get("critique_to_address") is None
            ans = str(correct + 1000) if wrong else str(correct)
            return {"content": {"reasoning": "work", "final_answer": ans},
                    "usage": {"input_tokens": 10, "output_tokens": 5, "cost": 0.001}}
        fa = str(payload["final_answer"])                       # verify
        ok = fa == str(_arith(payload["problem"]))
        return {"content": {"verified": ok, "confidence": 1.0 if ok else 0.1,
                            "critique": "" if ok else "recompute"},
                "usage": {"input_tokens": 8, "output_tokens": 3, "cost": 0.002}}
    return chat


def test_gsm8k_goal_predicate_injectable():
    """A GSM8K goal (no subtopics/target_score) reaches GOAL_SATISFIED via an injected
    AnswerGoal — proving the scheduler is no longer welded to CoverageGoal. Replay holds."""
    from dyndag import gsm8k_registry, AnswerGoal
    store = SQLiteEventStore(":memory:")
    r = Scheduler(store, gsm8k_registry(_gsm_chat(), mode="single"), budget=Budget(),
                  goal_predicate=AnswerGoal()).run({"question": "Compute 9 plus 9.", "problem_id": 0})
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    assert r.terminal_output["final_answer"] == "18", r.terminal_output
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok


def test_gsm8k_ssa_helps_vs_singlepass():
    """The paper's core comparison, offline: a solver wrong on the first pass, correct after a
    critique. Single-pass takes the wrong answer; SSA-generational iterates a NEW generation
    and gets it right — accuracy 0/6 -> 6/6, generations 1 -> 2, replay holds throughout."""
    from dyndag import gsm8k_registry, sample_tasks, run_eval, gsm8k_report
    tasks = sample_tasks()
    results = run_eval(tasks, {
        "single-pass": lambda: gsm8k_registry(_gsm_chat(first_wrong=True), mode="single"),
        "ssa":         lambda: gsm8k_registry(_gsm_chat(first_wrong=True), mode="ssa"),
    })
    sp = [results["single-pass"][t.name] for t in tasks]
    ss = [results["ssa"][t.name] for t in tasks]
    assert sum(t.success for t in sp) == 0, "single-pass should keep the wrong first answer"
    assert sum(t.success for t in ss) == len(tasks), "SSA should refine to the correct answer"
    assert all(t.generations == 1 for t in sp), [t.generations for t in sp]
    assert all(t.generations >= 2 for t in ss), [t.generations for t in ss]
    assert all(t.replay_ok for t in sp + ss), "replay-equivalence must hold during the eval"
    rep = gsm8k_report(results)
    assert "GSM8K EVAL" in rep and "$/correct" in rep, rep


def test_gsm8k_eval_metrics_surface():
    """Every paper metric surfaces on TaskResult / the report: accuracy, cost, $/correct,
    generations, wall-clock, replay — over a perfect fake solver (6/6)."""
    from dyndag import gsm8k_registry, sample_tasks, run_eval, accuracy, cost_per_correct
    tasks = sample_tasks()
    results = run_eval(tasks, {"ssa": lambda: gsm8k_registry(_gsm_chat(), mode="ssa")})
    assert accuracy(results, "ssa") == (6, 6), accuracy(results, "ssa")
    cpc = cost_per_correct(results, "ssa")
    assert cpc is not None and cpc > 0, cpc
    for t in results["ssa"].values():
        assert t.generations >= 1 and t.wall_s >= 0.0 and t.replay_ok, t


def test_gsm8k_judge_normalizes_answer():
    """The exact-match judge normalizes formatting: '18', ' $18 ', '18.0' all match gold
    '18'; '20' and a missing terminal do not — reading the structured final_answer, no regex."""
    from dyndag.tasks_gsm8k import _make_judge
    judge = _make_judge("18")

    def _r(ans):
        return type("R", (), {"terminal_output": {"kind": "solution", "final_answer": ans}})()
    assert judge(_r("18")) and judge(_r(" $18 ")) and judge(_r("18.0"))
    assert not judge(_r("20"))
    assert not judge(type("R", (), {"terminal_output": None})())


# --------------------------------------------------------------------------- #
# φ-nodes — merging competing branches (self-consistency, best-of-N)           #
# --------------------------------------------------------------------------- #
def _sc_chat(answers):
    """Fake `chat`: solvers return a preset sequence (one per sample, in id-sorted execution
    order); the checker, if reached, always approves. Drives self-consistency deterministically
    without a model — the diversity a real run gets from temperature is pre-canned here."""
    box = {"i": 0}
    def chat(messages, schema, name, required):
        if name == "solution":
            a = answers[box["i"] % len(answers)]
            box["i"] += 1
            return {"content": {"reasoning": "r", "final_answer": str(a)},
                    "usage": {"input_tokens": 4, "output_tokens": 2, "cost": 0.001}}
        return {"content": {"verified": True, "confidence": 1.0, "critique": ""},
                "usage": {"input_tokens": 2, "output_tokens": 1, "cost": 0.001}}
    return chat


def test_phi_argmax_selects_top_score():
    """Best-of-N: `argmax_score` picks the highest-scored competing candidate and records which
    branch won (the φ selector, for lineage)."""
    from dyndag import argmax_score
    cands = [("a", {"final_answer": "1"}, 0.3), ("b", {"final_answer": "2"}, 0.9),
             ("c", {"final_answer": "3"}, 0.5)]
    out, basis = argmax_score(cands)
    assert out["final_answer"] == "2" and basis["selected_from"] == "b", (out, basis)


def test_phi_majority_vote_self_consistency():
    """Self-consistency: `majority_vote` picks the modal answer across competing samples and
    records the tally; a lone dissenter is outvoted."""
    from dyndag import majority_vote
    cands = [(f"s{i}", {"final_answer": a}, None)
             for i, a in enumerate(["18", "18", "18", "20", "7"])]
    out, basis = majority_vote("final_answer")(cands)
    assert out["final_answer"] == "18", out
    assert basis["votes"] == {"18": 3, "20": 1, "7": 1} and basis["winner"] == "18", basis


def test_phi_merge_replays_identically():
    """A graph containing a φ-merge replays to an identical projection (the merge is a pure,
    id-sorted, deterministic fold) — INV-5.2 holds with competing branches + a φ-node."""
    from dyndag import gsm8k_registry, AnswerGoal
    store = SQLiteEventStore(":memory:")
    reg = gsm8k_registry(_sc_chat(["7", "7", "9"]), mode="sc", k=3)
    r = Scheduler(store, reg, budget=Budget(), goal_predicate=AnswerGoal()).run(
        {"question": "Compute 3 plus 4.", "problem_id": 0})
    assert r.reason == TerminationReason.GOAL_SATISFIED, r.reason
    phi = r.terminal_output.get("phi")
    assert phi and phi["policy"] == "majority_vote", r.terminal_output
    assert r.terminal_output["final_answer"] == "7", r.terminal_output
    assert any(rec.node_type == NodeType.PHI for rec in r.graph.nodes.values()), "no PHI node"
    ok, _, _ = verify_replay(store, r.graph, r.run_id)
    assert ok, "replay diverged with a φ-merge"


def test_phi_fail_closed_on_empty_candidates():
    """A φ-merge with no candidate inputs is a failed attempt (MalformedResultError), never a
    silent pass — fail-closed like every other executor (INV-4.2)."""
    from dyndag import PhiExecutor, majority_vote, Context, MalformedResultError
    phi = PhiExecutor(majority_vote("final_answer"))
    ctx = Context(run_id="r", goal={}, self_id="phi.g1", generation=1)
    try:
        phi.run({}, {}, ctx)
        assert False, "empty φ-merge should fail closed"
    except MalformedResultError:
        pass


def test_gsm8k_self_consistency_beats_singlepass():
    """The third eval arm on GSM8K: one greedy solver is wrong, but the MAJORITY of K sampled
    solvers is right — single-pass fails, self-consistency (φ majority-vote) succeeds."""
    from dyndag import gsm8k_registry, EvalTask, run_eval, AnswerGoal
    from dyndag.tasks_gsm8k import _make_judge
    task = EvalTask("q", {"question": "Compute 3 plus 4.", "problem_id": 0},
                    judge=_make_judge("7"), goal_predicate=AnswerGoal())
    results = run_eval([task], {
        "single": lambda: gsm8k_registry(_sc_chat(["20"]), mode="single"),
        "sc":     lambda: gsm8k_registry(_sc_chat(["20", "7", "7", "7", "7"]), mode="sc", k=5),
    })
    assert results["single"]["q"].success is False, "single-pass should take the wrong answer"
    assert results["sc"]["q"].success is True, "self-consistency should outvote the wrong sample"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)} invariant tests passed.")


if __name__ == "__main__":
    main()
