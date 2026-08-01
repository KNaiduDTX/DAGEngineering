"""
scheduler.py — the orchestration engine (the core loop).

This is where every layer meets. One tick =

    derive ready set (Kahn frontier)                         [Layer 1]
      -> route model + charge cost, emit NODE_STARTED        [Layer 5]
      -> resolve inputs + placeholders, run executor         [Layers 1,2]
      -> emit NODE_COMPLETED (single-assignment)             [Layers 1,6]
      -> constraint-filter + apply any emitted subgraph      [Layers 2,3]
      -> on GOAL signal, test goal predicate                 [Layer 3]
    then advance a generation, bounded by the governor        [Layer 7]

Because an expander's patch adds nodes one generation above their parents, the
graph unrolls on the fly while staying acyclic, and the loop terminates by one
of exactly four reasons (goal / generation cap / budget / no-progress).

Determinism note: the PoC executes each ready batch in a defined order. In
production the batch is the concurrency unit (asyncio.gather / a worker pool);
nothing else in the loop changes, because ordering within a batch is — by the
definition of the ready set — semantically irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core import (Event, EventType, NodeType, Provenance, Signal,
                   TerminationReason, new_batch_id, new_run_id, now_iso,
                   stable_edge_id, stable_node_id)
from .executors import (Context, ExecutionResult, Executor, MalformedResultError,
                        ModelRouter)
from .governor import Budget, CostGovernor
from .graph import GraphProjection
from .nodes import NodeSpec, SubgraphPatch, resolve_placeholders
from .steering import ConstraintFilter, CoverageGoal
from .store import EventStore


@dataclass
class RunResult:
    run_id: str
    reason: TerminationReason
    terminal_output: Optional[dict]
    ticks: int
    graph: GraphProjection
    governor: CostGovernor


@dataclass
class _Outcome:
    """Carries one node's execution from the (parallelizable) execute phase to the
    (serial, single-writer) commit phase. Holding the result as plain data is what
    lets the execute phase avoid touching shared state, so Phase 2 can run the whole
    batch under asyncio.gather while admit/commit stay serial (see Scheduler.run)."""
    node_id: str
    result: Optional[ExecutionResult] = None
    error: Optional[Exception] = None


class Scheduler:
    def __init__(
        self,
        store: EventStore,
        registry: dict[NodeType, Executor],
        router: Optional[ModelRouter] = None,
        budget: Optional[Budget] = None,
        constraint_filter: Optional[ConstraintFilter] = None,
        beam_width: int = 0,                      # 0 => no beam pruning
    ):
        self.store = store
        self.registry = registry
        self.router = router or ModelRouter()
        self.governor = CostGovernor(budget or Budget())
        self.filter = constraint_filter or ConstraintFilter()
        self.beam_width = beam_width
        self.graph = GraphProjection()

    # --------------------------------------------------------------------- #
    # event emission (every mutation goes through here -> the log)            #
    # --------------------------------------------------------------------- #
    def _emit(self, event: Event) -> Event:
        stored = self.store.append(event)   # provenance guard lives in the store
        self.graph.apply(stored)            # keep the live projection in lockstep
        return stored

    def _provenance(self, run_id: str, generation: int, parents: tuple[str, ...],
                    expander_id: Optional[str]) -> Provenance:
        return Provenance(
            run_id=run_id,
            batch_id=new_batch_id(run_id, generation),
            generation=generation,
            created_at=now_iso(),
            expander_id=expander_id,
            parent_ids=parents,
        ).validate()

    def _create_node(self, run_id: str, key: str, node_type: NodeType,
                     generation: int, parents: tuple[str, ...], params: dict,
                     expander_id: Optional[str]) -> str:
        node_id = stable_node_id(key, generation, parents)
        if node_id in self.graph.nodes:
            raise ValueError(f"id collision: {node_id} already exists")
        prov = self._provenance(run_id, generation, parents, expander_id)
        self._emit(Event(
            run_id=run_id, type=EventType.NODE_CREATED, node_id=node_id,
            payload={
                "node_type": node_type.value,
                "input_ids": list(parents),
                "params": params,
                "provenance": prov.as_dict(),
            },
        ))
        for p in parents:
            self._create_edge(run_id, p, node_id, generation, expander_id)
        return node_id

    def _create_edge(self, run_id: str, producer: str, consumer: str,
                     generation: int, expander_id: Optional[str]) -> None:
        prov = self._provenance(run_id, generation, (producer,), expander_id)
        self._emit(Event(
            run_id=run_id, type=EventType.EDGE_CREATED,
            node_id=stable_edge_id(producer, consumer),
            payload={"producer": producer, "consumer": consumer,
                     "provenance": prov.as_dict()},
        ))

    # --------------------------------------------------------------------- #
    # subgraph emission (Layer 2, primitive 4) with constraint filtering      #
    # --------------------------------------------------------------------- #
    def _apply_patch(self, run_id: str, expander_id: str, patch: SubgraphPatch) -> list[str]:
        specs, pruned = self.filter.apply(patch.added_nodes)
        for spec, reason in pruned:
            self._emit(Event(run_id=run_id, type=EventType.NODE_PRUNED,
                             payload={"key": spec.key, "reason": reason,
                                      "expander_id": expander_id}))
        if self.governor.would_exceed_nodes(self.graph, len(specs)):
            return []  # governor refuses the expansion; run will end on a bound

        # Resolve logical keys -> concrete ids in dependency order. A patch is a
        # small DAG; nodes whose inputs are all resolved (existing ids or already
        # -placed siblings) are created first.
        resolved: dict[str, str] = {}
        created: list[str] = []
        pending = list(specs)
        while pending:
            progressed = False
            for spec in list(pending):
                parents: list[str] = []
                ok = True
                for ik in spec.input_keys:
                    if ik in resolved:               # sibling placed earlier in this patch
                        parents.append(resolved[ik])
                    elif ik in self.graph.nodes:      # pre-existing node (concrete id)
                        parents.append(ik)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                gen = self.graph.next_generation_for(tuple(parents))
                nid = self._create_node(run_id, spec.key, spec.node_type, gen,
                                        tuple(parents), spec.params, expander_id)
                resolved[spec.key] = nid
                created.append(nid)
                pending.remove(spec)
                progressed = True
            if not progressed:
                raise ValueError(f"unresolvable patch from {expander_id}: "
                                 f"{[s.key for s in pending]}")
        self._emit(Event(run_id=run_id, type=EventType.SUBGRAPH_EMITTED,
                         node_id=expander_id,
                         payload={"added_nodes": created}))
        return created

    # --------------------------------------------------------------------- #
    # one node, in three phases: admit -> execute -> commit                   #
    #                                                                         #
    # The batch (the ready set) is dataflow-independent but NOT side-effect-  #
    # independent: admit and commit both touch shared state (governor, graph, #
    # store). So they run serially and are the single-writer serialization    #
    # point; only execute is pure and parallelizable. Phase 2 swaps the       #
    # execute loop for asyncio.gather and nothing else changes.               #
    # --------------------------------------------------------------------- #
    def _admit(self, run_id: str, node_id: str) -> None:
        """Serial, single-writer: route the node, charge its cost, record NODE_STARTED.
        Admission is where cost is spent, so it is where the cost cap is enforced
        (the caller checks the governor before admitting each node)."""
        rec = self.graph.nodes[node_id]
        binding = self.router.route(rec.node_type)
        self.governor.charge(binding.cost)
        self._emit(Event(run_id=run_id, type=EventType.NODE_STARTED, node_id=node_id,
                         payload={"tier": binding.tier.value, "model": binding.name,
                                  "cost": binding.cost}))

    def _execute(self, run_id: str, node_id: str, goal: dict) -> _Outcome:
        """Pure execution: read inputs, run the bound executor, validate the result.
        Touches NO shared state (no log append, no projection mutation, no governor),
        so a whole batch of _execute calls is safe to run concurrently. Returns the
        result (or the failure) as data for the serial commit phase."""
        rec = self.graph.nodes[node_id]
        inputs = {p: self.graph.nodes[p].output for p in self.graph.in_edges.get(node_id, set())}
        params = resolve_placeholders(rec.params, inputs)
        ctx = Context(run_id=run_id, goal=goal, self_id=node_id, generation=rec.generation)
        executor = self.registry[rec.node_type]
        try:
            result = executor.run(inputs, params, ctx)
            self._validate_result(result)   # fail-closed: never act on unvalidated output
        except Exception as exc:            # executor raised OR returned garbage -> one path
            return _Outcome(node_id=node_id, error=exc)
        return _Outcome(node_id=node_id, result=result)

    def _commit(self, run_id: str, outcome: _Outcome, predicate) -> Optional[dict]:
        """Serial, single-writer: fold one execution outcome into the log and graph.
        A failure becomes NODE_FAILED (+ retry/reopen); a success becomes
        NODE_COMPLETED (+ any emitted subgraph). Committing in a fixed order keeps
        the batch's event sequence deterministic and replayable. Returns the terminal
        output iff this node raised GOAL and the goal predicate accepts it."""
        node_id = outcome.node_id
        rec = self.graph.nodes[node_id]
        if outcome.error is not None:
            return self._fail(run_id, node_id, rec, outcome.error)

        result = outcome.result
        self._emit(Event(run_id=run_id, type=EventType.NODE_COMPLETED, node_id=node_id,
                         payload={"output": result.output, "score": result.score,
                                  "signal": result.signal.value}))
        if result.patch is not None:
            self._apply_patch(run_id, node_id, result.patch)

        if result.signal == Signal.GOAL and predicate.satisfied(result.output):
            self._emit(Event(run_id=run_id, type=EventType.GOAL_SATISFIED,
                             node_id=node_id, payload={"output": result.output}))
            return result.output   # accepted terminal (INV-3.1: predicate gated)
        return None

    def _fail(self, run_id: str, node_id: str, rec, exc: Exception) -> None:
        """Record a failed attempt and, if the GLOBAL retry budget allows, reopen
        the node for one more dispatch. Both a raised executor and a malformed
        result land here, so the scheduler has exactly one failure path."""
        self._emit(Event(run_id=run_id, type=EventType.NODE_FAILED, node_id=node_id,
                         payload={"error": repr(exc), "attempt": rec.attempts}))
        if self.governor.spend_retry():
            self._reopen(run_id, node_id)
        return None

    def _validate_result(self, result) -> None:
        """Fail-closed check on executor output BEFORE it can touch the log or the
        graph. The completion signal is what the scheduler branches on (INV-4.1),
        so its shape is load-bearing: a real model that returns the wrong type, an
        untyped signal, or a bad patch is handled as a failed attempt, never
        folded in. This is where a JSON-mode / schema-validate / repair loop
        attaches when real executors arrive (SPEC §9)."""
        if not isinstance(result, ExecutionResult):
            raise MalformedResultError(
                f"executor returned {type(result).__name__}, not ExecutionResult")
        if not isinstance(result.signal, Signal):
            raise MalformedResultError(f"signal {result.signal!r} is not a typed Signal")
        if result.score is not None and not isinstance(result.score, (int, float)):
            raise MalformedResultError(f"score {result.score!r} is neither numeric nor None")
        if result.patch is not None:
            self._validate_patch(result.patch)

    def _validate_patch(self, patch) -> None:
        """A patch is the sole construction primitive; a malformed one would strand
        or corrupt the graph, so it is validated structurally before instantiation."""
        if not isinstance(patch, SubgraphPatch):
            raise MalformedResultError(f"patch is {type(patch).__name__}, not SubgraphPatch")
        for spec in patch.added_nodes:
            if not isinstance(spec, NodeSpec):
                raise MalformedResultError(f"patch node is {type(spec).__name__}, not NodeSpec")
            if not isinstance(spec.node_type, NodeType):
                raise MalformedResultError(
                    f"node {spec.key!r} has non-typed node_type {spec.node_type!r}")
            if spec.node_type not in self.registry:
                raise MalformedResultError(
                    f"node {spec.key!r} type {spec.node_type.value} has no bound executor")
        for edge in patch.added_edges:
            if not (isinstance(edge, (tuple, list)) and len(edge) == 2):
                raise MalformedResultError(f"malformed edge {edge!r}")

    def _reopen(self, run_id: str, node_id: str) -> None:
        """Return a FAILED node to the frontier for another attempt — as a RECORDED
        fact (NODE_REOPENED), never an out-of-band mutation. The projection is a
        pure fold over the log (INV-0); if the reopen weren't itself an event, a
        replay would reconstruct the node as FAILED and diverge from the live graph
        whenever the run ends before the retry runs (INV-5.2). The fold resets the
        state to CREATED; the retry then emits a fresh NODE_STARTED."""
        self._emit(Event(run_id=run_id, type=EventType.NODE_REOPENED, node_id=node_id,
                         payload={"attempt": self.graph.nodes[node_id].attempts}))

    # --------------------------------------------------------------------- #
    # the run                                                                 #
    # --------------------------------------------------------------------- #
    def run(self, goal: dict) -> RunResult:
        run_id = new_run_id()
        self._emit(Event(run_id=run_id, type=EventType.RUN_STARTED,
                         payload={"goal": goal}))

        predicate = CoverageGoal(subtopics=goal["subtopics"],
                                 target_score=goal["target_score"])

        # Seed: a GOAL source node (gen 0) and the root ORCHESTRATOR (gen 1).
        goal_id = self._create_node(run_id, "goal", NodeType.GOAL, 0, (),
                                    {"goal": goal}, expander_id=None)
        self._create_node(run_id, "orchestrator", NodeType.ORCHESTRATOR, 1,
                          (goal_id,), {}, expander_id=None)

        terminal_output: Optional[dict] = None
        reason: Optional[TerminationReason] = None
        generation = 0

        while True:
            reason = self.governor.check(self.graph, generation)
            if reason is not None:
                break

            frontier = self.graph.ready_set()
            if not frontier:
                reason = TerminationReason.NO_PROGRESS
                break

            if self.beam_width:
                from .steering import beam_prune
                frontier, _dropped = beam_prune(self.graph, frontier, self.beam_width)

            # ADMIT (serial, single-writer). Cost is spent here, so the cap is
            # enforced here as it accrues: a wide batch cannot overshoot
            # max_total_cost by more than one node before admission stops (C).
            admitted: list[str] = []
            for node_id in frontier:                       # batch = parallel unit
                over = self.governor.check(self.graph, generation)
                if over is not None:
                    reason = over
                    break
                self._admit(run_id, node_id)
                admitted.append(node_id)

            # EXECUTE (pure). This is the seam Phase 2 runs under asyncio.gather;
            # serially for now, but nothing else in the loop needs to change (D).
            outcomes = [self._execute(run_id, node_id, goal) for node_id in admitted]

            # COMMIT (serial, single-writer, deterministic order). A satisfied goal
            # takes precedence over a cost bound tripped during admission.
            for outcome in outcomes:
                candidate = self._commit(run_id, outcome, predicate)
                if candidate is not None:
                    terminal_output = candidate
                    reason = TerminationReason.GOAL_SATISFIED
                    break

            if reason is not None:
                break

            generation += 1
            if generation > self.governor.budget.max_generation:
                reason = TerminationReason.GENERATION_CAP
                break

        self._emit(Event(run_id=run_id, type=EventType.RUN_TERMINATED,
                         payload={"reason": reason.value}))
        return RunResult(run_id=run_id, reason=reason, terminal_output=terminal_output,
                         ticks=generation, graph=self.graph, governor=self.governor)
