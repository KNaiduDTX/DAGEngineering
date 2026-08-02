# Dynamic-DAG Multi-Agent Orchestrator — Governing Specification (v1)

This document formalizes the seven approved layers into contracts, schemas, and
algorithms. The `dyndag/` package is its reference implementation; every clause
below points at the module that discharges it. Where the PoC narrows a
production choice (SQLite for Postgres, deterministic executors for models,
serial batches for concurrency), the narrowing is stated and §9 gives the
production form.

**Approved decisions this spec is bound to**

| Layer | Decision |
|---|---|
| 1 | Formal object as specified: typed SSA nodes, typed data-dependency edges with provenance, dataflow firing. |
| 2 | **Subgraph emission (highest-order primitive) is the sole construction mechanism.** |
| 3 | Goal-conditioned construction: chaining, value-guided search, critic/verifier nodes, constraint propagation, aggregation. |
| 4 | Agent = executor bound to a node type; complex agent = nested subgraph; **blackboard hybrid** coordination. |
| 5 | Event-sourced Postgres graph-of-record + Neo4j projection; topological scheduler; per-node cost routing. |
| 6 | Acyclicity via generation monotonicity + single-assignment. |
| 7 | Bounded governance: caps, global retry budget, four deterministic termination reasons. |

---

## §0 The central invariant — the DAG is a *trace*, not the program

The system never draws a graph and runs it. The graph **precipitates** as
computation proceeds. Iteration is preserved by *unrolling into fresh
generations*, not by cycles: a refinement of node *x* is a new node *x′* at a
higher generation with an edge *x → x′*. The materialized artifact is therefore
always a DAG, while behavior is fully recurrent.

> **INV-0.** The only mutable object in the system is the append-only event log.
> The graph, the frontier, and the blackboard are deterministic folds over it.
> Consequence: replaying the log reconstructs the system exactly (see INV-5.2).

Module: `store.py` (the log), `graph.py` (`GraphProjection.replay`), everything
else consumes projections.

---

## §1 Formal object (Layer 1)

The system is a sequence of graphs `G₀, G₁, …`, each a fold prefix of the log.
A single construction step is the Graph-of-Thoughts mutation

```
G' = T(G) ,  V' = (V ∪ V⁺) \ V⁻ ,  E' = (E ∪ E⁺) \ E⁻
```

In this system `V⁻ = E⁻ = ∅` (single-assignment: nodes are superseded, never
removed), so `T` is monotone-additive. Choosing `T` well is the whole problem;
§2–§3 specify it.

### 1.1 Node

A node is a **typed, single-assignment variable binding**.

| Field | Meaning |
|---|---|
| `node_id` | stable id `⟨key⟩.g⟨generation⟩`, unique by construction |
| `node_type` | its function in the dataflow graph (§4.2) |
| `generation` | integer level; `= max(parent.generation)+1` |
| `input_ids` | the producers this node reads (its in-edges) |
| `params` | static parameters, may contain `$producer` placeholders |
| `state` | `CREATED → STARTED → COMPLETED \| FAILED` (READY is derived) |
| `output` | written **exactly once**, on `COMPLETED` |
| `provenance` | the tuple of §1.3, stamped at creation |

Module: `graph.NodeRecord`, `core.NodeType`, `core.NodeState`.

### 1.2 Edge

An edge `A → B` asserts two things at once: **dataflow** (B reads A's output)
and **lineage** (B was created because of A). Both are kept. Edges carry
provenance identical in shape to nodes.

Module: `graph.GraphProjection.{in_edges,out_edges}`, `scheduler._create_edge`.

### 1.3 Provenance — provenance-at-write

```
Provenance = (run_id, batch_id, generation, created_at, expander_id, parent_ids)
```

> **INV-1.1.** No `NODE_CREATED`/`EDGE_CREATED` event is accepted unless its
> provenance validates (all of `run_id`, `batch_id`, `generation`, `created_at`
> present). Creation *is* the stamped event; there is no code path that mints a
> node/edge without lineage.

This is the structural elimination of NULL provenance (`started_at`,
`source_batch_id`, `pipeline_run_id` cannot be null because the write that
creates the row carries them). Sentinel markers survive only at a one-time
migration boundary and are never newly minted.

Module: `core.Provenance.validate`, enforced in `store._assert_provenance`.

### 1.4 Firing semantics (dataflow / Kahn)

A node is **READY** iff every producer is `COMPLETED`; it fires as soon as it is
ready. The set of ready nodes at a tick is the **parallelizable batch** — its
members are mutually independent by definition.

Module: `graph.GraphProjection.ready_set`.

### 1.5 Event schema (the log of record)

| `EventType` | payload keys | role |
|---|---|---|
| `RUN_STARTED` | `goal` | audit |
| `NODE_CREATED` | `node_type, input_ids, params, provenance` | fact + guard |
| `EDGE_CREATED` | `producer, consumer, provenance` | fact + guard |
| `NODE_STARTED` | `tier, model, cost` | dispatch + cost, stamps `started_at` |
| `NODE_COMPLETED` | `output, score, signal` | single-assignment write |
| `NODE_FAILED` | `error, attempt` | retry accounting |
| `NODE_REOPENED` | `attempt` | returns a FAILED node to the frontier as a *logged fact*, not a state mutation |
| `ACTIVITY_RECORDED` | `key, result` | a nondeterministic sub-step's result, recorded for replay / exactly-once |
| `SUBGRAPH_EMITTED` | `added_nodes` | lineage of construction |
| `NODE_PRUNED` | `key, reason, expander_id` | constraint-filter audit |
| `GOAL_SATISFIED` | `output` | terminal acceptance |
| `RUN_TERMINATED` | `reason` | one of four reasons (§7) |

Only **facts** are logged; derived states (READY) are recomputed, never stored.

---

## §2 Construction — subgraph emission (Layer 2, chosen primitive)

The sole construction mechanism is **an executor returning a `SubgraphPatch`**.
No static fan-out, no conditional-edge DSL — an expander computes the next
stratum as data and the scheduler applies it. This is "propagation of nodes and
edges on the fly, computed as downstream variables of prior nodes."

### 2.1 Patch contract

```
SubgraphPatch = (added_nodes : [NodeSpec], added_edges : [(key, key)])
NodeSpec      = (key, node_type, input_keys, params)
```

`input_keys` reference either a **concrete existing node id** (e.g. the expander
referencing itself via `ctx.self_id`, or a prior draft) or a **logical key of a
sibling** in the same patch. Edges are the union of `added_edges` and one edge
per `input_key`.

Module: `nodes.SubgraphPatch`, `nodes.NodeSpec`.

### 2.2 Apply algorithm (`scheduler._apply_patch`)

1. Run the **constraint filter** (§3.3); pruned specs emit `NODE_PRUNED`.
2. If the governor would exceed `max_nodes`, refuse the whole patch (run ends on
   a bound).
3. Resolve logical keys → concrete ids in **dependency order** (a patch is a
   small DAG): a spec is placeable when every `input_key` is a concrete id or an
   already-placed sibling. For each placeable spec:
   * `generation = max(parent.generation)+1`  ← the monotonicity rule (§6)
   * `node_id = ⟨key⟩.g⟨generation⟩`, asserted unique
   * emit `NODE_CREATED` (+ `EDGE_CREATED` per parent), all with provenance whose
     `expander_id` is the emitting node.
4. Emit one `SUBGRAPH_EMITTED` marker naming the created ids.

> **INV-2.1.** Every node a patch creates sits strictly above all its parents in
> generation. Hence every emitted edge is monotonic (⇒ acyclic, §6).

### 2.3 Placeholder substitution (primitive 1 under primitive 4)

A node's `params` may contain `$producer_id` references, resolved against
completed producers' outputs at dispatch (`nodes.resolve_placeholders`). A bare
`"$id"` yields the raw object; inline refs stringify. This is the LLMCompiler
`$1/$2` mechanism scoped to one node's parameter dict.

---

## §3 Steering toward a specific outcome (Layer 3)

Reactive construction is turned goal-seeking by four devices.

### 3.1 Chaining & the critic-controller loop

The reviewer is the **controller** in a setpoint/error loop: the target score is
the setpoint, the draft score the measurement, `error = target − score`.

* `error > 0` → do **not** loop; **emit a refinement stratum**: a new engineer
  version (SSA, higher generation) fed the failed draft + the critique, plus a
  fresh reviewer to re-measure.
* `error ≤ 0` → approve; **emit the aggregator** that converges the graph.

Bounded convergence is guaranteed by the generation cap (§7). Module:
`executors.ReviewerExecutor`.

### 3.2 Goal predicate (terminal acceptance)

A `GOAL`-signalled node is accepted as terminal **only if** the predicate holds
on its output; otherwise the search continues. The demo predicate:
`covered ⊇ required_subtopics ∧ final_score ≥ target`.

> **INV-3.1.** Termination with reason `GOAL_SATISFIED` implies the goal
> predicate returned true. A premature `GOAL` signal cannot end the run.

Module: `steering.CoverageGoal`, checked in `scheduler.run`.

### 3.3 Constraint propagation (arc-consistency in miniature)

`ConstraintFilter` runs on every patch **before instantiation**, dropping specs
that cannot advance the goal or would breach a budget. Killing branches at
construction time is cheaper than generate-then-verify. The target also flows
**downward as params** (the reviewer receives `target_score` from the
orchestrator) — constraint propagation toward the point of use.

Module: `steering.ConstraintFilter`.

### 3.4 Value-guided pruning & aggregation

`beam_prune` keeps the top-`width` frontier candidates by `value_function`
(critic scores dominate), making wide speculative expansion affordable.
Aggregation (`SynthesizerExecutor`, in-degree > 1) merges convergent branches to
a single terminal output.

Modules: `steering.beam_prune`, `executors.SynthesizerExecutor`.

---

## §4 Multi-agent binding (Layer 4)

### 4.1 Executor contract

```
Executor.run(inputs: {producer_id: output}, params: dict, ctx: Context)
        -> ExecutionResult(output, signal, score?, patch?)
```

`signal` is a **typed** value (`CONTINUE/DONE/NEEDS_REFINEMENT/GOAL`), never a
token parsed from `output`.

> **INV-4.1.** Node completion is a typed `ExecutionResult.signal`, and the
> scheduler branches only on it. The 'agent emitted `<complete>` as a literal
> string' failure class is unrepresentable.
> **INV-4.2 (fail-closed).** A result that is not a well-formed `ExecutionResult`
> — wrong type, an untyped `signal`, a non-numeric `score`, or a malformed patch —
> is rejected *at the dispatch boundary* and handled as a failed attempt; the
> scheduler never folds unvalidated output into the log or graph. This is the seam
> where a real model's structured-output validation + repair loop attaches (§9).

Module: `executors.Executor`, `executors.ExecutionResult`,
`executors.MalformedResultError`, enforced in `scheduler._validate_result`.

### 4.2 Roles are types, not personalities

| Role (`NodeType`) | Function in the graph |
|---|---|
| `GOAL` | terminal input source holding the target spec |
| `ORCHESTRATOR` | the expander — emits subgraphs (§2) |
| `SCOUT` | frontier retrieval |
| `ENGINEER` | generation / refinement |
| `REVIEWER` | critic — the error signal (§3.1) |
| `SYNTHESIZER` | aggregator — convergence |
| `TESTER/HARDENER/OPS/MAINTAINER/DEBUGGER` | critic/terminal variants, same contract |

A **complex agent is a nested subgraph**: its executor returns a `SubgraphPatch`
that expands into its own sub-DAG rather than a scalar value.

### 4.3 Blackboard mapping (the chosen hybrid)

The DAG **is** the board. `views.render_blackboard` projects it:

| Blackboard concept | Graph concept |
|---|---|
| card | node |
| column | node state |
| dependency arrow | edge |
| WIP limit | max concurrent `STARTED` (the scheduler's concurrency cap) |
| "done" | the typed `NODE_COMPLETED` event |
| shared memory | producer outputs read through in-edges (the ByteRover-style store) |

Construction stays single-writer (the scheduler applies patches), which keeps
the board auditable; agents contribute by emitting patches, not by mutating
shared state directly.

---

## §5 Implementation (Layer 5)

### 5.1 Storage — Postgres of record + Neo4j projection

| PoC | Production |
|---|---|
| `SQLiteEventStore` (`:memory:`) | Postgres `events(seq BIGSERIAL, run_id, type, node_id, payload JSONB, created_at)` |
| `GraphProjection` (in-memory fold) | Neo4j projection updated by folding the same events; the **write path stays in Postgres** |

The store is an adapter (`EventStore`); swapping is a one-class change. Provenance
enforcement lives at the store boundary so it holds for every backend.

### 5.2 Durability — event sourcing & replay

Everything nondeterministic (model calls, IO, timestamps, randomness) is an
**Activity** whose *result is recorded* as an event; the orchestration loop
consumes recorded results. This is the Temporal Workflow/Activity discipline.

This is realized at the executor seam by `Context.record(key, fn)`: the first run of
a step emits `ACTIVITY_RECORDED(key, result)`; any re-execution of the node (the
failure-path retry, or crash recovery) **replays** the recorded result instead of
calling `fn` again — the mechanism for exactly-once model calls and side effects. The
recorder buffers during the pure execute phase and the serial commit phase writes the
events, so the execute/commit boundary (§5.4) is preserved.

> **INV-5.1.** No node output is recomputed on recovery; it is read from
> `NODE_COMPLETED`. A recorded sub-step is likewise replayed, not re-run.
> **INV-5.2.** `replay(log).structural_hash() == live.structural_hash()`.

Module: `graph.GraphProjection.replay`, `views.verify_replay`, `activities.ActivityRecorder`.

### 5.3 Cost routing

Routing is a property of the **node type on the frontier**: high-fan-out /
low-stakes roles → local (cheap); outcome-gating roles (critic, aggregator,
planner) → cloud (expensive). Capability is spent only where it decides whether
the target is met — the mechanism that preserves a capped local-model budget.

Module: `executors.ModelRouter`.

### 5.4 Scheduler loop (`scheduler.run`)

```
seed GOAL(g0) and ORCHESTRATOR(g1)
loop:
    if governor.check(): terminate(reason)
    frontier = ready_set()                     # the parallel batch
    if frontier == ∅: terminate(NO_PROGRESS)   # deadlock/starvation guard
    frontier = beam_prune(frontier)            # optional

    admit:   for node in frontier:             # SERIAL, single-writer
                 if governor.check(): break     # cost cap enforced as it accrues
                 route + charge + NODE_STARTED
    execute: outcomes = [run(node) for node in admitted]   # PURE; Phase 2: asyncio.gather
                 # each run(): resolve inputs, executor.run(...), validate(result)
                 # raised OR malformed -> carried as a failure outcome
    commit:  for outcome in outcomes:          # SERIAL, single-writer, deterministic
                 if outcome.failed: NODE_FAILED; spend global retry; NODE_REOPENED (logged)
                 else: NODE_COMPLETED(output, score, signal); if patch: apply_patch()  # §2.2
                       if signal == GOAL and predicate(output): GOAL_SATISFIED; terminate

    generation += 1; if > cap: terminate(GENERATION_CAP)
```

The batch is dataflow-independent but not side-effect-independent, so **admit** and
**commit** (which touch the governor, graph, and store) stay serial and are the
single-writer serialization point; only **execute** is pure. This is the
execute-parallel / commit-serially boundary Phase 2 parallelizes (§9).

---

## §6 Acyclicity (Layer 6)

Two techniques, both active:

1. **Generation monotonicity** — an edge is legal iff `gen(consumer) >
   gen(producer)`. Enforced on both patch application and log fold
   (`graph.assert_can_add_edge`). O(1); makes the leveled-graph structure fall
   out for free.
2. **Single assignment** — a node's output is written once; a second
   `NODE_COMPLETED` raises (`SingleAssignmentError`). Refinement produces a new
   version, never a write-back.

Online cycle detection (Pearce–Kelly) is unnecessary here because monotonicity
already precludes cycles; §9 notes when you'd add it (arbitrary same-generation
edges).

> **INV-6.1.** The projection is acyclic at every prefix of the log.

---

## §7 Failure modes & governance (Layer 7)

All caps live in `CostGovernor` and are consulted before each tick and each
patch:

| Bound | Guards against |
|---|---|
| `max_generation` | unbounded unrolling (non-convergence) |
| `max_nodes` | breadth explosion |
| `max_total_cost` | spend runaway (local-model budget) |
| `retry_budget` (**global**) | retry storms (nested retries amplifying one failure) |

A run terminates for **exactly one** reason:

```
GOAL_SATISFIED | GENERATION_CAP | BUDGET_EXHAUSTED | NO_PROGRESS
```

> **INV-7.1.** Every run terminates with one of the four reasons; none is
> silent. `NO_PROGRESS` fires when the frontier is empty but the goal is unmet
> (deadlock/starvation), so a stuck graph is reported, not hung.

Module: `governor.CostGovernor`, `scheduler.run`.

---

## §8 Invariant register (and where the PoC proves each)

| Invariant | Proof in code |
|---|---|
| INV-0 / INV-5.2 log is source of truth; replay identical | `demo.py` DURABILITY; `test_cross_run_determinism`; `test_reopen_survives_replay_when_run_caps_after_failure`; `test_replay_equivalence_under_random_failures` (replay holds under injected failure) |
| INV-4.2 fail-closed on malformed output | `test_malformed_executor_output_is_fail_closed` |
| INV-5.1 recorded sub-step replayed (exactly-once) | `test_recorded_activity_is_replayed_on_retry` |
| INV-1.1 provenance-at-write (no NULLs) | `demo.py` GUARD 2 + PROVENANCE GAPS = none |
| INV-2.1 emitted edges monotonic | construction trace shows strictly rising `gen` |
| INV-3.1 goal predicate gates termination | `CoverageGoal.satisfied`; `test_budget_terminates_runaway` |
| INV-4.1 typed completion signal | `Signal` enum; scheduler branches on `result.signal` only |
| INV-6.1 acyclic at every prefix | `demo.py` GUARD 1 rejects the back-edge |
| INV-7.1 four termination reasons | `demo.py` (goal) + `test_budget_terminates_runaway` (cap) + `test_cost_cap_enforced_within_a_wide_tick` (mid-tick budget) |
| execute/commit ordering deterministic (§5.4, §9) | `test_batch_commit_ordering_is_deterministic`; `test_concurrent_execute_matches_serial` + `test_batch_executes_concurrently` (real parallelism, same result) |
| async executors → literal asyncio.gather (§9) | `test_async_executor_runs_and_replays`; `test_async_batch_runs_concurrently` |
| distributed workers ≡ batch; backpressure bounds in-flight (§9) | `test_workers_produce_same_result_as_batch`; `test_workers_pull_frontier_concurrently`; `test_backpressure_bounds_in_flight` |
| recorded activity replayed / exactly-once (§5.2) | `test_recorded_activity_is_replayed_on_retry` |
| real token→$ cost recorded (§5.3) | `test_real_cost_recorded_from_usage`; `test_openrouter_executors_run_and_record_provider_cost` (per-model provider cost) |
| retryable/terminal taxonomy; refusal not retried (§9) | `test_terminal_error_is_not_retried`; `test_openrouter_refusal_is_terminal` |
| real ORCHESTRATOR: bounded plan + guaranteed coverage (§9) | `test_openrouter_orchestrator_plans_and_guarantees_coverage` |
| cost-routing eval measures success vs cost (§9) | `test_eval_harness_ranks_policies_by_cost`; `test_eval_runs_mock_registry_offline` |
| SSA write-once | `test_single_assignment_write_once` |

Run `python demo.py` and `python test_invariants.py`; both assert these.

---

## §9 PoC → production delta

What changes, and nothing else does:

* **Batch → concurrency.** *Done.* The loop separates a pure **execute** phase from
  the serial **admit**/**commit** phases (§5.4); `Scheduler(max_workers=N)` runs the
  admitted batch concurrently, capped at `max_workers` (the real WIP limit), with
  results committed in admitted order so the run stays deterministic
  (`test_concurrent_execute_matches_serial`). Two execute paths: sync executors run in
  a thread pool (GIL releases during I/O); an **`async def` executor triggers a literal
  `asyncio.gather`** over the batch (semaphore = WIP cap; `ctx.record_async` for awaited
  steps) — `test_async_batch_runs_concurrently`. The store's `seq` remains the single
  serialization point.
* **Distributed workers + backpressure.** *Done (in-process reference).*
  `Scheduler.run_workers(n_workers, max_inflight)` inverts the loop to a **pull** model:
  workers pull ready nodes from a shared frontier queue and execute the pure phase
  concurrently, while a **single-writer coordinator** does all admit + commit — so the
  event log stays single-writer even across the worker boundary. It produces the *same
  result* as `run()` (`test_workers_produce_same_result_as_batch`). `max_inflight` is
  **backpressure** — the coordinator admits at most that many nodes ahead of commit, so
  it never over-charges cost or floods the queue (`test_backpressure_bounds_in_flight`).
  The asyncio implementation is the reference shape for real remote workers pulling from
  a durable queue (Redis/SQS) and posting outcomes back.
* **SQLite → Postgres**, in-memory projection → **Neo4j** fold. Same events, same
  guards (they sit at the store boundary).
* **Deterministic executors → real models** at the single `Executor.run` seam.
  Wrap each model/tool/IO call as a recorded Activity so INV-5.1 still holds.
  *Done for the local fleet, cloud fleet, and the planner — the role logic
  (ENGINEER/REVIEWER/ORCHESTRATOR) and prompts live once in `roles.py` over an abstract
  `chat`; each provider module supplies only a thin transport, so a new backend is a
  transport, not another copy of the roles:*
  - **Local** (`executors_ollama`): retrieval SCOUT (corpus ranking; offline lexical
    default, embeddings-ready) + Ollama ENGINEER + REVIEWER.
  - **Cloud** (`executors_openrouter`): OpenAI-compatible ORCHESTRATOR + ENGINEER +
    REVIEWER via a raw-urllib transport (OpenRouter default; provider-agnostic by
    base_url — Ollama /v1, vLLM, any skin). No SDK ⇒ `max_retries=0` by construction;
    refusals/content-filter/4xx → `TerminalExecutionError` (no retry — the terminal
    half of the retryable/terminal taxonomy); per-model cost is the provider's own
    `usage.cost`.
  - **Real ORCHESTRATOR** (the recursive-patch answer): the model emits a *flat* plan
    (`schemas.PLAN_SCHEMA`) and the executor assembles the subgraph deterministically,
    post-filling required subtopics for guaranteed coverage.
  All model calls are recorded activities with real token→$ cost; structured output +
  validate/repair land on the fail-closed guard (INV-4.2). GOAL/SYNTHESIZER stay mock.
  Run `python demo_ollama.py` (local) or `python eval_cost_routing.py` (cost routing).
* **Idempotency keys** on external side-effecting Activities so replay/retry does
  not double-execute the world. *Done at the seam:* `Context.record(key, fn)` +
  `ACTIVITY_RECORDED` make each recorded step exactly-once across retries (§5.2).
* **WIP as admission control**: enforce the blackboard WIP limit as a real
  concurrency cap on `STARTED` nodes.
* **Observability**: the graph is already the trace — expose `critical_path()`
  for latency and the provenance fields for lineage queries; emit the event log
  to your telemetry sink.
* **Online cycle detection** only if you relax the generation rule to allow
  same-generation cross-edges (then add Pearce–Kelly incremental topo order).

---

## Module map

| Module | Spec sections |
|---|---|
| `core.py` | §1 (ids, enums, provenance, events) |
| `store.py` | §1.3, §1.5, §5.1 (log + provenance guard) |
| `graph.py` | §1.1, §1.4, §6, lineage (§5, §8) |
| `nodes.py` | §2.1, §2.3 (patch, placeholders) |
| `activities.py` | §5.2 (recorded activities — replay/idempotency at the executor seam) |
| `schemas.py` | §2.1, §9 (shared structured-output contracts: DRAFT/REVIEW/PLAN — flat, non-recursive) |
| `executors.py` | §3.1, §4.1–4.2, §5.3, §9 (mock roles, critic loop, routing, retryable/terminal taxonomy) |
| `roles.py` | §3.1, §4.1, §5.2–5.3, §9 (**single source** of the model-backed ENGINEER/REVIEWER/ORCHESTRATOR logic + prompts + the shared validate→repair loop, over an abstract `chat`; real retrieval SCOUT) |
| `executors_ollama.py` | §3.1, §5.2 (native-Ollama `chat` — `format=` structured transport + registry; injectable call, offline-testable) |
| `executors_openrouter.py` | §4.1, §5.2–5.3, §9 (OpenAI-compatible `chat` — raw-urllib transport + refusal/cost extract + registry; `max_retries=0` by construction; injectable transport) |
| `evaluation.py` | §9 (cost-routing eval harness: policies, run_eval, cost_routing_report) |
| `steering.py` | §3.2–3.4 (predicate, filter, beam, value) |
| `governor.py` | §7 (bounds, retry budget) |
| `scheduler.py` | §2.2, §5.4 (the loop) |
| `views.py` | §4.3, §5.2, §8 (blackboard, replay, lineage, cost report) |
| `demo.py`, `demo_ollama.py`, `eval_cost_routing.py`, `test_invariants.py` | §8, §9 (executable proofs + live demos) |
