# dyndag — dynamic-DAG multi-agent orchestrator (PoC)

A runnable proof of concept for a multi-agent system whose task graph
**constructs itself on the fly**: expander nodes emit subgraphs computed as
downstream variables of prior nodes, a critic steers construction toward a target
outcome, and the whole artifact stays acyclic because iteration is expressed as
new generations (SSA), never cycles.

It is the reference implementation of [`SPEC.md`](SPEC.md), which formalizes the
seven approved layers. The **offline core** needs no external services and no
network — SQLite stands in for Postgres, deterministic executors stand in for
models, so it runs anywhere. **Real executors** (local Ollama + cloud
OpenAI-compatible) plug in at one seam and are entirely optional. The substrate is
then pointed at standard reasoning benchmarks (GSM8K, Hendrycks MATH, AIME) to
measure what the self-construction primitives actually buy.

> **The one idea.** The system never draws a graph and runs it. The graph
> *precipitates* as computation proceeds, and the only mutable object in the whole
> system is an append-only event log — the graph, the frontier, and the blackboard
> are all deterministic folds over that log. Replaying the log reconstructs the run
> bit-for-bit.

---

## Table of contents

- [Quickstart](#quickstart)
- [Requirements & installation](#requirements--installation)
- [Status](#status)
- [The mental model](#the-mental-model-why-a-dag-that-builds-itself)
- [What the demo shows (one run)](#what-the-demo-shows-one-run)
- [Repository layout](#repository-layout)
- [Architecture (layers → modules)](#architecture-layers--modules)
- [Concepts in depth](#concepts-in-depth)
  - [Nodes, generations, SSA](#nodes-generations-and-ssa-ids)
  - [The event log & store](#the-event-log--store)
  - [Provenance-at-write](#provenance-at-write)
  - [Subgraph emission, placeholders, dynamic joins](#subgraph-emission-placeholders-and-dynamic-joins)
  - [Executors, signals, fail-closed validation](#executors-signals-and-fail-closed-validation)
  - [Steering: critic loop, goal predicates, constraints](#steering-the-critic-loop-goal-predicates-constraints)
  - [φ-nodes and the merge policies](#φ-nodes-and-the-merge-policies)
  - [Cost routing, the governor, termination](#cost-routing-the-governor-and-termination)
  - [Concurrency](#concurrency)
  - [Durability: replay & recorded activities](#durability-replay--recorded-activities)
- [Plugging in real models](#plugging-in-real-models)
- [Benchmark evaluation](#benchmark-evaluation-the-research-spine)
- [Command reference](#command-reference)
- [Public API surface](#public-api-surface)
- [Testing](#testing)
- [Design commitments](#design-commitments-from-the-approved-decisions)
- [PoC → production delta](#poc--production-delta)
- [License](#license)

---

## Quickstart

```bash
python demo.py            # end-to-end run + all guards, with narrated output
python test_invariants.py # 48 focused checks of the load-bearing guarantees
```

Both exit `0` and assert their invariants — **no dependencies, no network**. On a
reference run `demo.py` constructs a 10-node graph over 7 generations (49 events),
reaches `goal_satisfied`, and verifies that replaying the log reproduces a
bit-identical graph. `test_invariants.py` prints `48 invariant tests passed.`

---

## Requirements & installation

| | |
|---|---|
| **Python** | 3.9+ (verified on 3.9.6) |
| **Offline core + test suite** | **zero third-party packages** — standard library only (`sqlite3`, `asyncio`, `concurrent.futures`, `urllib`, `hashlib`, `json`, `dataclasses`, `math.comb`) |
| **Local real models** (`executors_ollama`, `demo_ollama.py`) | the [`ollama`](https://pypi.org/project/ollama/) Python package **+** a running `ollama serve` with a model pulled |
| **Cloud / OpenAI-compatible models** (`executors_openrouter`, the evals) | **no package** — a raw-`urllib` transport speaks the OpenAI wire format; you only need an API key (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`) |
| **Benchmark data** | user-downloaded JSONL (GSM8K / MATH / AIME) — never committed (see [`.gitignore`](.gitignore)) |

There is deliberately **no `requirements.txt` / `pyproject.toml`**: the core has
nothing to install. The repo is run in place — clone it and run the scripts from
the repository root, where `import dyndag` resolves:

```bash
git clone <this-repo> && cd <repo-root>
python demo.py            # works immediately, offline
pip install ollama        # ONLY if you want the native-Ollama local path
export OPENROUTER_API_KEY=...   # ONLY if you want the cloud path
```

Every real-model entry point **skips cleanly** (prints a `[skip]` notice and exits
`0`) when its provider is unavailable, so nothing here hard-fails for lack of a key
or a server.

---

## Status

The self-constructing substrate, a real local+cloud executor fleet, full
concurrency, and a benchmark eval stack (cost-routing + GSM8K/MATH/AIME with an
independent re-derivation checker) are done and merged. Persistence is the
remaining phase.

| Phase | | |
|---|---|---|
| **0 — Harden substrate** | ✅ done | failure-path durability, fail-closed dispatch, mid-tick cost cap, execute/commit boundary; failure-injection + property sweeps |
| **1 — Executor seam** | ✅ done | recorded-activity + idempotency; structured output + validate→repair; real role-types (SCOUT/ORCHESTRATOR/ENGINEER/REVIEWER) across local + cloud; retryable/terminal error taxonomy; real token→$ cost; cost-routing eval |
| **2 — Concurrency** | ✅ done | parallel execute batch (thread pool + literal `asyncio.gather`); distributed workers pulling the frontier; backpressure |
| **Eval — GSM8K** | ✅ done | five arms — single-pass, SSA, self-consistency (φ majority-vote), combined (φ-of-SSA via a dynamic join), and `combined_v2` (greedy-anchored, verify-gated merge); accuracy, \$/correct, generations, wall-clock, live replay, paired McNemar |
| **Eval — MATH / AIME** | ✅ done | Hendrycks competition-math + AIME via a ported `is_equiv` LaTeX-equivalence grader; an **independent re-derivation CHECKER** (verify by re-solving, not self-assessment); a parallel, checkpointed, resumable runner |
| **3 — Persistence** | 🔴 next | Postgres `EventStore` adapter, snapshotting, resume-from-log (the `EventStore` ABC + recorded activities already lay the foundation) |

---

## The mental model (why "a DAG that builds itself")

Classic agent frameworks draw a workflow graph and then execute it. This system
inverts that: an executor's output can be **structure** (a `SubgraphPatch`), so the
graph grows as a *trace* of what the agents decided to do. Three rules keep that
safe and auditable:

1. **Single assignment + generation monotonicity ⇒ acyclic.** A node is written
   exactly once. "Refining" node `x` does not mutate it; it creates `x′` at a
   strictly higher *generation* with an edge `x → x′`. Because every emitted edge
   climbs in generation, the materialized artifact is always a DAG — while the
   *behavior* is fully recurrent (loops are unrolled into fresh generations). This
   is where the `.g<N>` in every node id comes from: `engineer.g3` and
   `engineer.g5` are two distinct SSA versions of the same logical role.

2. **The log is the only source of truth.** Every fact is an append-only event.
   The graph, the ready frontier, and the Kanban "blackboard" are *derived* by
   folding the log; derived states (like `READY`) are never stored. Consequence:
   `replay(log).structural_hash() == live.structural_hash()` — durability is a
   theorem, and the demo checks it.

3. **Construction is data, not code.** The sole construction primitive is "an
   executor returns a patch." There is no static fan-out and no conditional-edge
   DSL; the scheduler applies whatever subgraph an expander computes, after a
   fail-closed validation pass.

The seven layers from `SPEC.md`, in one line each: **(1)** typed SSA nodes +
dataflow firing; **(2)** subgraph emission as the only construction mechanism;
**(3)** goal-conditioned steering (critic loop, goal predicate, constraints,
aggregation); **(4)** agents = node-type executors, blackboard = the DAG;
**(5)** event-sourced store + topological scheduler + cost routing; **(6)**
acyclicity by generation + single-assignment; **(7)** bounded governance with four
deterministic termination reasons.

---

## What the demo shows (one run)

A goal — *"produce a vetted brief covering ingestion, reasoning, provenance,
scoring ≥ 0.80"* — drives (exact trace from a live `python demo.py`):

1. the **orchestrator** (`orchestrator.g1`) emits a subgraph: 3 scouts (`g2`) → an
   engineer (`engineer.g3`) → a reviewer (`reviewer.g4`);
2. the **reviewer** scores the first draft **0.60**, so it emits a *refinement*
   stratum — `engineer.g5` (a new SSA version fed the failed draft + critique) then
   `reviewer.g6` — a loop with **no cycle**;
3. the refined draft scores **0.90**, so the reviewer approves and emits a
   **synthesizer** (`synthesizer.g7`) that aggregates to the terminal brief and
   raises `GOAL`;
4. the injected **goal predicate** accepts it → `goal_satisfied` in **7
   generations**, **10 nodes**, **49 events**.

Then it verifies, in the same run: a **back-edge is rejected** (acyclicity guard),
a **lineage-less write is rejected** (provenance-at-write guard), **replay
reconstructs a bit-identical graph** (durability — matching structural hashes),
cost splits across **local vs cloud** tiers (`local: 5.0`, `cloud: 20.0` nominal
units), and there are **zero provenance gaps**.

---

## Repository layout

```
.
├── README.md                     this document
├── SPEC.md                       the governing specification — contracts, invariants, PoC→prod delta
├── LICENSE                       proprietary license (Plutus CCE LLC)
├── demo.py                       offline end-to-end run + guards (no deps, no network)
├── demo_ollama.py                Phase-1 slice: a real local ENGINEER (Ollama), rest mock
├── eval_cost_routing.py          cost-routing eval: all-local vs all-cloud vs type-aware
├── eval_gsm8k.py                 GSM8K eval: single / ssa / sc / combined + paired McNemar
├── run_eval_parallel.py          parallel, checkpointed, resumable GSM8K/MATH/AIME runner
├── test_invariants.py            48 dependency-free invariant checks (the executable proofs)
└── dyndag/                       the package (≈ 2,900 LOC)
    ├── __init__.py               public API surface (see "Public API surface")
    ├── core.py              216  ids, enums, Provenance (+ guard), the Event record
    ├── store.py             123  append-only event log + provenance-at-write guard (SQLite)
    ├── graph.py             262  the projection: fold, ready-set (Kahn), acyclicity, lineage, joins
    ├── nodes.py             104  NodeSpec, SubgraphPatch, JoinSpec, $placeholder resolution
    ├── activities.py         65  recorded activities — exactly-once model/IO calls across retries
    ├── schemas.py            84  structured-output contracts: DRAFT / REVIEW / PLAN / SOLUTION / VERIFY
    ├── executors.py         333  mock role fleet, ExecutionResult/Context, ModelRouter, error taxonomy
    ├── roles.py             246  model-backed ENGINEER/REVIEWER/ORCHESTRATOR (single source) + repair loop + retrieval SCOUT
    ├── executors_ollama.py   71  native-Ollama `chat` transport (format= structured output)
    ├── executors_openrouter.py 141 OpenAI-compatible `chat` transport (raw urllib, refusal + cost)
    ├── phi.py               176  φ-merge of competing branches + 5 merge policies
    ├── steering.py           99  goal predicates (injectable), constraint filter, beam pruning
    ├── governor.py           70  Budget + CostGovernor (caps, global retry budget)
    ├── scheduler.py         587  the loop: admit→execute→commit; batch / async / distributed workers; join materialization
    ├── views.py             120  blackboard, replay verification, lineage + cost reports
    ├── evaluation.py        254  eval harness: policies, cost-routing + GSM8K reports, paired McNemar
    ├── tasks_gsm8k.py       478  GSM8K as a DAG task: SOLVER/CHECKER/ANSWER roles, 5 modes, exact-match judge
    └── tasks_math.py        261  MATH/AIME: ported Hendrycks is_equiv grader + \boxed{} extraction + registry
```

---

## Architecture (layers → modules)

```
core.py                 ids, enums, Provenance (provenance-at-write), Event    L1 L5
store.py                append-only log + provenance guard (Postgres analog)   L5
graph.py                fold-of-log projection, ready-set, acyclicity,         L1 L6
                        lineage, activity cache, dynamic-join fold
nodes.py                NodeSpec, SubgraphPatch, JoinSpec, placeholders        L2
activities.py           recorded activities — exactly-once model calls/replay  L5
schemas.py              structured-output contracts (DRAFT/REVIEW/PLAN +       L2
                        SOLUTION/VERIFY)
executors.py            mock role executors, typed signals, ModelRouter,       L3 L4 L5
                        retryable/terminal error taxonomy
roles.py                model-backed ENGINEER/REVIEWER/ORCHESTRATOR over an     L3 L4
                        abstract chat (one copy) + prompts + retrieval SCOUT
executors_ollama.py     native-Ollama chat: format= transport + registry       L5
executors_openrouter.py OpenAI-compatible chat: urllib transport, refusal+cost L4 L5
phi.py                  φ-merge of competing branches: argmax / majority /      L4 L6
                        confidence-weighted / greedy-anchored / union
steering.py             pluggable goal predicate, constraint filter, beam      L3
governor.py             caps + GLOBAL retry budget                             L7
scheduler.py            the loop: admit→execute→commit; batch / async /        L5
                        distributed workers; materializes dynamic joins
views.py                blackboard, replay verify, lineage + cost report       L4 L5
tasks_gsm8k.py          GSM8K task: solver/checker/answer roles; single/ssa/   L3
                        sc/combined/combined_v2 (φ-of-SSA via dynamic join)
tasks_math.py           MATH/AIME: LaTeX-equivalence grader + same role fleet  L3
evaluation.py           eval harness: cost-routing + GSM8K/MATH reports        L5
```

---

## Concepts in depth

### Nodes, generations, and SSA ids

A node is a **typed, single-assignment variable binding** with a stable id
`⟨key⟩.g⟨generation⟩`. `generation = max(parent.generation) + 1`, which is the one
rule that makes every edge monotonic. Node state moves
`CREATED → STARTED → COMPLETED | FAILED` (`READY` is *derived*, never stored). The
output is written exactly once on `COMPLETED`; a second completion raises
`SingleAssignmentError`. (`core.NodeType`, `core.NodeState`, `graph.NodeRecord`.)

Node **types are functions, not personalities**:

| `NodeType` | Function in the dataflow graph |
|---|---|
| `GOAL` | terminal input source holding the target spec |
| `ORCHESTRATOR` | the expander — emits subgraphs |
| `SCOUT` | frontier retrieval |
| `ENGINEER` | generation / refinement (GSM8K/MATH: the SOLVER) |
| `REVIEWER` | critic — the error signal (GSM8K/MATH: the independent CHECKER) |
| `SYNTHESIZER` | aggregator — convergence (GSM8K/MATH: the ANSWER finalizer) |
| `PHI` | φ-merge — selects/combines competing branches, records the selector |
| `TESTER` / `HARDENER` / `OPS` / `MAINTAINER` / `DEBUGGER` | critic/terminal variants, identical contract (declared, unbound in the PoC) |

### The event log & store

The store assigns a monotonic, gap-free `seq` (the total order that makes replay
deterministic) and enforces the provenance guard. The PoC uses `SQLiteEventStore`
(`:memory:` by default); the interface is an abstract `EventStore`, so
SQLite→Postgres is a one-class change and nothing above the store knows the
difference. The full event vocabulary (`core.EventType`):

| `EventType` | payload keys | role |
|---|---|---|
| `RUN_STARTED` | `goal` | audit |
| `NODE_CREATED` | `node_type, input_ids, params, provenance` | fact + guard |
| `EDGE_CREATED` | `producer, consumer, provenance` | fact + guard |
| `NODE_STARTED` | `tier, model, cost` | dispatch + cost |
| `NODE_COMPLETED` | `output, score, signal, usage, cost_usd` | single-assignment write |
| `NODE_FAILED` | `error, attempt, terminal` | retry accounting |
| `NODE_REOPENED` | `attempt` | a FAILED node returned to the frontier, as a logged fact |
| `ACTIVITY_RECORDED` | `key, result` | a nondeterministic sub-step's result, for replay / exactly-once |
| `SUBGRAPH_EMITTED` | `added_nodes` | lineage of construction |
| `NODE_PRUNED` | `key, reason, expander_id` | constraint-filter audit |
| `JOIN_DECLARED` | `key, quorum, policy` | a dynamic fan-in point to materialize later |
| `JOIN_CONTRIBUTED` | `key, contributor` | one lineage bound its result to a join |
| `GOAL_SATISFIED` | `output` | terminal acceptance |
| `RUN_TERMINATED` | `reason` | one of four reasons |

### Provenance-at-write

Every node and edge carries a `Provenance` tuple
`(run_id, batch_id, generation, created_at, expander_id, parent_ids)`, **stamped at
the moment of creation**. The store rejects any `NODE_CREATED` / `EDGE_CREATED`
whose provenance does not validate (`ProvenanceError`). There is deliberately no
code path that mints a node or edge without lineage — so `started_at`,
`source_batch_id`, and `pipeline_run_id` are structurally incapable of being NULL.
`views.provenance_gaps()` exists to *prove* the set is empty, not to repair it.

### Subgraph emission, placeholders, and dynamic joins

An expander returns a `SubgraphPatch(added_nodes, added_edges, declared_joins,
contributions)`. `scheduler._apply_patch` (1) runs the constraint filter (pruned
specs emit `NODE_PRUNED`), (2) refuses the whole patch if it would exceed
`max_nodes`, (3) resolves logical keys → concrete ids **in dependency order** (a
patch is a small DAG), stamping each new node at `max(parent.generation)+1`, and
(4) emits one `SUBGRAPH_EMITTED` marker.

A node's `params` may contain `$producer` placeholders resolved against upstream
outputs at dispatch (`nodes.resolve_placeholders`): a bare `"$id"` yields the raw
object, inline refs stringify — the LLMCompiler `$1/$2` mechanism scoped to one
node's parameter dict.

**Dynamic joins** (`JoinSpec`) solve adaptive fan-in: when K lineages refine to
*data-dependent* depth, their terminal ids are unknown when the merge is created,
and monotonic edges forbid wiring the merge to them after the fact. So an expander
*declares* a quorum-K join (`JOIN_DECLARED`); each lineage *contributes* its
certified result (`JOIN_CONTRIBUTED`); and once K contributions arrive the
scheduler **materializes** the join as a `PHI` node over the now-concrete
contributors, at `generation = max(contributor)+1`. Acyclicity and replay hold with
no monotonicity relaxation. This is the "graph builds itself on the fly" made real
for adaptive merging (`graph.ready_joins`, `scheduler._materialize_joins`).

### Executors, signals, and fail-closed validation

```
Executor.run(inputs: {producer_id: output}, params: dict, ctx: Context)
        -> ExecutionResult(output, signal, score?, patch?, usage?)
```

`signal` is a **typed** `Signal` (`CONTINUE / DONE / NEEDS_REFINEMENT / GOAL`),
never a control token parsed from text — the structural fix for the "agent emitted
`<complete>` as a literal string" failure class. Before any result can touch the
log or graph, `scheduler._validate_result` + `_validate_patch` reject anything
malformed — wrong type, an untyped signal, a non-numeric score, a patch referencing
a node type with no bound executor, an invalid join quorum — and treat it as a
**failed attempt** (fail-closed). This is exactly the seam where a real model's
structured-output validation + repair loop attaches.

Two error classes drive the retry taxonomy: `MalformedResultError` is **retryable**
(a fresh dispatch may parse), while `TerminalExecutionError` (refusal, content
filter, 4xx) fails the node **without spending the retry budget** (retrying a
refusal just refuses again).

### Steering: the critic loop, goal predicates, constraints

The reviewer is the **controller** in a setpoint/error loop: target score is the
setpoint, `error = target − score`. `error > 0` emits a *refinement stratum* (a new
engineer generation fed the critique + a fresh reviewer); `error ≤ 0` emits the
aggregator. Bounded convergence is guaranteed by the generation cap.

A `GOAL`-signalled node is accepted as terminal **only if the goal predicate
holds** — a premature `GOAL` cannot end the run. The predicate is **injectable**
and is the *only* task-specific coupling in the scheduler: `CoverageGoal`
(covered ⊇ required subtopics ∧ score ≥ target) for the brief task, `AnswerGoal`
(a non-empty final answer was produced) for GSM8K/MATH. A new task is therefore a
predicate + a role fleet, not a scheduler change. `ConstraintFilter` runs on every
patch before instantiation (arc-consistency in miniature); `beam_prune` keeps the
top-`width` frontier candidates by value. (`steering.py`.)

### φ-nodes and the merge policies

The `SYNTHESIZER` aggregates one approved branch (arity-1). A **φ-node** is the
arity-N generalization — the SSA φ-function made concrete — that merges *competing*
definitions of the same result and records the selector (`output["phi"]`) for
lineage. Every policy is a **pure, deterministic, id-sorted fold**, so the graph
stays replay-identical; empty input is fail-closed. Five policies ship
(`phi.py`):

| Policy | Strategy | Used by |
|---|---|---|
| `argmax_score` | best-of-N: highest critic score wins | mid-graph best-of-N |
| `majority_vote` | self-consistency: most frequent answer (optional `norm` so equivalent LaTeX forms aggregate) | `sc` |
| `confidence_weighted_vote` | verifier-weighted: answer with the greatest *total* critic confidence | `combined` |
| `greedy_anchored_select` | **dilution fix**: default to the greedy anchor; override only when an alternative's summed independent corroboration beats it by `margin` | `combined_v2` |
| `union_covered` | combine (not select): union `covered`, merge `sections` | brief fan-in |

### Cost routing, the governor, and termination

Routing is a property of the **node type on the frontier**: high-fan-out /
low-stakes roles (scout, engineer) → **local** (cheap); outcome-gating roles
(critic, aggregator, planner) → **cloud** (expensive). Capability is spent only
where it decides whether the target is met (`executors.ModelRouter`). Real dollar
cost prefers the provider's own `usage.cost`, falling back to a per-tier token-rate
table, `None` for mocks.

All caps live in `CostGovernor` (`governor.py`), consulted before each tick and
each patch. Defaults (`Budget`): `max_generation=12`, `max_nodes=200`,
`max_total_cost=500.0`, `retry_budget=8` (**global** — the defense against retry
storms). A run terminates for **exactly one** reason:

```
GOAL_SATISFIED | GENERATION_CAP | BUDGET_EXHAUSTED | NO_PROGRESS
```

`NO_PROGRESS` fires when the frontier is empty but the goal is unmet, so a stuck
graph is *reported*, not hung.

### Concurrency

Each tick splits into **admit → execute → commit**, where `execute` is pure and the
only parallelizable phase; admit and commit stay serial and single-writer (the
store's `seq` is the serialization point). Four modes, all producing the same
result:

- **Serial** — `Scheduler(...).run(goal)` (default; deterministic, offline).
- **Thread pool** — `Scheduler(max_workers=N)` runs the batch's sync executors
  concurrently (GIL releases during I/O); `N` is the real WIP cap.
- **Async** — an `async def` executor triggers a literal `asyncio.gather` over the
  batch (`ctx.record_async` for awaited steps; a semaphore = the WIP cap).
- **Distributed workers** — `Scheduler(...).run_workers(n_workers, max_inflight)`
  inverts the loop to a **pull** model: workers pull ready nodes from a shared
  frontier queue while a single-writer coordinator admits + commits.
  **Backpressure** (`max_inflight`, default `2×n_workers`) paces admission to
  commit. This in-process asyncio version is the reference shape for real remote
  workers on a durable queue (Redis/SQS).

### Durability: replay & recorded activities

Everything nondeterministic (a model call, retrieval, a timestamp) is wrapped as a
**recorded activity** via `ctx.record(key, fn)` (or `ctx.record_async`): the first
run emits `ACTIVITY_RECORDED(key, result)`; any re-execution (a retry, or crash
recovery) **replays** the recorded result instead of calling `fn` again — the
mechanism for exactly-once model calls and side effects. The recorder buffers during
the pure execute phase; the serial commit phase writes the events, so the
execute/commit boundary is preserved. `views.verify_replay` folds the log into a
fresh projection and asserts its structural hash equals the live one — the
durability claim, made falsifiable.

---

## Plugging in real models

The model-backed roles live **once** in `roles.py` (ENGINEER / REVIEWER /
ORCHESTRATOR + their prompts + the shared `chat_json_with_repair` validate→repair
loop), written against an abstract `chat` capability:

```python
chat(messages, schema, name, required) -> {"content": <dict>, "usage": <dict>}
```

Each provider supplies only a thin `chat` **transport**, so adding a backend is a
transport — not another copy of the roles. Both transports are injectable
(`call=` / `transport=`), so every path is unit-testable **offline** with a fake —
no package, no server, no key. Two ship:

- **Local** — `executors_ollama.py`: native Ollama structured output
  (`ollama.chat(..., format=schema)`) + `temperature` sampling, plus a provider-
  neutral retrieval **SCOUT** (`roles.RetrievalScoutExecutor` — ranks a corpus per
  subtopic; offline lexical Jaccard default, embeddings-ready via a pluggable
  ranker).
- **Cloud / any OpenAI-compatible** — `executors_openrouter.py`: a raw-`urllib`
  transport over `/chat/completions`. **Provider-agnostic by `base_url`**
  (OpenRouter default; point it at Ollama's `/v1`, vLLM, LM Studio, any skin). No
  SDK, so **`max_retries = 0` by construction** (no nested retry storm);
  refusals / content filters / 4xx → `TerminalExecutionError` (never retried);
  per-model cost is the provider's own `usage.cost`.

The **real ORCHESTRATOR** answers the recursive-patch problem: structured-output
backends reject recursive schemas, so the model emits a **flat** `PLAN` (which
scouts, an engineer brief, a review rubric) and the executor assembles the
`scouts → engineer → reviewer` subgraph deterministically, **post-filling** every
required subtopic so coverage is guaranteed no matter what the model plans.

```bash
pip install ollama && ollama serve && ollama pull llama3.2
python demo_ollama.py                 # a real local ENGINEER, everything else mock
python demo_ollama.py llama3.2        # override the model (first CLI arg)

OPENROUTER_API_KEY=... python eval_cost_routing.py   # all-local vs all-cloud vs type-aware
```

`eval_cost_routing.py` runs the default task set under three routing policies and
prints success-vs-cost — the measurable claim being that **type-aware** routing
matches all-cloud's task success at a fraction of the cost while beating all-local
on success (a weak local critic converges less reliably). It skips any policy whose
provider is unavailable.

---

## Benchmark evaluation (the research spine)

The substrate is pointed at standard reasoning benchmarks to **ablate the two
construction primitives** — refinement (SSA) and merge (φ). The same role fleet
runs over the same provider-agnostic `chat`; only the prompt, the goal predicate,
and the grading judge differ between tasks. **The DAG never sees the gold answer**:
the goal predicate only certifies that *an* answer was produced (and survived the
critic/merge); correctness lives in the eval judge alone, so the graph cannot cheat
toward termination.

### The independent re-derivation checker (the correctness mechanism)

The CHECKER (bound to `REVIEWER`) verifies a solution by **re-solving the problem
from scratch** — it is shown neither the gold answer nor the solver's answer (to
avoid rubber-stamping its own reasoning) — and reports an `independent_answer` + a
`confidence`. The *code* then compares the two derivations with a task-specific
equivalence function, so refinement triggers on genuine **disagreement** (an
objective signal) rather than on a model's self-assessment. This is the fix for the
self-verification bias that let a same-model "is this correct?" critic approve its
own confident errors (so the SSA loop never fired). (`schemas.VERIFY_SCHEMA`,
`tasks_gsm8k.CheckerRole`.)

### The five arms

| Arm | Refine? | Merge? | Shape |
|---|---|---|---|
| `single` | — | — | `solver → answer` (one shot, no critic) |
| `ssa` | ✅ | — | `solver → checker → …` adaptive-depth refinement (each retry a new generation) |
| `sc` | — | ✅ | K sampled solvers → a `PHI` **majority-vote** |
| `combined` | ✅ | ✅ | K adaptive lineages → a **verifier-weighted** `PHI` over a **dynamic join** (the compositional claim) |
| `combined_v2` | ✅ | ✅ | **greedy anchor + verify-first gate**: on checker *agreement* return the greedy answer (≡ `single`, no dilution); only on *disagreement* fan out K−1 temp-sampled exploratory lineages into a **greedy-anchored** merge that overrides the anchor only on strong independent corroboration |

`combined_v2` is the newest arm and exists to fix *merge dilution* — the failure
where noisy temperature samples outvote a correct greedy answer. By construction it
contains `single` as a special case and **cannot dilute below it**, yet a
genuinely better-corroborated answer still wins. (`tasks_gsm8k.GsmOrchestrator`,
`phi.greedy_anchored_select`; the `--margin` flag sets the override threshold.)

### The metrics

`gsm8k_report` prints the paper's table — **accuracy**, total cost, **\$ per
correct answer** (the cost/quality frontier), mean **generations** (did SSA
iteration help?), mean **wall-clock** (latency under concurrency), and
**replay-equivalence holding live during the eval** (even under temperature-sampled
self-consistency and the adaptive join, because every sample is a recorded
activity). `paired_report` adds a **paired McNemar** analysis over the *shared* task
set — the right test when every policy runs identical problems: it controls for
per-problem difficulty, exposes strict domination, and tells a real accuracy gap
from noise (`evaluation.mcnemar`, an exact two-sided binomial p-value).

### GSM8K

Grade-school math, graded by exact match on the final number (`tasks_gsm8k`: a
robust last-numeric-token extractor handles `1,024 → 1024`, `18.0 → 18`, prose
wrappers). The dataset is user-downloaded (MIT-licensed) and never committed:

```bash
curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o test.jsonl
OPENROUTER_API_KEY=... python eval_gsm8k.py --data test.jsonl --n 250 \
    --configs single,ssa,sc,combined --k 5 --dump results.json
```

Provider is auto-detected (Ollama at `:11434`, else `OPENROUTER_API_KEY` →
`--cloud-model`); with no `--data` it runs a tiny authored arithmetic sample
(plumbing smoke, **not** a benchmark number). Note `eval_gsm8k.py` exposes
`single,ssa,sc,combined`; the `combined_v2` arm is driven through
`run_eval_parallel.py`.

### MATH (Hendrycks) & AIME

Competition math whose answers are LaTeX (`\frac`, `\sqrt`, tuples, degrees), so
grading is **equivalence, not exact match**. `tasks_math.py` ports the canonical
Hendrycks `is_equiv` normalization (fraction / sqrt / unit / spacing canonicalization
+ a numeric fallback for pure decimals) and extracts the last `\boxed{}`. The φ merge
votes on the *normalized* answer so equivalent LaTeX forms aggregate instead of
splitting the vote. The loader reads JSONL rows `{problem, answer, level, subject}`
and supports a `min_level` difficulty filter (1–5); AIME is handled as MATH-format
JSONL. Data is user-downloaded and gitignored (`math*.jsonl`, `aime*.jsonl`).

### The parallel runner

`run_eval_parallel.py` is the time-efficient path for larger runs against a network
model (e.g. Ollama Cloud): it overlaps all `(config, task)` runs in a flat thread
pool, **checkpoints every completed run to JSONL immediately**, **resumes** by
replaying the checkpoint on restart (survives a kill), and **retries** errored runs.
It covers all five arms and both graders:

```bash
python run_eval_parallel.py --task math --data aime2025.jsonl --n 30 --pool 8
python run_eval_parallel.py --task math --data math500.jsonl --configs single,combined_v2 --margin 1.0
python run_eval_parallel.py --task gsm8k --data test.jsonl --configs single,ssa,sc,combined,combined_v2
```

> **Note on cost.** The default cloud/`/v1` model names in the eval scripts
> (`deepseek/deepseek-v4-flash`, `deepseek-v4-flash:cloud`) are defaults, not
> endorsements — override `--cloud-model` / `--model`. A local Ollama that fronts
> Cloud models bills real tokens, so a non-zero "all-local" cost is correct, not a
> fallback.

---

## Command reference

| Script | Purpose | Key flags |
|---|---|---|
| `demo.py` | offline end-to-end run + guards | *(none)* |
| `test_invariants.py` | 48 invariant checks | *(none)* |
| `demo_ollama.py` | real local ENGINEER (Ollama), rest mock | `[model]` (positional; default `llama3.2`) |
| `eval_cost_routing.py` | all-local / all-cloud / type-aware routing | `[local_model] [cloud_model]`; env `OPENROUTER_API_KEY` |
| `eval_gsm8k.py` | GSM8K: single/ssa/sc/combined + McNemar | `--data --n --seed --configs --k --sc-temperature --local-model --cloud-model --max-workers --dump` |
| `run_eval_parallel.py` | parallel, resumable GSM8K/MATH/AIME runner | `--task{gsm8k,math} --data --n --seed --k --pool --sc-temperature --margin --configs --model --min-level --ckpt --out` |

---

## Public API surface

`from dyndag import ...` (see [`dyndag/__init__.py`](dyndag/__init__.py)):

- **Engine** — `Scheduler`, `RunResult`, `SQLiteEventStore`, `EventStore`,
  `GraphProjection`, `Budget`, `CostGovernor`, `ModelRouter`.
- **Construction** — `NodeSpec`, `SubgraphPatch`, `JoinSpec`, `ActivityRecorder`.
- **Executors & contracts** — `Executor`, `ExecutionResult`, `Context`,
  `MalformedResultError`, `TerminalExecutionError`, `default_registry`.
- **Enums & records** — `NodeType`, `NodeState`, `Signal`, `Tier`,
  `TerminationReason`, `Provenance`, `Event`, `EventType`.
- **Roles & providers** — `EngineerRole`, `ReviewerRole`, `OrchestratorRole`,
  `RetrievalScoutExecutor`, `chat_json_with_repair`, `ollama_chat`,
  `ollama_registry`, `openrouter_chat`, `openrouter_registry`, `http_transport`.
- **φ-merge** — `PhiExecutor`, `argmax_score`, `majority_vote`,
  `confidence_weighted_vote`, `greedy_anchored_select`, `union_covered`.
- **Steering** — `GoalPredicate`, `CoverageGoal`, `AnswerGoal`, `ConstraintFilter`.
- **Views** — `render_blackboard`, `verify_replay`, `lineage_report`,
  `provenance_gaps`, `cost_report`.
- **Eval** — `EvalTask`, `TaskResult`, `DEFAULT_TASKS`, `run_task`, `run_eval`,
  `cost_routing_report`, `gsm8k_report`, `mcnemar`, `paired_report`,
  `per_problem_rows`, `accuracy`, `cost_per_correct`, `savings_pct`.
- **Tasks** — `gsm8k_registry`, `load_gsm8k`, `sample_tasks`, `SolverRole`,
  `CheckerRole`, `AnswerExecutor`, `GsmOrchestrator`, `math_registry`, `load_math`,
  `is_equiv`.

A minimal offline run:

```python
from dyndag import Scheduler, SQLiteEventStore, default_registry, Budget

goal = {"topic": "demo", "subtopics": ["a", "b"], "target_score": 0.80}
result = Scheduler(SQLiteEventStore(), default_registry(), budget=Budget()).run(goal)
print(result.reason.value, result.terminal_output)
```

---

## Testing

`python test_invariants.py` runs **48** dependency-free checks and prints
`48 invariant tests passed.` (the custom runner raises on the first failing assert).
The suite is offline and hermetic: the deterministic mock fleet never touches a
network, failure is driven by injection wrappers (`_Flaky`, `_RandomFlaky`), and
the real-provider paths are exercised with **injected fake transports** (no package,
no server, no key). Coverage spans single-assignment, cross-run determinism,
replay-under-random-failure, fail-closed dispatch, mid-tick cost caps,
deterministic batch commit, exactly-once recorded activities, real token→\$ cost,
thread-pool / async-gather / distributed-worker concurrency with backpressure, the
retryable/terminal taxonomy, the real planner's guaranteed coverage, the five φ
policies, dynamic-join materialization, every GSM8K arm, `combined_v2`'s
no-dilution + override behavior, paired McNemar, and the MATH LaTeX-equivalence
grader. `SPEC.md` §8 maps each invariant to the exact test(s) and demo guard(s) that
prove it.

---

## Design commitments (from the approved decisions)

- **Subgraph emission is the only construction primitive** — no static fan-out,
  no edge DSL; structure is data an expander returns.
- **Agents are node-type executors**; a complex agent is a nested subgraph.
- **Blackboard = the DAG itself** — cards are nodes, WIP caps concurrency, "done"
  is a typed event (not a parsed token).
- **The log is the only source of truth**; the graph is a fold over it.

---

## PoC → production delta

`SPEC.md` §9 is the authoritative list; in brief, what changes and nothing else
does: **SQLite → Postgres** (events table) and the in-memory projection → a **Neo4j**
fold (same events, same guards, which sit at the store boundary); the in-process
asyncio **distributed workers → real remote workers** on a durable queue (Redis/SQS);
and the deterministic mock executors → **real models** at the single `Executor.run`
seam (already done for the local fleet, cloud fleet, and planner). Observability is
largely free — the graph *is* the trace: `critical_path()` gives latency, the
provenance fields give lineage, and the event log streams to any telemetry sink.
Online cycle detection (Pearce–Kelly) is only needed if you relax the generation
rule to allow same-generation cross-edges.

See [`SPEC.md`](SPEC.md) for the full contracts, schemas, invariants, and the
PoC→prod delta.

---

## License

Copyright (c) 2026 Karthik Naidu and Associated Researchers. All rights reserved.

This software is proprietary. Plutus CCE LLC is granted a non-exclusive,
non-transferable, non-sublicensable, limited license for internal use only,
subject to the terms and conditions in the [LICENSE](LICENSE) file.

All intellectual property rights are retained by Karthik Naidu and the
Associated Researchers listed as contributors to this project. For
licensing inquiries, contact: karthik.naidu.east@gmail.com.
