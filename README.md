# dyndag — dynamic-DAG multi-agent orchestrator (PoC)

A runnable proof of concept for a multi-agent system whose task graph
**constructs itself on the fly**: expander nodes emit subgraphs computed as
downstream variables of prior nodes, a critic steers construction toward a target
outcome, and the whole artifact stays acyclic because iteration is expressed as
new generations (SSA), never cycles.

It is the reference implementation of `SPEC.md`, which formalizes the seven
approved layers. The **offline core** needs no external services and no network —
SQLite stands in for Postgres, deterministic executors stand in for models, so it
runs anywhere. **Real executors** (local Ollama + cloud OpenAI-compatible) plug in at
one seam and are entirely optional (see *Real executors*, below).

## Quickstart

```bash
python demo.py            # end-to-end run + all guards, with narrated output
python test_invariants.py # 28 focused checks of the load-bearing guarantees
```

Both exit 0 and assert their invariants — no dependencies, no network.

## Status

The self-constructing substrate, a real local+cloud executor fleet, and full
concurrency are done and merged; persistence is the remaining phase.

| Phase | | |
|---|---|---|
| **0 — Harden substrate** | ✅ done | failure-path durability, fail-closed dispatch, mid-tick cost cap, execute/commit boundary; failure-injection + property sweeps |
| **1 — Executor seam** | ✅ done | recorded-activity + idempotency; structured output + validate→repair; 4 real role-types (SCOUT/ORCHESTRATOR/ENGINEER/REVIEWER) across local + cloud; error taxonomy; real token→$ cost; cost-routing eval |
| **2 — Concurrency** | ✅ done | parallel execute batch (thread pool + literal `asyncio.gather`); distributed workers pulling the frontier; backpressure |
| **Eval — GSM8K** | ✅ done | four ablation arms — single-pass, SSA (refine), self-consistency (φ-merge), combined (φ-of-SSA) — reporting accuracy, \$/correct, generations, wall-clock, replay-during-eval, and a paired McNemar analysis (via an injectable goal predicate) |
| **3 — Persistence** | 🔴 next | Postgres `EventStore` adapter, snapshotting, resume-from-log (the `EventStore` ABC + recorded activities already lay the foundation) |

## What the demo shows (one run)

A goal — *"produce a vetted brief covering ingestion, reasoning, provenance,
scoring ≥ 0.80"* — drives:

1. the **orchestrator** emitting a subgraph (3 scouts → engineer → reviewer);
2. the **reviewer** scoring the first draft **0.60**, so it emits a *refinement*
   stratum (`engineer.g5`, a new SSA version fed the failed draft + critique, then
   `reviewer.g6`) — a loop with **no cycle**;
3. the refined draft scoring **0.90**, so the reviewer approves and emits a
   **synthesizer** that aggregates to the terminal brief and raises `GOAL`;
4. the **goal predicate** accepting it → `goal_satisfied` in 7 generations.

Then it verifies: a **back-edge is rejected** (acyclicity), a **lineage-less
write is rejected** (provenance-at-write), **replay reconstructs a bit-identical
graph** (durability), cost splits across **local vs cloud** tiers, and there are
**zero provenance gaps**.

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
phi.py                  φ-merge of competing branches: argmax / majority /     L4 L6
                        confidence-weighted / union; records the selector
tasks_gsm8k.py          GSM8K task: solver/checker/answer roles; single/ssa/   L3
                        sc/combined (adaptive φ-of-SSA via dynamic join)
steering.py             pluggable goal predicate, constraint filter, beam      L3
governor.py             caps + GLOBAL retry budget                             L7
scheduler.py            the loop: admit→execute→commit; batch / async /        L5
                        distributed workers; materializes dynamic joins
views.py                blackboard, replay verify, lineage + cost report       L4 L5
evaluation.py           eval harness: cost-routing + GSM8K reports             L5
```

## Plugging in real models

The model-backed roles live **once** in `roles.py` (ENGINEER / REVIEWER / ORCHESTRATOR
+ their prompts + the shared validate→repair loop), written against an abstract `chat`
capability. Each provider module supplies only a thin `chat` transport, so adding a
backend is a transport — not another copy of the roles. (The mock roles in
`executors.py` stay the reference bodies you'd start from.) Every model/tool/IO call is
wrapped as a *recorded activity* (`ctx.record(key, fn)`, stored as an `ACTIVITY_RECORDED`
event) so replay/retry never re-runs it — see `SPEC.md` §5.2 and §9.

Real executors ship for both tiers (thin transports over the shared roles):

* **Local** (`executors_ollama.py`): a retrieval **SCOUT** (ranks a corpus per subtopic
  — offline lexical default, embeddings-ready) + Ollama **ENGINEER** and **REVIEWER**.
* **Cloud** (`executors_openrouter.py`): OpenAI-compatible **ORCHESTRATOR**, **ENGINEER**,
  **REVIEWER** over a raw-`urllib` transport — **provider-agnostic** (OpenRouter default;
  point `base_url` at Ollama's `/v1`, vLLM, any skin). No SDK, so `max_retries=0` by
  construction; refusals/content-filter/4xx are **terminal** (never retried); per-model
  cost is the provider's own `usage.cost`.
* The **real ORCHESTRATOR** answers the recursive-patch problem: the model emits a *flat*
  plan and the executor assembles the subgraph deterministically, post-filling required
  subtopics so coverage is guaranteed.

All model calls are structured-output + validate/repair, recorded as replay-safe
exactly-once activities, with real token→$ cost. Run `python demo_ollama.py` (local
fleet) or `python eval_cost_routing.py` (the **cost-routing eval**: all-local vs
all-cloud vs type-aware). Both skip cleanly when a provider isn't available; the
offline `demo.py` / `test_invariants.py` never require any.

## GSM8K evaluation

The substrate pointed at a standard benchmark (`tasks_gsm8k.py`): grade-school math word
problems, graded by exact match on the final number. SOLVER / CHECKER / ANSWER roles run over
the same provider-agnostic `chat`, in four configurations that **ablate the two construction
primitives** — refinement (SSA) and merge (φ):

* **single-pass** — neither. One solver, no critic.
* **SSA-generational** — *refinement only*. A CHECKER verifies the reasoning and, when
  unconvinced, emits a *new generation* solver (`solver.g{k+1}`) fed the critique — iteration
  without a cycle.
* **self-consistency** — *merge only*. K solvers sample competing reasoning paths in parallel;
  a **φ-node** (`phi.py`) majority-votes their answers into one, recording the tally as the
  merge's provenance.
* **combined (φ-of-SSA)** — *both*. K **adaptive** lineages, each refined to its own critic's
  satisfaction (like SSA), then merged by a **verifier-weighted** φ (consensus scaled by the
  critics' confidence). The lineages have *data-dependent depth*, so they fan into a **dynamic
  join** (`JoinSpec`): the scheduler materializes the φ over the K certified answers once quorum
  arrives — deferred static fan-in, so acyclicity and replay hold with no monotonicity
  relaxation. This is the on-the-fly self-construction the compositional claim actually needs.

`eval_gsm8k.py` reports the five numbers the paper is built around — **accuracy**, **\$ per
correct answer** (the cost/quality frontier), mean **generations**, mean **wall-clock** (latency
under concurrency), and **replay-equivalence** holding live during the eval (even under
temperature-sampled self-consistency, because every sample is a recorded activity) — plus a
**paired McNemar** analysis over the *same* problems (the right test when every config runs the
identical task set: it exposes strict domination and whether a small accuracy gap is just noise).
The DAG never sees the gold answer — the goal predicate (`AnswerGoal`) only certifies that *an*
answer survived the critic or merge; correctness lives in the eval judge alone.

```bash
curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o test.jsonl
python eval_gsm8k.py --data test.jsonl --n 250 --configs single,ssa,sc,combined --k 5 --dump results.json
```

Provider is auto-detected (Ollama or `OPENROUTER_API_KEY`) and skips cleanly when absent; with
no `--data` it runs a tiny authored sample (plumbing smoke, not a benchmark number).

## Concurrency

The tick splits into **admit → execute → commit**, where `execute` is pure and the
only parallelizable phase; admit and commit stay serial and single-writer (the store's
`seq` is the serialization point). Four execution modes, all producing the same result:

* **Serial** — `Scheduler(...).run(goal)` (default; deterministic, offline).
* **Thread pool** — `Scheduler(max_workers=N)` runs the batch's sync executors
  concurrently (GIL releases during I/O); `N` is the real WIP cap.
* **Async** — an `async def` executor triggers a literal `asyncio.gather` over the
  batch (`ctx.record_async` for awaited steps).
* **Distributed workers** — `Scheduler(...).run_workers(n_workers, max_inflight)`
  inverts the loop to a **pull** model: workers pull ready nodes from a shared frontier
  queue and execute, while a single-writer coordinator admits + commits. **Backpressure**
  (`max_inflight`) paces admission to commit. The in-process asyncio version is the
  reference shape for real remote workers on a durable queue (Redis/SQS).

## Design commitments (from the approved decisions)

* **Subgraph emission is the only construction primitive** — no static fan-out,
  no edge DSL; structure is data an expander returns.
* **Agents are node-type executors**; a complex agent is a nested subgraph.
* **Blackboard = the DAG itself** — cards are nodes, WIP caps concurrency, "done"
  is a typed event (not a parsed token).
* **The log is the only source of truth**; the graph is a fold over it.

See `SPEC.md` for the full contracts, schemas, invariants, and the PoC→prod delta.

## License

Copyright (c) 2026 Karthik Naidu and Associated Researchers. All rights reserved.

This software is proprietary. Plutus CCE LLC is granted a non-exclusive,
non-transferable, non-sublicensable, limited license for internal use only,
subject to the terms and conditions in the [LICENSE](LICENSE) file.

All intellectual property rights are retained by Karthik Naidu and the
Associated Researchers listed as contributors to this project. For
licensing inquiries, contact: karthik.naidu.east@gmail.com.

