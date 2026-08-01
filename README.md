# dyndag — dynamic-DAG multi-agent orchestrator (PoC)

A runnable proof of concept for a multi-agent system whose task graph
**constructs itself on the fly**: expander nodes emit subgraphs computed as
downstream variables of prior nodes, a critic steers construction toward a target
outcome, and the whole artifact stays acyclic because iteration is expressed as
new generations (SSA), never cycles.

It is the reference implementation of `SPEC.md`, which formalizes the seven
approved layers. No external services and no network — SQLite stands in for
Postgres, deterministic executors stand in for models, so it runs anywhere.

## Quickstart

```bash
python demo.py            # end-to-end run + all guards, with narrated output
python test_invariants.py # focused checks of the load-bearing guarantees
```

Both exit 0 and assert their invariants.

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
core.py       ids, enums, Provenance (provenance-at-write), Event      L1 L5
store.py      append-only log + provenance guard  (Postgres analog)    L5
graph.py      fold-of-log projection, ready-set, acyclicity, lineage   L1 L6
nodes.py      NodeSpec, SubgraphPatch, placeholder resolution          L2
executors.py  role executors, typed signals, ModelRouter (cost)        L3 L4 L5
steering.py   goal predicate, constraint filter, beam, value fn        L3
governor.py   caps + GLOBAL retry budget                               L7
scheduler.py  the tick loop where every layer meets                    L5
views.py      blackboard (Kanban), replay verify, lineage report       L4 L5
```

## Plugging in real models

Replace the body of any `Executor.run` in `executors.py` with a model call. The
scheduler, provenance, steering, durability, and governance layers are unchanged.
Wrap each model/tool/IO call as a *recorded activity* (`ctx.record(key, fn)`, stored
as an `ACTIVITY_RECORDED` event) so replay/retry never re-runs it — see `SPEC.md`
§5.2 and §9.

Real executors ship for both tiers:

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
exactly-once activities, with real token→$ cost. Batches run concurrently under
`Scheduler(max_workers=N)`. Run `python demo_ollama.py` (local fleet) or
`python eval_cost_routing.py` (the **cost-routing eval**: all-local vs all-cloud vs
type-aware). Both skip cleanly when a provider isn't available; the offline
`demo.py` / `test_invariants.py` never require any.

## Design commitments (from the approved decisions)

* **Subgraph emission is the only construction primitive** — no static fan-out,
  no edge DSL; structure is data an expander returns.
* **Agents are node-type executors**; a complex agent is a nested subgraph.
* **Blackboard = the DAG itself** — cards are nodes, WIP caps concurrency, "done"
  is a typed event (not a parsed token).
* **The log is the only source of truth**; the graph is a fold over it.

See `SPEC.md` for the full contracts, schemas, invariants, and the PoC→prod delta.
