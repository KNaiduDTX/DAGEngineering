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

A worked fleet ships in `executors_ollama.py`: a real retrieval **SCOUT** (ranks a
corpus per subtopic — offline lexical default, embeddings-ready) plus real Ollama
**ENGINEER** and **REVIEWER** with structured output, a validate/repair loop, recorded
activities (replay-safe, exactly-once), and real token→$ cost; GOAL/SYNTHESIZER stay
mock. With a real critic the score is a genuine model judgment and the loop converges
for real (or terminates on the generation cap). Batches run concurrently under
`Scheduler(max_workers=N)`. Run it with `python demo_ollama.py` (needs `pip install
ollama` + a pulled model); it prints a skip notice if Ollama isn't reachable. The
offline `demo.py` / `test_invariants.py` never require it.

## Design commitments (from the approved decisions)

* **Subgraph emission is the only construction primitive** — no static fan-out,
  no edge DSL; structure is data an expander returns.
* **Agents are node-type executors**; a complex agent is a nested subgraph.
* **Blackboard = the DAG itself** — cards are nodes, WIP caps concurrency, "done"
  is a typed event (not a parsed token).
* **The log is the only source of truth**; the graph is a fold over it.

See `SPEC.md` for the full contracts, schemas, invariants, and the PoC→prod delta.
