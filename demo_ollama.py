"""
demo_ollama.py — the Phase 1 local slice, live.

Same goal as demo.py, but the ENGINEER is a real local model (Ollama) while every
other role stays a deterministic mock. It proves the executor seam end-to-end on the
same event-sourced substrate: a real model call, structured output, a validate->repair
loop, and the recorded-activity discipline (the model call is replayed on retry, not
re-run).

Prereqs: `pip install ollama`, `ollama serve`, and a model pulled
(default `llama3.2` — `ollama pull llama3.2`). Override the model with the first CLI arg:

    python demo_ollama.py llama3.2

Unlike demo.py this needs a running model, so it is NOT part of the offline test suite;
if Ollama isn't reachable it prints a skip notice and exits 0.
"""

from __future__ import annotations

import sys

from dyndag import Budget, Scheduler, SQLiteEventStore, lineage_report, verify_replay
from dyndag.executors_ollama import ollama_registry


def preflight(model: str) -> str | None:
    """Return an error string if Ollama+model can't be reached, else None."""
    try:
        import ollama
    except ImportError:
        return "ollama not installed — run: pip install ollama"
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": "ping"}],
                    options={"num_predict": 1})
    except Exception as exc:  # server down, model not pulled, etc.
        return (f"could not reach Ollama with model '{model}': {exc}\n"
                f"      start it with `ollama serve` and `ollama pull {model}`")
    return None


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"

    err = preflight(model)
    if err:
        print(f"[skip] {err}")
        return

    goal = {
        "topic": "stateless semantic enrichment",
        "subtopics": ["ingestion", "reasoning", "provenance"],
        "target_score": 0.80,
    }

    store = SQLiteEventStore(":memory:")
    sched = Scheduler(store=store, registry=ollama_registry(model=model), budget=Budget())
    result = sched.run(goal)

    print("=" * 74)
    print("PHASE 1 SLICE — real ENGINEER (Ollama), everything else mock")
    print("=" * 74)
    print(f"model              : {model}")
    print(f"termination        : {result.reason.value}")
    print(f"generations (ticks): {result.ticks}")
    print(f"nodes constructed  : {len(result.graph.nodes)}")
    if result.terminal_output:
        print(f"terminal brief     : score={result.terminal_output['final_score']} "
              f"covered={result.terminal_output['covered']}")

    # Show a genuinely model-generated section.
    for rec in result.graph.nodes.values():
        if rec.node_type.value == "engineer" and isinstance(rec.output, dict):
            sections = rec.output.get("sections", {})
            if sections:
                k = next(iter(sections))
                print(f"sample section     : [{k}] {str(sections[k])[:220]}")
                break

    # Recorded activities: the model calls, durable in the log.
    n_activities = sum(len(v) for v in result.graph.activities.values())
    print(f"recorded activities: {n_activities} (model calls captured for replay)")

    ok, live_h, rep_h = verify_replay(store, result.graph, result.run_id)
    print(f"replay identical   : {ok}")

    print()
    print(lineage_report(result.graph))


if __name__ == "__main__":
    main()
