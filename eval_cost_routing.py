"""
eval_cost_routing.py — the cost-routing claim, measured live.

Runs the default task set under three routing policies and prints success-vs-cost:

  * all-local   — every chat role on a local model (Ollama's OpenAI-compatible /v1)
  * all-cloud   — every chat role on OpenRouter
  * type-aware  — cheap/high-fan-out roles local (scout retrieval, engineer), outcome
                  -gating roles cloud (orchestrator, reviewer)

All three use the SAME provider-agnostic OpenAI-compatible executors — only base_url +
model differ — so the comparison is apples-to-apples. Real per-node dollar cost comes
off the event-sourced graph (OpenRouter reports usage.cost; local is ~free).

The measurable claim: type-aware matches all-cloud's task success at a fraction of the
cost, while beating all-local on success (a weak local critic converges less reliably).
This script needs Ollama running and/or an OPENROUTER_API_KEY; it skips any policy whose
provider isn't available and reports what it can.

Usage:
    OPENROUTER_API_KEY=... python eval_cost_routing.py [local_model] [cloud_model]
"""

from __future__ import annotations

import os
import sys

from dyndag import (DEFAULT_TASKS, NodeType, RetrievalScoutExecutor, cost_routing_report,
                    default_registry, run_eval)
from dyndag.roles import EngineerRole, OrchestratorRole, ReviewerRole
from dyndag.executors_openrouter import openrouter_chat

OLLAMA_V1 = "http://localhost:11434/v1"
OPENROUTER_V1 = "https://openrouter.ai/api/v1"


def _chat_fleet(base_url, model, api_key):
    """A registry: real retrieval SCOUT + ORCHESTRATOR/ENGINEER/REVIEWER over one
    OpenAI-compatible `chat` (one endpoint). GOAL/SYNTHESIZER stay mock."""
    chat = openrouter_chat(model=model, base_url=base_url, api_key=api_key)
    reg = default_registry()
    reg[NodeType.SCOUT] = RetrievalScoutExecutor()
    reg[NodeType.ORCHESTRATOR] = OrchestratorRole(chat)
    reg[NodeType.ENGINEER] = EngineerRole(chat)
    reg[NodeType.REVIEWER] = ReviewerRole(chat)
    return reg


def _type_aware(local_model, cloud_model, cloud_key):
    """Cheap/high-fan-out roles local, outcome-gating roles cloud — the same roles, wired
    to different `chat` backends by base_url."""
    local = openrouter_chat(model=local_model, base_url=OLLAMA_V1, api_key="ollama")
    cloud = openrouter_chat(model=cloud_model, base_url=OPENROUTER_V1, api_key=cloud_key)
    reg = default_registry()
    reg[NodeType.SCOUT] = RetrievalScoutExecutor()          # local retrieval (no chat)
    reg[NodeType.ENGINEER] = EngineerRole(local)
    reg[NodeType.ORCHESTRATOR] = OrchestratorRole(cloud)
    reg[NodeType.REVIEWER] = ReviewerRole(cloud)
    return reg


def _ollama_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def main() -> None:
    local_model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"
    cloud_model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-4o-mini"
    key = os.environ.get("OPENROUTER_API_KEY")

    have_local = _ollama_up()
    have_cloud = bool(key)

    policies = {}
    if have_local:
        policies["all-local"] = lambda: _chat_fleet(OLLAMA_V1, local_model, "ollama")
    if have_cloud:
        policies["all-cloud"] = lambda: _chat_fleet(OPENROUTER_V1, cloud_model, key)
    if have_local and have_cloud:
        policies["type-aware"] = lambda: _type_aware(local_model, cloud_model, key)

    if not policies:
        print("[skip] no providers available.")
        print(f"       Ollama at localhost:11434 reachable: {have_local}")
        print(f"       OPENROUTER_API_KEY set: {have_cloud}")
        print("       Start Ollama (`ollama serve` + `ollama pull llama3.2`) and/or set "
              "OPENROUTER_API_KEY, then re-run.")
        return

    print(f"policies: {list(policies)}  | local={local_model} cloud={cloud_model}\n")
    results = run_eval(DEFAULT_TASKS, policies)
    print(cost_routing_report(results))
    print()
    for pname, tasks in results.items():
        for t in tasks.values():
            note = f" error={t.error}" if t.error else ""
            print(f"  {pname:<12} {t.task:<14} success={t.success} reason={t.reason} "
                  f"cost=${t.cost_usd:.6f} ticks={t.ticks}{note}")


if __name__ == "__main__":
    main()
