"""
evaluation.py — the cost-routing eval harness.

The measurable claim this exists to test (the workshop-paper contribution): *type-aware
local-vs-cloud routing on a self-constructing agent DAG achieves equal task success at
lower cost than all-cloud and all-local baselines.* The substrate already records real
per-node dollar cost and token usage (Layer 5), so an eval is: run a task set under
each routing policy and compare success rate against total cost.

A **policy** is just a registry factory (which executor runs each role). The harness is
provider-agnostic — it runs whatever registry it's handed and reads cost/success off the
event-sourced graph — so it's fully offline-testable with fake-transport registries, and
the live `eval_cost_routing.py` wires it to real Ollama / OpenRouter fleets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .core import TerminationReason
from .governor import Budget
from .scheduler import RunResult, Scheduler
from .store import SQLiteEventStore


@dataclass
class EvalTask:
    name: str
    goal: dict
    judge: Optional[Callable[[RunResult], bool]] = None   # default: reached the goal


@dataclass
class TaskResult:
    task: str
    policy: str
    success: bool
    reason: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    ticks: int
    nodes: int
    error: Optional[str] = None


# A small default task set (goals carry a corpus for the retrieval scout).
DEFAULT_TASKS = [
    EvalTask("enrichment", {
        "topic": "stateless semantic enrichment",
        "subtopics": ["ingestion", "reasoning", "provenance"], "target_score": 0.80,
        "corpus": [
            {"id": "ing", "text": "Ingestion streams parse and normalize raw records for the pipeline."},
            {"id": "rea", "text": "Reasoning applies inference rules to derive semantic facts."},
            {"id": "pro", "text": "Provenance stamps each fact with run id, batch, and source lineage."},
            {"id": "noise", "text": "Unrelated office logistics and snack budgets."}]}),
    EvalTask("observability", {
        "topic": "distributed tracing", "subtopics": ["spans", "sampling"], "target_score": 0.80,
        "corpus": [
            {"id": "sp", "text": "Spans capture a unit of work with start, end, and parent links."},
            {"id": "sa", "text": "Sampling decides which traces to keep to bound overhead."},
            {"id": "x", "text": "Irrelevant gardening notes."}]}),
]


def run_task(task: EvalTask, registry: dict, max_workers: int = 1,
             budget: Optional[Budget] = None) -> TaskResult:
    """Run one task under one registry (policy) and read metrics off the graph."""
    store = SQLiteEventStore(":memory:")
    try:
        r = Scheduler(store, registry, budget=budget or Budget(),
                      max_workers=max_workers).run(dict(task.goal))
    except Exception as exc:                       # provider down, etc. -> a failed run
        return TaskResult(task.name, "?", False, "error", 0.0, 0, 0, 0, 0, error=repr(exc))
    success = task.judge(r) if task.judge else (r.reason == TerminationReason.GOAL_SATISFIED)
    nodes = r.graph.nodes.values()
    cost = sum(rec.cost_usd or 0.0 for rec in nodes)
    ti = sum((rec.usage or {}).get("input_tokens", 0) for rec in nodes)
    to = sum((rec.usage or {}).get("output_tokens", 0) for rec in nodes)
    return TaskResult(task.name, "?", success, r.reason.value, round(cost, 6),
                      ti, to, r.ticks, len(r.graph.nodes))


def run_eval(tasks: list, policies: dict, max_workers: int = 1) -> dict:
    """policies: {name: () -> registry}. Returns {policy: {task_name: TaskResult}}."""
    results: dict = {}
    for pname, factory in policies.items():
        results[pname] = {}
        for task in tasks:
            tr = run_task(task, factory(), max_workers=max_workers)
            tr.policy = pname
            results[pname][task.name] = tr
    return results


def _policy_totals(results: dict) -> dict:
    """{policy: (successes, total, cost_usd, tok_in, tok_out)}."""
    totals = {}
    for pname, trs in results.items():
        rows = list(trs.values())
        totals[pname] = (
            sum(1 for t in rows if t.success), len(rows),
            round(sum(t.cost_usd for t in rows), 6),
            sum(t.tokens_in for t in rows), sum(t.tokens_out for t in rows))
    return totals


def savings_pct(results: dict, cheaper: str, dearer: str) -> Optional[float]:
    """Percent cost reduction of `cheaper` vs `dearer` (None if `dearer` is free)."""
    t = _policy_totals(results)
    if cheaper not in t or dearer not in t or t[dearer][2] <= 0:
        return None
    return round(100.0 * (t[dearer][2] - t[cheaper][2]) / t[dearer][2], 1)


def cost_routing_report(results: dict) -> str:
    """Success rate + cost per policy, and the cost saving of the cheapest fully
    -successful policy vs the most expensive one (the cost-routing claim, measured)."""
    totals = _policy_totals(results)
    lines = ["COST-ROUTING EVAL",
             f"  {'policy':<16}{'success':<12}{'cost $':<14}{'tokens in/out'}"]
    for pname, (succ, tot, cost, ti, to) in totals.items():
        lines.append(f"  {pname:<16}{f'{succ}/{tot}':<12}{f'{cost:.6f}':<14}{ti}/{to}")
    # Compare fully-successful policies by cost.
    full = {p: v for p, v in totals.items() if v[1] and v[0] == v[1]}
    if len(full) >= 2:
        cheapest = min(full, key=lambda p: full[p][2])
        dearest = max(full, key=lambda p: full[p][2])
        if full[dearest][2] > 0 and cheapest != dearest:
            pct = savings_pct(results, cheapest, dearest)
            lines.append("")
            lines.append(f"  equal success ({full[cheapest][0]}/{full[cheapest][1]}): "
                         f"'{cheapest}' costs {pct}% less than '{dearest}' "
                         f"(${full[cheapest][2]:.6f} vs ${full[dearest][2]:.6f})")
    return "\n".join(lines)
