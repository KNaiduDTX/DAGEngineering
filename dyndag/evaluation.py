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

import time
from dataclasses import dataclass
from math import comb
from typing import Callable, Optional

from .core import NodeType, TerminationReason
from .governor import Budget
from .scheduler import RunResult, Scheduler
from .steering import GoalPredicate
from .store import SQLiteEventStore
from .views import verify_replay


@dataclass
class EvalTask:
    name: str
    goal: dict
    judge: Optional[Callable[[RunResult], bool]] = None   # default: reached the goal
    goal_predicate: Optional[GoalPredicate] = None        # None => scheduler's CoverageGoal


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
    generations: int = 0        # SOLVER/ENGINEER attempts — the SSA-specific metric
    wall_s: float = 0.0         # wall-clock of the run (latency under concurrency)
    replay_ok: bool = True      # INV-5.2 verified live for this run (False on a crashed run)
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
    """Run one task under one registry (policy) and read metrics off the graph. The task's
    goal_predicate (e.g. GSM8K's AnswerGoal) is threaded into the scheduler; None keeps the
    scheduler's default CoverageGoal. Wall-clock and replay-equivalence are measured here."""
    store = SQLiteEventStore(":memory:")
    t0 = time.perf_counter()
    try:
        r = Scheduler(store, registry, budget=budget or Budget(), max_workers=max_workers,
                      goal_predicate=task.goal_predicate).run(dict(task.goal))
    except Exception as exc:                       # provider down, etc. -> a failed run
        return TaskResult(task.name, "?", False, "error", 0.0, 0, 0, 0, 0,
                          wall_s=round(time.perf_counter() - t0, 4), replay_ok=False,
                          error=repr(exc))
    wall_s = round(time.perf_counter() - t0, 4)
    success = task.judge(r) if task.judge else (r.reason == TerminationReason.GOAL_SATISFIED)
    nodes = r.graph.nodes.values()
    cost = sum(rec.cost_usd or 0.0 for rec in nodes)
    ti = sum((rec.usage or {}).get("input_tokens", 0) for rec in nodes)
    to = sum((rec.usage or {}).get("output_tokens", 0) for rec in nodes)
    # generations = number of SOLVER/ENGINEER attempts: how many times the graph re-derived
    # the answer (the SSA-specific metric). replay = INV-5.2, verified live per run.
    generations = sum(1 for rec in nodes if rec.node_type == NodeType.ENGINEER)
    replay_ok, _, _ = verify_replay(store, r.graph, r.run_id)
    return TaskResult(task.name, "?", success, r.reason.value, round(cost, 6),
                      ti, to, r.ticks, len(r.graph.nodes),
                      generations=generations, wall_s=wall_s, replay_ok=replay_ok)


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


# --------------------------------------------------------------------------- #
# GSM8K report — accuracy vs the cost/quality frontier (the paper table)      #
# --------------------------------------------------------------------------- #
def accuracy(results: dict, policy: str) -> tuple:
    """(correct, total) for a policy — success is the task's exact-match judge."""
    rows = list(results[policy].values())
    return sum(1 for t in rows if t.success), len(rows)


def cost_per_correct(results: dict, policy: str) -> Optional[float]:
    """Total $ spent divided by correct answers — the cost/quality frontier point. None when
    the policy got nothing right (cost per correct is undefined)."""
    rows = list(results[policy].values())
    correct = sum(1 for t in rows if t.success)
    if correct == 0:
        return None
    return round(sum(t.cost_usd for t in rows) / correct, 6)


def gsm8k_report(results: dict) -> str:
    """The paper's evaluation table: per policy — accuracy, total cost, **cost per correct
    answer** (the frontier), mean **generations** (does SSA iteration help?), mean **wall
    clock** (latency under concurrency), and **replay-equivalence** over the runs that
    executed (INV-5.2 holding in practice, live during the eval)."""
    lines = ["GSM8K EVAL",
             f"  {'policy':<16}{'accuracy':<12}{'cost $':<12}{'$/correct':<12}"
             f"{'gen(avg)':<10}{'wall s(avg)':<13}{'replay'}"]
    for pname, trs in results.items():
        rows = list(trs.values())
        ran = [t for t in rows if t.error is None]         # exclude crashed runs (no graph)
        corr, tot = accuracy(results, pname)
        cost = sum(t.cost_usd for t in rows)
        cpc = cost_per_correct(results, pname)
        cpc_s = f"{cpc:.6f}" if cpc is not None else "n/a"
        gen_avg = (sum(t.generations for t in ran) / len(ran)) if ran else 0.0
        wall_avg = (sum(t.wall_s for t in ran) / len(ran)) if ran else 0.0
        rok = sum(1 for t in ran if t.replay_ok)
        lines.append(f"  {pname:<16}{f'{corr}/{tot}':<12}{f'{cost:.6f}':<12}{cpc_s:<12}"
                     f"{gen_avg:<10.2f}{wall_avg:<13.4f}{rok}/{len(ran)} ok")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Paired analysis — the right test when every policy ran the SAME problems     #
# --------------------------------------------------------------------------- #
def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant counts (b, c): under H0 each discordant
    pair is a fair coin, so p = 2·P(X ≤ min(b,c) | Binom(b+c, 0.5)), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(results: dict, a: str, b: str) -> tuple:
    """Paired McNemar for two policies over the SHARED task set. Returns (a_only, b_only, p):
    a_only = tasks `a` got right and `b` wrong, b_only the reverse (the discordant pairs — the
    only ones that carry signal), and p the exact two-sided p-value. Far more powerful than
    comparing marginal accuracy rates, because it controls for per-problem difficulty."""
    ta, tb = results[a], results[b]
    keys = [k for k in ta if k in tb]
    a_only = sum(1 for k in keys if ta[k].success and not tb[k].success)
    b_only = sum(1 for k in keys if tb[k].success and not ta[k].success)
    return a_only, b_only, _mcnemar_p(a_only, b_only)


def paired_report(results: dict) -> str:
    """Pairwise paired comparison (McNemar) across all policies on the shared task set, with a
    note when one policy's correct set strictly contains the other's (strict domination)."""
    pols = list(results)
    lines = ["PAIRED COMPARISON (McNemar on shared tasks; only discordant pairs count)"]
    for i in range(len(pols)):
        for j in range(i + 1, len(pols)):
            a, b = pols[i], pols[j]
            ao, bo, p = mcnemar(results, a, b)
            dom = ""
            if ao == 0 and bo > 0:
                dom = f"   [{b} strictly dominates {a}]"
            elif bo == 0 and ao > 0:
                dom = f"   [{a} strictly dominates {b}]"
            lines.append(f"  {a:<10} vs {b:<10}  {a}-only={ao:<3} {b}-only={bo:<3} p={p:.4f}{dom}")
    return "\n".join(lines)


def per_problem_rows(results: dict) -> list:
    """Flatten to per-(policy, task) rows for dumping — the raw material for a paired analysis
    or an audit of which problems each policy got right."""
    rows = []
    for pname, trs in results.items():
        for t in trs.values():
            rows.append({"policy": pname, "task": t.task, "success": t.success,
                         "reason": t.reason, "generations": t.generations,
                         "cost_usd": t.cost_usd, "wall_s": t.wall_s,
                         "replay_ok": t.replay_ok, "error": t.error})
    return rows
