"""
tasks_gsm8k.py — the GSM8K benchmark as a self-constructing DAG task.

GSM8K is grade-school math word problems, each graded by exact match on a single final
number. That maps onto the substrate with no new machinery beyond a task-specific goal
predicate (steering.AnswerGoal) and a small role fleet over the SAME provider-agnostic
`chat` seam the brief task uses (roles._LlmRole + ollama_chat / openrouter_chat):

  SOLVER   (bound to ENGINEER)     — reason step-by-step, emit a final answer
  CHECKER  (bound to REVIEWER)     — verify the reasoning (NO gold answer), approve or steer
  ANSWER   (bound to SYNTHESIZER)  — finalize the chosen solution, raise Signal.GOAL

Reusing the existing node *types* means the cost router's local/cloud split (ENGINEER local,
REVIEWER cloud) applies unchanged, so a GSM8K run is already cost-routed.

Two configurations ship here — the paper's core comparison:
  * single-pass          — orchestrator -> solver -> answer   (one shot, no critic)
  * SSA-generational     — orchestrator -> solver -> checker  (critic drives refinement,
                           each retry a NEW generation: solver.g{k+1}, never a cycle)
(The self-consistency configuration — K samples merged by a PHI node — arrives with the PHI
node type.)

Correctness is NEVER visible to the DAG: the goal predicate only certifies that an answer
was produced (and survived the critic); the gold label lives in the eval judge alone, so the
graph cannot 'cheat' toward termination.
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

from .core import NodeType, Signal
from .executors import Executor, ExecutionResult, default_registry
from .nodes import NodeSpec, SubgraphPatch
from .phi import PhiExecutor, majority_vote
from .roles import ChatJson, _LlmRole
from .schemas import SOLUTION_SCHEMA, VERIFY_SCHEMA
from .steering import AnswerGoal


# --------------------------------------------------------------------------- #
# Roles (model-backed, over the abstract `chat`)                              #
# --------------------------------------------------------------------------- #
class SolverRole(_LlmRole):
    """Reason step-by-step and emit a final answer (bound to ENGINEER — generation). On a
    refinement it is fed the prior attempt + the checker's critique. The model call is a
    recorded activity, so a retry/replay reuses the same sample."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        revision = params.get("revision", 0)
        question = params["question"]
        critique = params.get("critique")
        prior = next((v for v in inputs.values()
                      if isinstance(v, dict) and v.get("kind") == "solution"), None)
        messages = [
            {"role": "system", "content": (
                "You are a careful math problem solver. Work through the problem step by step, "
                "then state the final numeric answer. Put ONLY the number (no units, no words, "
                "no thousands separators) in final_answer. Respond with JSON only.")},
            {"role": "user", "content": json.dumps({
                "problem": question,
                "prior_attempt": prior.get("final_answer") if prior else None,
                "critique_to_address": critique})},
        ]
        recorded = ctx.record(f"solve_rev{revision}",
                              lambda: self.chat(messages, SOLUTION_SCHEMA, "solution",
                                                ("reasoning", "final_answer")))
        data, usage = recorded["content"], recorded.get("usage")
        return ExecutionResult(output={
            "kind": "solution", "question": question, "revision": revision,
            "reasoning": str(data.get("reasoning", "")),
            "final_answer": str(data.get("final_answer", "")).strip(),
        }, signal=Signal.CONTINUE, usage=usage)


class CheckerRole(_LlmRole):
    """Verify the solver's reasoning (bound to REVIEWER — critique). It does NOT see the gold
    answer; it judges self-consistency and reports confidence. Mirrors ReviewerRole's steering:
    verified-and-confident (or the revision cap reached) => emit the terminal ANSWER node
    (Signal.DONE); else emit a refined solver (SSA g+1) fed the critique + a fresh checker
    (Signal.NEEDS_REFINEMENT). Bounded convergence via the revision cap and the governor."""

    def __init__(self, chat: ChatJson, target_confidence: float = 0.9, max_revision: int = 3):
        super().__init__(chat)
        self.target_confidence = target_confidence
        self.max_revision = max_revision

    def run(self, inputs, params, ctx) -> ExecutionResult:
        solution = next(v for v in inputs.values()
                        if isinstance(v, dict) and v.get("kind") == "solution")
        sol_id = next(k for k, v in inputs.items() if v is solution)
        question = solution["question"]
        revision = solution.get("revision", 0)
        messages = [
            {"role": "system", "content": (
                "You are a strict math checker. Decide whether the solution's reasoning is "
                "correct and internally consistent for the problem. Return verified (bool), "
                "your confidence from 0.0 to 1.0, and a one-line critique naming any error to "
                "fix. Respond with JSON only.")},
            {"role": "user", "content": json.dumps({
                "problem": question, "reasoning": solution.get("reasoning", ""),
                "final_answer": solution.get("final_answer", "")})},
        ]
        recorded = ctx.record(f"check_{sol_id}",
                              lambda: self.chat(messages, VERIFY_SCHEMA, "verify",
                                                ("verified", "confidence")))
        data, usage = recorded["content"], recorded.get("usage")
        verified = bool(data.get("verified"))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        critique = str(data.get("critique", ""))

        # Approve when the critic is satisfied, or when the revision cap is hit (take the
        # best-so-far answer so the run terminates with a solution rather than starving).
        if (verified and confidence >= self.target_confidence) or revision >= self.max_revision:
            answer = NodeSpec(key="answer", node_type=NodeType.SYNTHESIZER,
                              input_keys=(sol_id, ctx.self_id), params={})
            return ExecutionResult(
                output={"verdict": "approve", "verified": verified,
                        "confidence": round(confidence, 3)},
                signal=Signal.DONE, score=confidence, usage=usage,
                patch=SubgraphPatch(added_nodes=[answer]))

        refined = NodeSpec(key="solver", node_type=NodeType.ENGINEER,
                           input_keys=(sol_id, ctx.self_id),
                           params={"question": question, "revision": revision + 1,
                                   "critique": critique})
        re_check = NodeSpec(key="checker", node_type=NodeType.REVIEWER,
                            input_keys=("solver",), params={})
        return ExecutionResult(
            output={"verdict": "reject", "verified": verified,
                    "confidence": round(confidence, 3), "critique": critique},
            signal=Signal.NEEDS_REFINEMENT, score=confidence, usage=usage,
            patch=SubgraphPatch(added_nodes=[refined, re_check]))


class AnswerExecutor(Executor):
    """Finalizer (bound to SYNTHESIZER — aggregation). Emits the terminal solution
    (Signal.GOAL); the scheduler then applies AnswerGoal. Pure: it forwards the single chosen
    solution among its inputs (the self-consistency path uses a PHI merge instead)."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        solution = next(v for v in inputs.values()
                        if isinstance(v, dict) and v.get("kind") == "solution")
        return ExecutionResult(output={
            "kind": "solution", "question": solution.get("question"),
            "final_answer": solution.get("final_answer", ""),
            "reasoning": solution.get("reasoning", ""),
            "revision": solution.get("revision", 0),
        }, signal=Signal.GOAL)


class GsmOrchestrator(Executor):
    """The GSM8K expander (bound to ORCHESTRATOR). Deterministic — no model, no cost: GSM8K
    planning is fixed. Reads the question from the goal and emits the initial subgraph for the
    chosen `mode`:
      single => solver -> answer                   (one pass, no critic)
      ssa    => solver -> checker                  (the checker drives the SSA refinement loop)
      sc     => solver_0..solver_{k-1} -> phi       (self-consistency: K samples, φ majority-vote)
    """

    def __init__(self, mode: str = "ssa", k: int = 5):
        if mode not in ("single", "ssa", "sc"):
            raise ValueError(f"unknown GSM8K mode {mode!r} (expected 'single', 'ssa' or 'sc')")
        self.mode = mode
        self.k = max(1, int(k))

    def run(self, inputs, params, ctx) -> ExecutionResult:
        goal = ctx.goal if isinstance(ctx.goal, dict) else {}
        question = goal.get("question", "")

        if self.mode == "sc":
            # K sibling solvers (distinct ids => K independent, individually-recorded samples
            # => diverse under temperature, yet replay-stable) fan into one φ-merge.
            solvers = [NodeSpec(key=f"solver_{i}", node_type=NodeType.ENGINEER,
                                input_keys=(ctx.self_id,),
                                params={"question": question, "revision": 0, "sample": i})
                       for i in range(self.k)]
            phi = NodeSpec(key="phi", node_type=NodeType.PHI,
                           input_keys=tuple(s.key for s in solvers), params={})
            return ExecutionResult(
                output={"mode": "sc", "plan": [s.key for s in solvers] + ["phi"]},
                signal=Signal.CONTINUE, patch=SubgraphPatch(added_nodes=[*solvers, phi]))

        solver = NodeSpec(key="solver", node_type=NodeType.ENGINEER,
                          input_keys=(ctx.self_id,),
                          params={"question": question, "revision": 0})
        if self.mode == "single":
            tail = NodeSpec(key="answer", node_type=NodeType.SYNTHESIZER,
                            input_keys=("solver",), params={})
        else:  # ssa
            tail = NodeSpec(key="checker", node_type=NodeType.REVIEWER,
                            input_keys=("solver",), params={})
        return ExecutionResult(
            output={"mode": self.mode, "plan": ["solver", tail.key]},
            signal=Signal.CONTINUE, patch=SubgraphPatch(added_nodes=[solver, tail]))


# --------------------------------------------------------------------------- #
# Registry (policy) factory                                                   #
# --------------------------------------------------------------------------- #
def gsm8k_registry(chat: ChatJson, mode: str = "ssa", k: int = 5,
                   target_confidence: float = 0.9, max_revision: int = 3) -> dict:
    """A registry (eval policy) for GSM8K over a provider-agnostic `chat`. SOLVER=ENGINEER
    (local tier), CHECKER=REVIEWER (cloud tier) — so the cost router's split applies for free.
    `mode` selects single-pass, SSA-generational, or self-consistency ('sc', `k` samples merged
    by a φ-node majority-vote). GOAL/SCOUT stay mock (GSM8K emits no SCOUT)."""
    registry = default_registry()
    registry[NodeType.ORCHESTRATOR] = GsmOrchestrator(mode=mode, k=k)
    registry[NodeType.ENGINEER] = SolverRole(chat)
    registry[NodeType.REVIEWER] = CheckerRole(chat, target_confidence=target_confidence,
                                              max_revision=max_revision)
    registry[NodeType.SYNTHESIZER] = AnswerExecutor()
    # φ-merge over the K competing solver samples: majority-vote on final_answer, terminal.
    registry[NodeType.PHI] = PhiExecutor(majority_vote("final_answer"),
                                         signal=Signal.GOAL, kind="solution")
    return registry


# --------------------------------------------------------------------------- #
# Data + judge (exact match on the gold final number)                         #
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(s) -> str:
    """The last number in a string — thousands separators removed and a trailing '.0'
    collapsed ('1,024' -> '1024', '18.0' -> '18'). Models routinely wrap the answer in prose
    ('the result is 7.', 'times(6, 2) = 12'), so grading keys on the FINAL numeric token: the
    standard GSM8K extraction, kept robust here by asking for a structured final_answer field
    first. Returns '' when there is no number."""
    nums = _NUM.findall(str(s))
    if not nums:
        return ""
    t = nums[-1].replace(",", "")
    if re.fullmatch(r"-?\d+\.0+", t):
        t = t.split(".")[0]
    return t


def _extract_pred(terminal_output) -> str:
    """The predicted answer as a normalized number: the last number in final_answer, falling
    back to the last number in the reasoning when the model left final_answer non-numeric."""
    if not isinstance(terminal_output, dict):
        return ""
    return (_last_number(terminal_output.get("final_answer", ""))
            or _last_number(terminal_output.get("reasoning", "")))


def _gold_of(answer: str) -> str:
    """GSM8K gold = the text after the final '####' in the answer field."""
    return answer.rsplit("####", 1)[-1].strip()


def _make_judge(gold: str):
    """A judge closure over the gold label: correct iff the run produced a terminal answer
    whose final number equals the gold number. The DAG never sees `gold`."""
    g = _last_number(gold)

    def judge(r) -> bool:
        return (r.terminal_output is not None and g != ""
                and _extract_pred(r.terminal_output) == g)

    return judge


def load_gsm8k(path: str, n: Optional[int] = None, seed: Optional[int] = None) -> list:
    """Load GSM8K test problems from a JSONL file (lines of {question, answer}, the answer
    ending in '#### <n>') into EvalTasks carrying an exact-match judge + an AnswerGoal
    predicate. `n` (with optional `seed`) takes a random subset — the standard way to bound a
    workshop run. The dataset is user-downloaded (MIT-licensed) and not bundled here."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [_task_from_row(i, row) for i, row in enumerate(rows)]


def _task_from_row(i: int, row: dict):
    from .evaluation import EvalTask   # local import: tasks_gsm8k is a leaf, evaluation isn't
    return EvalTask(name=f"gsm{i}", goal={"question": row["question"], "problem_id": i},
                    judge=_make_judge(_gold_of(row["answer"])), goal_predicate=AnswerGoal())


# A tiny, AUTHORED arithmetic set (NOT GSM8K rows) for offline/hermetic tests and demos —
# enough to exercise the harness without a model, a network, or the dataset. Phrased so a
# trivial regex solver can answer them, keeping offline tests deterministic.
_SAMPLE = [
    ("Compute 3 plus 4.", "7"),
    ("Compute 12 minus 5.", "7"),
    ("Compute 6 times 2.", "12"),
    ("Compute 10 plus 15.", "25"),
    ("Compute 20 minus 8.", "12"),
    ("Compute 9 plus 9.", "18"),
]


def sample_tasks() -> list:
    """The authored arithmetic set as EvalTasks (exact-match judge + AnswerGoal). Offline,
    hermetic, no GSM8K data required."""
    from .evaluation import EvalTask
    return [EvalTask(name=f"s{i}", goal={"question": q, "problem_id": i},
                     judge=_make_judge(g), goal_predicate=AnswerGoal())
            for i, (q, g) in enumerate(_SAMPLE)]
