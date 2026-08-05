"""
tasks_math.py — the MATH (Hendrycks) competition-math benchmark as a self-constructing DAG task.

Unlike GSM8K, MATH answers are LaTeX expressions (\\frac, \\sqrt, tuples, degrees, ...), so
grading is EQUIVALENCE, not exact match. This module supplies:
  * a validated equivalence grader — a port of the canonical Hendrycks `is_equiv` normalization,
    plus \\boxed{} extraction from the model's answer;
  * a JSONL loader (problem + gold `answer` + level/subject);
  * MATH-framed solver/checker prompts, wired to the SAME roles as GSM8K (tasks_gsm8k) — only the
    prompt and the judge differ, so the substrate, the four arms, and the paired analysis are
    reused verbatim.

The φ merge (self-consistency / combined) votes on the NORMALIZED answer, so equivalent LaTeX
forms aggregate rather than splitting the vote — a fairness fix specific to LaTeX answers.

The DAG still never sees the gold: the goal predicate certifies an answer was produced;
correctness (equivalence to the gold) lives only in the eval judge.
"""

from __future__ import annotations

import json
import random
from typing import Optional

from .core import NodeType, Signal
from .executors import default_registry
from .phi import (PhiExecutor, confidence_weighted_vote, greedy_anchored_select, majority_vote)
from .roles import ChatJson
from .steering import AnswerGoal
from .tasks_gsm8k import AnswerExecutor, CheckerRole, GsmOrchestrator, SolverRole


# --------------------------------------------------------------------------- #
# Equivalence grader — a port of Hendrycks et al.'s MATH `is_equiv`            #
# --------------------------------------------------------------------------- #
def _last_boxed(s: str) -> Optional[str]:
    """The content of the LAST \\boxed{...} (balanced braces), or None if there is no box."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(s) and s[i] != "{":
        i += 1
    if i >= len(s):
        return None
    depth, start = 0, i
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
    return None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr and substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + substr[2:]
            else:
                new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ia, ib = int(a), int(b)
        if string != f"{ia}/{ib}":
            return string
        sign = "-" if (ia < 0) != (ib < 0) else ""   # pull the sign OUT so -1/2 == -\frac{1}{2}
        return sign + "\\frac{" + str(abs(ia)) + "}{" + str(abs(ib)) + "}"
    except ValueError:
        return string


def _remove_right_units(string):
    return string.split("\\text{ ")[0] if "\\text{ " in string else string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    parts = string.split("\\sqrt")
    new_string = parts[0]
    for split in parts[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    """Canonical MATH answer normalization (Hendrycks): collapse LaTeX spacing / sizing / fraction
    / sqrt / units variants to a comparable form."""
    string = str(string).replace("\n", "")
    string = string.replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")     # drop spaces BEFORE sqrt-fixing so '\\sqrt 3' -> '\\sqrt3'
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _num_clean(s: str) -> str:
    """Strip currency / percent / degree / spacing so a plain number can be parsed."""
    return (str(s).replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
            .replace("^\\circ", "").replace("^{\\circ}", "").replace(",", "").replace(" ", ""))


def _to_float(s: str):
    try:
        return float(_num_clean(s))
    except (ValueError, TypeError):
        return None


def is_equiv(a: Optional[str], b: Optional[str]) -> bool:
    """True iff two MATH answers are equivalent: canonical string normalization, then a numeric
    fallback for pure decimals (so 18.9 == 18.90, 0.5 == 0.50, $18.90 == 18.9) that the string
    path misses. The numeric path fires only when BOTH sides parse as plain numbers, so it can't
    false-positive on expressions."""
    if a is None or b is None:
        return a is None and b is None
    try:
        if _strip_string(a) == _strip_string(b):
            return True
        fa, fb = _to_float(a), _to_float(b)
        if fa is not None and fb is not None:
            return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))
        return False
    except Exception:
        return str(a) == str(b)


def _extract_pred(terminal_output) -> str:
    """The model's answer: the LAST \\boxed{} content if it wrapped one (despite instructions),
    else the raw final_answer, else a boxed answer found in the reasoning."""
    if not isinstance(terminal_output, dict):
        return ""
    fa = str(terminal_output.get("final_answer", "")).strip()
    boxed = _last_boxed(fa)
    if boxed is not None:
        return boxed
    if fa:
        return fa
    boxed_r = _last_boxed(str(terminal_output.get("reasoning", "")))
    return boxed_r if boxed_r is not None else ""


def _make_judge(gold: str):
    """A judge closure over the gold LaTeX answer: correct iff the terminal answer is equivalent.
    The DAG never sees `gold`."""
    g = str(gold)

    def judge(r) -> bool:
        return r.terminal_output is not None and is_equiv(_extract_pred(r.terminal_output), g)

    return judge


# --------------------------------------------------------------------------- #
# Data loader                                                                  #
# --------------------------------------------------------------------------- #
def load_math(path: str, n: Optional[int] = None, seed: Optional[int] = None,
              min_level: Optional[int] = None) -> list:
    """Load MATH problems from a JSONL ({problem, answer, level, subject}) into EvalTasks with the
    equivalence judge + AnswerGoal. `min_level` keeps only problems at or above a difficulty (1-5);
    `n`/`seed` take a random subset."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if min_level is None or int(r.get("level", 0) or 0) >= min_level:
                rows.append(r)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [_task_from_row(i, r) for i, r in enumerate(rows)]


def _task_from_row(i: int, row: dict):
    from .evaluation import EvalTask   # local import: tasks_math is a leaf, evaluation isn't
    return EvalTask(name=f"math{i}",
                    goal={"question": row["problem"], "problem_id": i,
                          "level": row.get("level"), "subject": row.get("subject")},
                    judge=_make_judge(row["answer"]), goal_predicate=AnswerGoal())


# --------------------------------------------------------------------------- #
# MATH prompts + registry (same roles as GSM8K, different instruction + judge) #
# --------------------------------------------------------------------------- #
MATH_SOLVER_SYS = (
    "You are an expert competition mathematician. Solve the problem step by step, then give the "
    "final answer. In final_answer put ONLY the final answer as a compact LaTeX expression with no "
    "surrounding text and no \\boxed — e.g. \\frac{3}{2}, 2\\sqrt{5}, (3,\\frac{\\pi}{2}), 7, 145. "
    "Respond with JSON only.")
MATH_CHECKER_SYS = (
    "You are an independent competition-math verifier. Solve the problem YOURSELF, from scratch, "
    "with brief reasoning — do NOT assume any prior attempt is correct. Put ONLY your final answer "
    "as a compact LaTeX expression (no surrounding text, no \\boxed) in independent_answer, and "
    "your confidence from 0.0 to 1.0 that it is correct. Respond with JSON only.")


def math_registry(chat: ChatJson, mode: str = "ssa", k: int = 5,
                  critic_chat: Optional[ChatJson] = None,
                  target_confidence: float = 0.9, max_revision: int = 3,
                  explore_chat: Optional[ChatJson] = None, margin: float = 1.0) -> dict:
    """A registry (eval policy) for MATH — identical structure to gsm8k_registry, but MATH-framed
    prompts and (crucially) a φ merge that operates on the NORMALIZED answer so equivalent LaTeX
    forms aggregate. `mode`: single / ssa / sc / combined (φ-of-SSA vote) / combined_v2 (greedy
    -anchored SELECT — never dilutes below single). `explore_chat` is the temp>0 chat for
    combined_v2's exploratory lineages (its anchor stays on the greedy `chat`)."""
    registry = default_registry()
    registry[NodeType.ORCHESTRATOR] = GsmOrchestrator(mode=mode, k=k)
    registry[NodeType.ENGINEER] = SolverRole(chat, system=MATH_SOLVER_SYS, explore_chat=explore_chat)
    registry[NodeType.REVIEWER] = CheckerRole(critic_chat or chat, target_confidence=target_confidence,
                                              max_revision=max_revision, system=MATH_CHECKER_SYS,
                                              equiv=is_equiv)
    registry[NodeType.SYNTHESIZER] = AnswerExecutor()
    if mode == "combined_v2":
        policy = greedy_anchored_select("final_answer", norm=_strip_string, margin=margin)
    elif mode == "combined":
        policy = confidence_weighted_vote("final_answer", norm=_strip_string)
    else:
        policy = majority_vote("final_answer", norm=_strip_string)
    registry[NodeType.PHI] = PhiExecutor(policy, signal=Signal.GOAL, kind="solution")
    return registry
