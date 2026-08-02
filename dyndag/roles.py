"""
roles.py — provider-agnostic LLM role executors.

The role *logic* — how the ENGINEER drafts, how the REVIEWER scores + steers, how the
ORCHESTRATOR plans and assembles the subgraph — is identical no matter which model
backend runs it. It depends only on a `chat` capability:

    chat(messages, schema, name, required) -> {"content": <dict>, "usage": <dict>}

which raises MalformedResultError (retryable, repair exhausted) or TerminalExecutionError
(refusal). Each provider (native Ollama, OpenAI-compatible/OpenRouter) supplies its own
`chat` via a thin transport; these roles are written **once** here. Previously the
ENGINEER/REVIEWER bodies were duplicated in both provider modules — this module is the
single source of truth for that logic (and the prompts).

`chat_json_with_repair` is the shared validate->repair loop over a provider's single-call
primitive. `RetrievalScoutExecutor` is real retrieval with a pluggable ranker — provider
-neutral (no model), so it lives with the other roles rather than in a provider module.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .core import NodeType, Signal
from .executors import ExecutionResult, Executor, MalformedResultError
from .nodes import NodeSpec, SubgraphPatch
from .schemas import DRAFT_SCHEMA, PLAN_SCHEMA, REVIEW_SCHEMA

# A provider's single structured call: (messages, schema, name) -> (content_str, usage).
# May raise TerminalExecutionError (refusal). `name` is used by OpenAI-style backends
# (the json_schema name) and ignored by native ones.
ChatOnce = Callable[[list, dict, str], tuple]
# The role-facing capability: adds the validate->repair loop, returns parsed content.
ChatJson = Callable[[list, dict, str, tuple], dict]


def chat_json_with_repair(call_once: ChatOnce, messages: list, schema: dict, name: str,
                          required: tuple, repair_attempts: int) -> dict:
    """Shared validate->repair loop over a provider's single-call primitive: call the
    model, parse JSON, check the required keys, and on failure re-prompt with the bad
    output. Returns {"content": dict, "usage": {...}} or raises MalformedResultError
    (retryable). A TerminalExecutionError from `call_once` (a refusal) propagates
    unchanged — this loop only catches parse/validation errors."""
    convo = list(messages)
    last_err = ""
    for _ in range(repair_attempts + 1):
        content, usage = call_once(convo, schema, name)
        try:
            data = json.loads(content or "")
            if not isinstance(data, dict) or any(k not in data for k in required):
                raise ValueError(f"missing required keys {required}")
            return {"content": data, "usage": usage}
        except (json.JSONDecodeError, ValueError, TypeError) as err:
            last_err = str(err)
            convo = convo + [
                {"role": "assistant", "content": content or ""},
                {"role": "user", "content": (
                    f"That was not valid per the schema ({last_err}). "
                    f"Return ONLY a JSON object matching the schema.")},
            ]
    raise MalformedResultError(
        f"no schema-valid output in {repair_attempts + 1} attempts: {last_err}")


class _LlmRole(Executor):
    """Base for a model-backed role: holds the provider-agnostic `chat` capability."""
    def __init__(self, chat: ChatJson):
        self.chat = chat


class EngineerRole(_LlmRole):
    """Generation / refinement. Content is real; `quality` stays deterministic so the
    mock-critic path still converges (a real REVIEWER scores independently)."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        revision = params.get("revision", 0)
        subtopics = list(params["subtopics"])
        topic = params["topic"]
        material = [v for v in inputs.values() if isinstance(v, dict)]
        messages = [
            {"role": "system", "content": ("You are a technical writer. Using the source "
                                           "material, write one section per required subtopic. "
                                           "Respond with JSON only.")},
            {"role": "user", "content": json.dumps({
                "topic": topic, "required_subtopics": subtopics, "source_material": material,
                "critique_to_address": params.get("critique")})},
        ]
        recorded = ctx.record(f"draft_rev{revision}",
                              lambda: self.chat(messages, DRAFT_SCHEMA, "draft",
                                                ("covered", "sections")))
        data, usage = recorded["content"], recorded.get("usage")
        quality = round(0.60 + 0.30 * min(revision, 1), 3)   # deterministic — mock critic path
        return ExecutionResult(output={
            "kind": "draft", "topic": topic, "revision": revision, "covered": subtopics,
            "quality": quality, "note": "revised per critique" if revision else "initial pass",
            "supersedes": None,
            "sections": {s.get("subtopic", ""): s.get("text", "")
                         for s in data.get("sections", []) if isinstance(s, dict)},
        }, signal=Signal.CONTINUE, score=quality, usage=usage)


class ReviewerRole(_LlmRole):
    """The critic: scores the draft's content against the target, then steers exactly
    like the mock — reject => a refinement stratum (new engineer fed the critique + a
    fresh reviewer); approve => the synthesizer. Bounded convergence via the cap (§7)."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        target = params["target_score"]
        subtopics = list(params["subtopics"])
        draft = next(v for v in inputs.values() if isinstance(v, dict) and v.get("kind") == "draft")
        draft_id = next(k for k, v in inputs.items() if v is draft)
        messages = [
            {"role": "system", "content": ("You are a strict technical reviewer. Score the "
                                           "brief from 0.0 to 1.0 on how completely and accurately "
                                           "it covers the required subtopics, and give a one-line "
                                           "critique of what to improve. Respond with JSON only.")},
            {"role": "user", "content": json.dumps({
                "required_subtopics": subtopics, "target_score": target,
                "brief_sections": draft.get("sections", {})})},
        ]
        recorded = ctx.record(f"review_{draft_id}",
                              lambda: self.chat(messages, REVIEW_SCHEMA, "review",
                                                ("score", "critique")))
        data, usage = recorded["content"], recorded.get("usage")
        score = max(0.0, min(1.0, float(data["score"])))
        critique = str(data.get("critique", ""))

        if target - score > 1e-9:
            refined = NodeSpec(key="engineer", node_type=NodeType.ENGINEER,
                               input_keys=(draft_id, ctx.self_id),
                               params={"topic": draft["topic"], "subtopics": subtopics,
                                       "revision": draft["revision"] + 1, "critique": critique})
            re_review = NodeSpec(key="reviewer", node_type=NodeType.REVIEWER, input_keys=("engineer",),
                                 params={"target_score": target, "subtopics": subtopics})
            return ExecutionResult(
                output={"verdict": "reject", "score": round(score, 3), "critique": critique},
                signal=Signal.NEEDS_REFINEMENT, score=score, usage=usage,
                patch=SubgraphPatch(added_nodes=[refined, re_review]))

        synth = NodeSpec(key="synthesizer", node_type=NodeType.SYNTHESIZER,
                         input_keys=(draft_id, ctx.self_id),
                         params={"subtopics": subtopics, "target_score": target})
        return ExecutionResult(output={"verdict": "approve", "score": round(score, 3)},
                               signal=Signal.DONE, score=score, usage=usage,
                               patch=SubgraphPatch(added_nodes=[synth]))


class OrchestratorRole(_LlmRole):
    """The expander — the answer to the recursive-patch problem. The model emits a FLAT
    plan (scouts + focus, an engineer brief, a review rubric); this role assembles the
    scouts -> engineer -> reviewer subgraph deterministically and post-fills required
    subtopics so the goal predicate stays reachable no matter what the model plans."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        goal = ctx.goal if isinstance(ctx.goal, dict) else {}
        required = list(goal.get("subtopics", []))
        topic = goal.get("topic", "")
        target = goal.get("target_score", 0.8)
        messages = [
            {"role": "system", "content": (
                "You are a research orchestrator. Plan how to produce a vetted brief: choose "
                "the scouts (one per subtopic to investigate, each with a focus), a brief for "
                "the engineer, and a review rubric. You MUST include every required subtopic "
                "among the scouts. Respond with JSON only.")},
            {"role": "user", "content": json.dumps({
                "topic": topic, "required_subtopics": required, "target_score": target})},
        ]
        recorded = ctx.record("plan",
                              lambda: self.chat(messages, PLAN_SCHEMA, "plan",
                                                ("scouts", "engineer_brief", "review_rubric")))
        data, usage = recorded["content"], recorded.get("usage")

        planned: dict = {}
        for s in data.get("scouts", []):
            if isinstance(s, dict) and s.get("subtopic"):
                planned[str(s["subtopic"])] = str(s.get("focus", ""))
        for st in required:                                   # guarantee coverage
            planned.setdefault(st, "")
        subtopics = list(planned.keys())

        scouts = [NodeSpec(key=f"scout_{st}", node_type=NodeType.SCOUT,
                           input_keys=(ctx.self_id,),
                           params={"subtopic": st, "topic": topic, "focus": focus})
                  for st, focus in planned.items()]
        engineer = NodeSpec(key="engineer", node_type=NodeType.ENGINEER,
                            input_keys=tuple(s.key for s in scouts),
                            params={"topic": topic, "subtopics": subtopics, "revision": 0,
                                    "brief": data.get("engineer_brief", "")})
        reviewer = NodeSpec(key="reviewer", node_type=NodeType.REVIEWER, input_keys=("engineer",),
                            params={"target_score": target, "subtopics": subtopics,
                                    "rubric": data.get("review_rubric", "")})
        patch = SubgraphPatch(added_nodes=[*scouts, engineer, reviewer])
        return ExecutionResult(
            output={"plan": subtopics, "engineer_brief": data.get("engineer_brief", "")},
            signal=Signal.CONTINUE, patch=patch, usage=usage)


# --------------------------------------------------------------------------- #
# Retrieval scout — provider-neutral (no model), a real ranked retrieval       #
# --------------------------------------------------------------------------- #
def _overlap_rank(query: str, text: str) -> float:
    """Deterministic lexical relevance (Jaccard over tokens) — real retrieval ranking
    with no external service, so it runs offline. Swap for an embeddings ranker for
    semantic retrieval."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    d = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


class RetrievalScoutExecutor(Executor):
    """Frontier retrieval: ranks a corpus (from `ctx.goal['corpus']`) by relevance to
    this scout's subtopic and returns the top hits as facts. Pluggable ranker (default
    lexical overlap, offline; pass an embeddings ranker for semantic search). The
    retrieval is a recorded activity, so a retry/recovery replays the exact hits."""

    def __init__(self, top_k: int = 3, ranker: Optional[Callable[[str, str], float]] = None):
        self.top_k = top_k
        self.ranker = ranker or _overlap_rank

    def run(self, inputs, params, ctx) -> ExecutionResult:
        subtopic = params["subtopic"]
        topic = params.get("topic", "")
        corpus = ctx.goal.get("corpus", []) if isinstance(ctx.goal, dict) else []
        query = f"{topic} {subtopic}".strip()

        def _retrieve() -> dict:
            scored = []
            for i, doc in enumerate(corpus):
                text = doc["text"] if isinstance(doc, dict) else str(doc)
                did = doc.get("id", f"doc{i}") if isinstance(doc, dict) else f"doc{i}"
                scored.append((self.ranker(query, text), did, text))
            scored.sort(key=lambda s: (-s[0], s[1]))
            return {"hits": [{"id": did, "text": text}
                             for score, did, text in scored[:self.top_k] if score > 0]}

        result = ctx.record(f"retrieve_{subtopic}", _retrieve)
        hits = result["hits"]
        return ExecutionResult(output={
            "subtopic": subtopic, "facts": [h["text"] for h in hits],
            "sources": [h["id"] for h in hits], "source": f"corpus://{topic}/{subtopic}",
        }, signal=Signal.CONTINUE)
