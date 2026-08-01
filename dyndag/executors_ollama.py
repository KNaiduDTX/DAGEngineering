"""
executors_ollama.py — the real executors (Phase 1 slice).

Replaces the mock ENGINEER and REVIEWER with ones that call a local model via Ollama;
GOAL/SCOUT/SYNTHESIZER stay deterministic mocks. Together they demonstrate the four
things Phase 1 must get right, in a small surface:

  * real model calls at the single Executor.run seam (generation AND critique);
  * structured output — Ollama's `format=<json schema>` constrains draft and review;
  * a validate -> repair loop that lands on the fail-closed guard (MalformedResultError);
  * the recorded-activity discipline — every model call is wrapped in ctx.record, so a
    retry or crash-recovery REPLAYS it instead of paying for it again (idempotency).

With the REVIEWER real, the critic loop's score is a genuine model judgment and
convergence is real: the loop either clears the target or terminates on the generation
cap (INV-7.1) — the substrate guarantees termination even if the critic never approves.
The SYNTHESIZER reads the reviewer's score (not the draft's self-report), so no other
node changes.

Retries: the Ollama client does not auto-retry, so the engine's GLOBAL retry budget
(governor) stays the single authority — no nested retry storm (the Anthropic SDK needs
max_retries=0 for the same reason; see the Phase 1 findings).

`ollama` is imported lazily inside the call, and nothing in the package imports this at
module load except the (lazy-safe) re-export — so the offline demo/tests never require
Ollama. Run the live slice with `python demo_ollama.py`.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .core import NodeType, Signal
from .executors import (ExecutionResult, Executor, MalformedResultError,
                        default_registry)
from .nodes import NodeSpec, SubgraphPatch
from .schemas import DRAFT_SCHEMA, REVIEW_SCHEMA


def _usage_of(resp) -> dict:
    """Real token counts from an Ollama ChatResponse (dict or object)."""
    get = (lambda k: resp.get(k)) if isinstance(resp, dict) else (lambda k: getattr(resp, k, None))
    return {"input_tokens": get("prompt_eval_count") or 0,
            "output_tokens": get("eval_count") or 0}


def _ollama_chat_json(model: str, host: Optional[str], messages: list,
                      schema: dict, required: tuple, repair_attempts: int) -> dict:
    """Call the model with a JSON-schema constraint, parse, validate the required keys,
    and repair on failure by showing the model its bad output. Returns
    {"content": <parsed dict>, "usage": {input_tokens, output_tokens}} — both
    JSON-serializable, so the whole thing round-trips the event log as one recorded
    activity (real cost is replayed too). Raises MalformedResultError if repair is
    exhausted -> the scheduler treats it as a failed attempt (fail-closed)."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            f"ollama not installed — `pip install ollama`, run `ollama serve`, "
            f"and `ollama pull {model}`"
        ) from exc
    client = ollama.Client(host=host) if host else ollama

    convo = list(messages)
    last_err = ""
    for _ in range(repair_attempts + 1):
        resp = client.chat(model=model, messages=convo, format=schema)
        content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        try:
            data = json.loads(content)
            if not isinstance(data, dict) or any(k not in data for k in required):
                raise ValueError(f"missing required keys {required}")
            return {"content": data, "usage": _usage_of(resp)}
        except (json.JSONDecodeError, ValueError) as err:
            last_err = str(err)
            convo = convo + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    f"That was not valid per the schema ({last_err}). "
                    f"Return ONLY a JSON object matching the schema.")},
            ]
    raise MalformedResultError(
        f"Ollama produced no schema-valid output in {repair_attempts + 1} attempts: {last_err}")


class OllamaEngineerExecutor(Executor):
    """Generation/refinement via a local model, with schema-constrained output and a
    bounded repair loop. Content is real; `quality` stays deterministic so the mock
    critic path still converges (the real REVIEWER below scores independently)."""

    def __init__(self, model: str = "llama3.2", host: Optional[str] = None,
                 repair_attempts: int = 2):
        self.model, self.host, self.repair_attempts = model, host, repair_attempts

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
                "topic": topic, "required_subtopics": subtopics,
                "source_material": material,
                "critique_to_address": params.get("critique")})},
        ]
        recorded = ctx.record(
            f"draft_rev{revision}",
            lambda: _ollama_chat_json(self.model, self.host, messages, DRAFT_SCHEMA,
                                      ("covered", "sections"), self.repair_attempts),
        )
        data, usage = recorded["content"], recorded.get("usage")
        quality = round(0.60 + 0.30 * min(revision, 1), 3)   # deterministic — mock critic path
        return ExecutionResult(
            output={
                "kind": "draft", "topic": topic, "revision": revision,
                "covered": subtopics, "quality": quality,
                "note": "revised per critique" if revision else "initial pass",
                "supersedes": None,
                "sections": {s.get("subtopic", ""): s.get("text", "")
                             for s in data.get("sections", []) if isinstance(s, dict)},
            },
            signal=Signal.CONTINUE, score=quality, usage=usage)


class OllamaReviewerExecutor(Executor):
    """The critic, for real: scores the draft's actual content against the target via a
    model call, then steers exactly like the mock — reject => emit a refinement stratum
    (new engineer fed the critique + a fresh reviewer); approve => emit the synthesizer.
    Bounded convergence is guaranteed by the generation cap (§7)."""

    def __init__(self, model: str = "llama3.2", host: Optional[str] = None,
                 repair_attempts: int = 2):
        self.model, self.host, self.repair_attempts = model, host, repair_attempts

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
        recorded = ctx.record(
            f"review_{draft_id}",
            lambda: _ollama_chat_json(self.model, self.host, messages, REVIEW_SCHEMA,
                                      ("score", "critique"), self.repair_attempts),
        )
        data, usage = recorded["content"], recorded.get("usage")
        score = max(0.0, min(1.0, float(data["score"])))
        critique = str(data.get("critique", ""))

        if target - score > 1e-9:
            # steer: emit a refinement stratum (new SSA nodes, forward edges)
            refined = NodeSpec(
                key="engineer", node_type=NodeType.ENGINEER,
                input_keys=(draft_id, ctx.self_id),
                params={"topic": draft["topic"], "subtopics": subtopics,
                        "revision": draft["revision"] + 1, "critique": critique})
            re_review = NodeSpec(
                key="reviewer", node_type=NodeType.REVIEWER, input_keys=("engineer",),
                params={"target_score": target, "subtopics": subtopics})
            return ExecutionResult(
                output={"verdict": "reject", "score": round(score, 3), "critique": critique},
                signal=Signal.NEEDS_REFINEMENT, score=score, usage=usage,
                patch=SubgraphPatch(added_nodes=[refined, re_review]))

        # approve: emit the aggregator that converges to the terminal output
        synth = NodeSpec(
            key="synthesizer", node_type=NodeType.SYNTHESIZER,
            input_keys=(draft_id, ctx.self_id),
            params={"subtopics": subtopics, "target_score": target})
        return ExecutionResult(
            output={"verdict": "approve", "score": round(score, 3)},
            signal=Signal.DONE, score=score, usage=usage,
            patch=SubgraphPatch(added_nodes=[synth]))


def _overlap_rank(query: str, text: str) -> float:
    """Deterministic lexical relevance (Jaccard over tokens) — real retrieval ranking
    with no external service, so it runs offline. Swap for an embeddings ranker
    (e.g. Ollama's embeddings endpoint) for semantic retrieval."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    d = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


class RetrievalScoutExecutor(Executor):
    """Frontier retrieval, for real: ranks a corpus (from `ctx.goal['corpus']`) by
    relevance to this scout's subtopic and returns the top hits as facts. The ranking
    is pluggable — the default is deterministic lexical overlap (offline); pass an
    embeddings-backed `ranker` for semantic search. The retrieval is a recorded
    activity, so a retry/recovery replays the exact hits (and, with a real ranker,
    doesn't re-embed)."""

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
        return ExecutionResult(
            output={
                "subtopic": subtopic,
                "facts": [h["text"] for h in hits],
                "sources": [h["id"] for h in hits],
                "source": f"corpus://{topic}/{subtopic}",
            },
            signal=Signal.CONTINUE)


def ollama_registry(model: str = "llama3.2", top_k: int = 3, **kwargs) -> dict:
    """`default_registry()` with SCOUT (real retrieval), ENGINEER, and REVIEWER (real
    Ollama) replaced; GOAL/SYNTHESIZER stay deterministic mocks. This is the Phase 1
    real-executor fleet. The SCOUT reads its corpus from `goal['corpus']` and uses
    offline lexical retrieval by default (no model)."""
    registry = default_registry()
    registry[NodeType.SCOUT] = RetrievalScoutExecutor(top_k=top_k)
    registry[NodeType.ENGINEER] = OllamaEngineerExecutor(model=model, **kwargs)
    registry[NodeType.REVIEWER] = OllamaReviewerExecutor(model=model, **kwargs)
    return registry
