"""
executors_openrouter.py — cloud executors over the OpenAI-compatible API (OpenRouter).

Provider-AGNOSTIC by construction. A thin raw-stdlib (urllib) transport speaks the
OpenAI `/chat/completions` wire format, so the same code targets OpenRouter (cloud), a
local OpenAI-compatible server (Ollama's `/v1`, vLLM, LM Studio), or any OpenAI skin —
change `base_url` + `model`. No provider SDK is imported, which is the whole point:

  * **max_retries = 0 by construction** — there is no retry layer beneath the engine's
    global retry budget, so no nested retry storm (the collision the governor exists to
    prevent; a provider SDK would need max_retries=0 set explicitly and re-checked on
    every upgrade — this has nothing to un-set);
  * **refusals / content filters / 4xx → TerminalExecutionError** — the node fails
    without spending a retry (retrying a refusal just refuses again);
  * **per-model cost is the provider's own number** — `usage: {include: true}` returns
    `usage.cost` in dollars, so there is no local pricing table to drift as models and
    prices change. The scheduler records that dollar figure directly.

The transport is injectable (`transport=`), so the parse / validate / repair / refusal
/ cost logic is fully unit-testable offline with a canned completion — no network, no
key. Nothing in the package imports this at load beyond the (import-safe) re-export.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .core import NodeType, Signal
from .executors import (ExecutionResult, Executor, MalformedResultError,
                        TerminalExecutionError, default_registry)
from .nodes import NodeSpec, SubgraphPatch
from .schemas import DRAFT_SCHEMA, REVIEW_SCHEMA

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# A transport maps (model, messages, response_format) -> an OpenAI-compatible
# completion dict. The default is the real HTTP call; tests inject a fake.
Transport = Callable[[str, list, Optional[dict]], dict]


def _json_schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def http_transport(base_url: str, api_key: Optional[str], timeout: float = 60.0) -> Transport:
    """A raw-urllib transport for an OpenAI-compatible `/chat/completions` endpoint.
    No SDK, no retries. 429 / 5xx / network -> retryable RuntimeError; 4xx / no-key ->
    TerminalExecutionError. The key is read once here (arg or OPENROUTER_API_KEY /
    OPENAI_API_KEY); a missing key only fails when the transport is actually called, so
    building a registry offline is fine."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def _call(model: str, messages: list, response_format: Optional[dict]) -> dict:
        import urllib.error
        import urllib.request
        if not key:
            raise TerminalExecutionError("no API key — set OPENROUTER_API_KEY (or pass api_key=)")
        body = {"model": model, "messages": messages, "usage": {"include": True}}
        if response_format is not None:
            body["response_format"] = response_format
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if 400 <= exc.code < 500 and exc.code != 429:
                raise TerminalExecutionError(f"OpenRouter {exc.code}: {detail}")
            raise RuntimeError(f"OpenRouter {exc.code} (retryable): {detail}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection error (retryable): {exc}")
        if isinstance(data, dict) and data.get("error"):
            raise TerminalExecutionError(f"OpenRouter error: {data['error']}")
        return data

    return _call


def _extract(resp: dict) -> tuple:
    """Pull (content_str, usage_dict) from an OpenAI-compatible completion, mapping a
    refusal / content filter to TerminalExecutionError. usage carries the provider's
    dollar `cost` when present."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise TerminalExecutionError(f"model refused: {message['refusal']}")
    if choice.get("finish_reason") in ("content_filter", "error"):
        raise TerminalExecutionError(
            f"terminal finish_reason={choice.get('finish_reason')} "
            f"native={choice.get('native_finish_reason')}")
    u = resp.get("usage") or {}
    usage = {"input_tokens": u.get("prompt_tokens", 0) or 0,
             "output_tokens": u.get("completion_tokens", 0) or 0}
    if u.get("cost") is not None:
        usage["cost"] = u["cost"]
    return message.get("content"), usage


class _OpenRouterExecutor(Executor):
    """Base: OpenAI-compatible chat with schema-constrained output, a bounded repair
    loop, refusal->terminal, and provider cost. Subclasses add role behavior."""

    def __init__(self, model: str = "openai/gpt-4o-mini",
                 base_url: str = OPENROUTER_BASE_URL, api_key: Optional[str] = None,
                 repair_attempts: int = 2, transport: Optional[Transport] = None):
        self.model = model
        self.repair_attempts = repair_attempts
        self._transport = transport or http_transport(base_url, api_key)

    def _chat_json(self, messages: list, schema: dict, name: str, required: tuple) -> dict:
        """Call, validate, repair. Returns {"content": dict, "usage": {...cost}}.
        MalformedResultError (retryable) if repair exhausts; TerminalExecutionError
        (no retry) on refusal — the latter is raised out of _extract/transport."""
        convo = list(messages)
        fmt = _json_schema_format(name, schema)
        last_err = ""
        for _ in range(self.repair_attempts + 1):
            content, usage = _extract(self._transport(self.model, convo, fmt))
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
            f"OpenRouter produced no schema-valid output in {self.repair_attempts + 1} "
            f"attempts: {last_err}")


class OpenRouterEngineerExecutor(_OpenRouterExecutor):
    """Generation/refinement over the OpenAI-compatible API. Content is real; `quality`
    stays deterministic so the mock-critic path converges (a real REVIEWER scores)."""

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
                              lambda: self._chat_json(messages, DRAFT_SCHEMA, "draft",
                                                      ("covered", "sections")))
        data, usage = recorded["content"], recorded.get("usage")
        quality = round(0.60 + 0.30 * min(revision, 1), 3)
        return ExecutionResult(output={
            "kind": "draft", "topic": topic, "revision": revision, "covered": subtopics,
            "quality": quality, "note": "revised per critique" if revision else "initial pass",
            "supersedes": None,
            "sections": {s.get("subtopic", ""): s.get("text", "")
                         for s in data.get("sections", []) if isinstance(s, dict)},
        }, signal=Signal.CONTINUE, score=quality, usage=usage)


class OpenRouterReviewerExecutor(_OpenRouterExecutor):
    """The critic over the OpenAI-compatible API: scores the draft's content and steers
    (reject -> refinement stratum / approve -> synthesizer), like the mock but real."""

    def run(self, inputs, params, ctx) -> ExecutionResult:
        target = params["target_score"]
        subtopics = list(params["subtopics"])
        draft = next(v for v in inputs.values() if isinstance(v, dict) and v.get("kind") == "draft")
        draft_id = next(k for k, v in inputs.items() if v is draft)
        messages = [
            {"role": "system", "content": ("You are a strict technical reviewer. Score the brief "
                                           "0.0-1.0 on coverage and accuracy of the required "
                                           "subtopics, and give a one-line critique. JSON only.")},
            {"role": "user", "content": json.dumps({
                "required_subtopics": subtopics, "target_score": target,
                "brief_sections": draft.get("sections", {})})},
        ]
        recorded = ctx.record(f"review_{draft_id}",
                              lambda: self._chat_json(messages, REVIEW_SCHEMA, "review",
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


def openrouter_registry(model: str = "openai/gpt-4o-mini", base_url: str = OPENROUTER_BASE_URL,
                        api_key: Optional[str] = None, transport: Optional[Transport] = None,
                        **kwargs) -> dict:
    """`default_registry()` with ENGINEER and REVIEWER over the OpenAI-compatible API
    (OpenRouter default). The ORCHESTRATOR is added in the next slice; GOAL/SCOUT/
    SYNTHESIZER stay mock. Pass `transport=` to target a fake (tests) or any other
    OpenAI-compatible endpoint (e.g. Ollama's /v1) without changing the executors."""
    common = dict(model=model, base_url=base_url, api_key=api_key, transport=transport, **kwargs)
    registry = default_registry()
    registry[NodeType.ENGINEER] = OpenRouterEngineerExecutor(**common)
    registry[NodeType.REVIEWER] = OpenRouterReviewerExecutor(**common)
    return registry
