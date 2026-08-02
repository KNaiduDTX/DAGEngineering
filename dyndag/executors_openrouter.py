"""
executors_openrouter.py — the OpenAI-compatible backend as a provider-agnostic `chat`.

Provider-AGNOSTIC by construction: a raw-stdlib (urllib) transport speaks the OpenAI
`/chat/completions` wire format, so the same `chat` targets OpenRouter (cloud), a local
OpenAI-compatible server (Ollama's `/v1`, vLLM, LM Studio), or any skin — change
`base_url` + `model`. No provider SDK, which is the whole point:

  * **max_retries = 0 by construction** — no retry layer beneath the engine's global
    retry budget, so no nested retry storm; nothing to un-set or re-check on upgrade;
  * **refusals / content filters / 4xx → TerminalExecutionError** — the node fails
    without spending a retry (retrying a refusal just refuses again);
  * **per-model cost is the provider's own number** — `usage: {include: true}` returns
    `usage.cost` in dollars, so there is no local pricing table to drift.

The ENGINEER/REVIEWER/ORCHESTRATOR role logic lives once in `roles.py`; this module only
provides the transport, the extract (refusal + cost), and the `chat` factory + registry.
The transport is injectable (`transport=`), so the whole path is offline-testable with a
canned completion — no network, no key.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .core import NodeType
from .executors import TerminalExecutionError, default_registry
from .roles import (ChatJson, ChatOnce, EngineerRole, OrchestratorRole, ReviewerRole,
                    chat_json_with_repair)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# A transport maps (model, messages, response_format) -> an OpenAI-compatible completion
# dict. The default is the real HTTP call; tests inject a fake.
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


def _openai_call(model: str, transport: Transport) -> ChatOnce:
    """A single structured call over an OpenAI-compatible transport (with refusal + cost)."""
    def call_once(messages, schema, name):
        return _extract(transport(model, messages, _json_schema_format(name, schema)))
    return call_once


def openrouter_chat(model: str = "openai/gpt-4o-mini", base_url: str = OPENROUTER_BASE_URL,
                    api_key: Optional[str] = None, repair_attempts: int = 2,
                    transport: Optional[Transport] = None) -> ChatJson:
    """A provider-agnostic `chat` over the OpenAI-compatible API (OpenRouter default).
    Pass `transport=` a fake to unit-test the repair loop + roles offline, or point
    `base_url` at any OpenAI-compatible endpoint (e.g. Ollama's /v1)."""
    call_once = _openai_call(model, transport or http_transport(base_url, api_key))

    def chat(messages, schema, name, required):
        return chat_json_with_repair(call_once, messages, schema, name, required, repair_attempts)
    return chat


def openrouter_registry(model: str = "openai/gpt-4o-mini", base_url: str = OPENROUTER_BASE_URL,
                        api_key: Optional[str] = None, transport: Optional[Transport] = None,
                        **kwargs) -> dict:
    """`default_registry()` with a real ORCHESTRATOR, ENGINEER, and REVIEWER over the
    OpenAI-compatible API (OpenRouter default). GOAL/SCOUT/SYNTHESIZER stay mock (the
    eval composes a real retrieval SCOUT where wanted)."""
    chat = openrouter_chat(model=model, base_url=base_url, api_key=api_key,
                           transport=transport, **kwargs)
    registry = default_registry()
    registry[NodeType.ORCHESTRATOR] = OrchestratorRole(chat)
    registry[NodeType.ENGINEER] = EngineerRole(chat)
    registry[NodeType.REVIEWER] = ReviewerRole(chat)
    return registry
