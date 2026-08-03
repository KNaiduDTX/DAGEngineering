"""
executors_ollama.py — the native-Ollama backend as a provider-agnostic `chat`.

The ENGINEER/REVIEWER/ORCHESTRATOR role logic lives once in `roles.py`; this module only
supplies the Ollama-native structured-chat primitive (`ollama.chat(..., format=schema)`)
wrapped as a `ChatJson` via the shared repair loop, plus the registry. `ollama` is
imported lazily, and the single call is injectable (`call=`) so the whole path is
offline-testable with a fake — no package, no server.

(The OpenAI-compatible backend in `executors_openrouter`, pointed at Ollama's `/v1`, is
an alternative local path; this native one uses Ollama's own `format=` structured output,
the more established structured-output route for Ollama.)
"""

from __future__ import annotations

from typing import Optional

from .core import NodeType
from .executors import default_registry
from .roles import (ChatJson, ChatOnce, EngineerRole, RetrievalScoutExecutor, ReviewerRole,
                    chat_json_with_repair)


def _usage_of(resp) -> dict:
    """Real token counts from an Ollama ChatResponse (dict or object)."""
    get = (lambda k: resp.get(k)) if isinstance(resp, dict) else (lambda k: getattr(resp, k, None))
    return {"input_tokens": get("prompt_eval_count") or 0,
            "output_tokens": get("eval_count") or 0}


def _native_call(model: str, host: Optional[str], temperature: Optional[float] = None) -> ChatOnce:
    """A single structured call over the native `ollama` package (lazy import). `temperature`
    (when set) samples via Ollama's `options` — needed for self-consistency diversity."""
    def call_once(messages, schema, name):     # `name` unused — native format= has no schema name
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                f"ollama not installed — `pip install ollama`, run `ollama serve`, "
                f"and `ollama pull {model}`") from exc
        client = ollama.Client(host=host) if host else ollama
        kwargs = {"options": {"temperature": temperature}} if temperature is not None else {}
        resp = client.chat(model=model, messages=messages, format=schema, **kwargs)
        content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        return content, _usage_of(resp)
    return call_once


def ollama_chat(model: str = "llama3.2", host: Optional[str] = None, repair_attempts: int = 2,
                call: Optional[ChatOnce] = None, temperature: Optional[float] = None) -> ChatJson:
    """A provider-agnostic `chat` backed by native Ollama (structured output via `format=`).
    Pass `call=` a fake single-call primitive to unit-test the repair loop + roles offline
    (no `ollama` package, no server). `temperature` samples (needed for self-consistency)."""
    call_once = call or _native_call(model, host, temperature=temperature)

    def chat(messages, schema, name, required):
        return chat_json_with_repair(call_once, messages, schema, name, required, repair_attempts)
    return chat


def ollama_registry(model: str = "llama3.2", top_k: int = 3, **kwargs) -> dict:
    """`default_registry()` with a real retrieval SCOUT + native-Ollama ENGINEER and
    REVIEWER; GOAL/ORCHESTRATOR/SYNTHESIZER stay mock. The SCOUT reads its corpus from
    `goal['corpus']` and uses offline lexical retrieval by default (no model)."""
    chat = ollama_chat(model=model, **kwargs)
    registry = default_registry()
    registry[NodeType.SCOUT] = RetrievalScoutExecutor(top_k=top_k)
    registry[NodeType.ENGINEER] = EngineerRole(chat)
    registry[NodeType.REVIEWER] = ReviewerRole(chat)
    return registry
