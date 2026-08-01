"""
executors_ollama.py — the first REAL executor (Phase 1 slice).

Replaces the mock ENGINEER with one that calls a local model via Ollama; every other
role stays a deterministic mock. It demonstrates the four things Phase 1 must get
right, in the smallest surface:

  * a real model call at the single Executor.run seam;
  * structured output — Ollama's `format=<json schema>` constrains the draft;
  * a validate -> repair loop that lands on the fail-closed guard (MalformedResultError);
  * the recorded-activity discipline — the model call is wrapped in ctx.record, so a
    retry or crash-recovery REPLAYS it instead of paying for it again (idempotency).

Determinism note: the ENGINEER generates real *content*, but the draft's `quality`
stays the deterministic, revision-based value the mock used, so the (still mock) critic
loop converges predictably. Scoring is the REVIEWER's job; making it real is the next
slice. Coverage is likewise claimed against the required subtopics — verifying it
against the generated text is also the future REVIEWER's job.

Retries: the Ollama Python client does not auto-retry, so the engine's GLOBAL retry
budget (governor) stays the single authority on retries — no nested retry storm (the
Anthropic SDK needs max_retries=0 for the same reason; see the Phase 1 findings).

Nothing else in the package imports this module, and `ollama` is imported lazily
inside the call — so the offline demo/tests never require Ollama to be installed.
Run the live slice with `python demo_ollama.py`.
"""

from __future__ import annotations

import json
from typing import Optional

from .core import NodeType, Signal
from .executors import (ExecutionResult, Executor, MalformedResultError,
                        default_registry)

# JSON schema constraining the model's draft content. Deliberately FLAT — a
# patch-emitting role (orchestrator/reviewer) would need a bounded-depth schema
# because structured-output backends reject recursive schemas (Phase 1 finding).
_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"subtopic": {"type": "string"}, "text": {"type": "string"}},
                "required": ["subtopic", "text"],
            },
        },
    },
    "required": ["covered", "sections"],
}


class OllamaEngineerExecutor(Executor):
    """Generation/refinement via a local Ollama model, with schema-constrained output
    and a bounded repair loop. Drop-in for the mock EngineerExecutor."""

    def __init__(self, model: str = "llama3.2", host: Optional[str] = None,
                 repair_attempts: int = 2):
        self.model = model
        self.host = host
        self.repair_attempts = repair_attempts

    def _generate(self, messages: list[dict]) -> dict:
        """One recorded activity: call the model, validate, and repair on failure.
        Returns the parsed content dict (JSON-serializable, so it round-trips through
        the event log). Raises MalformedResultError if repair is exhausted -> the
        scheduler treats it as a failed attempt (fail-closed)."""
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                f"ollama not installed — `pip install ollama`, run `ollama serve`, "
                f"and `ollama pull {self.model}`"
            ) from exc
        client = ollama.Client(host=self.host) if self.host else ollama

        convo = list(messages)
        last_err = ""
        for _ in range(self.repair_attempts + 1):
            resp = client.chat(model=self.model, messages=convo, format=_DRAFT_SCHEMA)
            content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            try:
                data = json.loads(content)
                if not (isinstance(data, dict) and "covered" in data and "sections" in data):
                    raise ValueError("missing required keys 'covered'/'sections'")
                return data
            except (json.JSONDecodeError, ValueError) as err:
                last_err = str(err)
                convo = convo + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        f"That was not valid per the schema ({last_err}). "
                        f"Return ONLY a JSON object matching the schema.")},
                ]
        raise MalformedResultError(
            f"Ollama engineer produced no schema-valid output in "
            f"{self.repair_attempts + 1} attempts: {last_err}")

    def run(self, inputs, params, ctx) -> ExecutionResult:
        revision = params.get("revision", 0)
        subtopics = list(params["subtopics"])
        topic = params["topic"]
        critique = params.get("critique")

        material = [v for v in inputs.values() if isinstance(v, dict)]
        system = ("You are a technical writer. Using the provided source material, write "
                  "one section per required subtopic. Respond with JSON only.")
        user = {"topic": topic, "required_subtopics": subtopics,
                "source_material": material, "critique_to_address": critique}
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)}]

        # The model call is a recorded activity: a retry/recovery replays it.
        data = ctx.record(f"draft_rev{revision}", lambda: self._generate(messages))

        quality = round(0.60 + 0.30 * min(revision, 1), 3)   # deterministic — critic converges
        draft = {
            "kind": "draft",
            "topic": topic,
            "revision": revision,
            "covered": subtopics,                # coverage claim; real check is the REVIEWER's job
            "quality": quality,
            "note": "revised per critique" if revision else "initial pass",
            "supersedes": None,
            "sections": {s.get("subtopic", ""): s.get("text", "")
                         for s in data.get("sections", []) if isinstance(s, dict)},
        }
        return ExecutionResult(output=draft, signal=Signal.CONTINUE, score=quality)


def ollama_registry(model: str = "llama3.2", **kwargs) -> dict:
    """`default_registry()` with the ENGINEER replaced by a real Ollama-backed one;
    every other role stays a deterministic mock. This is the Phase 1 slice."""
    registry = default_registry()
    registry[NodeType.ENGINEER] = OllamaEngineerExecutor(model=model, **kwargs)
    return registry
