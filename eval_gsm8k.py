"""
eval_gsm8k.py — GSM8K accuracy vs the cost/quality frontier, measured live.

Runs GSM8K under reasoning-strategy policies on ONE provider, so the comparison isolates the
STRATEGY (not the model):

  * single-pass  — one solver, no critic
  * ssa          — solver + checker: the SSA refinement loop (each retry a NEW generation,
                   solver.g{k+1}, never a cycle)

The report prints the paper's five numbers: accuracy, $ per correct answer, mean generations
(does SSA iteration help?), mean wall-clock (latency under concurrency), and replay-
equivalence holding live during the eval.

Provider is auto-detected (Ollama at :11434, else OPENROUTER_API_KEY); the run skips if
neither is available. Data is a GSM8K test JSONL; with no --data it falls back to the tiny
AUTHORED sample set (a plumbing smoke test, not a real benchmark number).

    curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o test.jsonl
    OPENROUTER_API_KEY=... python eval_gsm8k.py --data test.jsonl --n 250 --configs single,ssa
"""

from __future__ import annotations

import argparse
import os

from dyndag import gsm8k_registry, gsm8k_report, load_gsm8k, run_eval, sample_tasks
from dyndag.executors_openrouter import openrouter_chat

OLLAMA_V1 = "http://localhost:11434/v1"
OPENROUTER_V1 = "https://openrouter.ai/api/v1"
_KNOWN = ("single", "ssa")          # self-consistency ('sc') arrives with the PHI node


def _ollama_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _pick_chat(local_model: str, cloud_model: str):
    """One `chat` for all policies (the comparison is strategy, not provider). Prefer cloud
    when a key is set, else a local Ollama endpoint via its OpenAI-compatible /v1."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return openrouter_chat(model=cloud_model, base_url=OPENROUTER_V1, api_key=key), f"cloud:{cloud_model}"
    if _ollama_up():
        return openrouter_chat(model=local_model, base_url=OLLAMA_V1, api_key="ollama"), f"local:{local_model}"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="GSM8K eval on the self-constructing DAG.")
    ap.add_argument("--data", default=None, help="GSM8K test.jsonl (omit => authored sample set)")
    ap.add_argument("--n", type=int, default=250, help="random subset size (with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="single,ssa", help="comma list of: single, ssa")
    ap.add_argument("--local-model", default="llama3.2")
    ap.add_argument("--cloud-model", default="openai/gpt-4o-mini")
    ap.add_argument("--max-workers", type=int, default=1, help="WIP cap per task (concurrency)")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in _KNOWN]
    if unknown:
        print(f"[skip] unknown configs {unknown}; known: {list(_KNOWN)} "
              f"(self-consistency 'sc' arrives with the PHI node).")
        return

    chat, provider = _pick_chat(args.local_model, args.cloud_model)
    if chat is None:
        print("[skip] no provider available.")
        print("       Start Ollama (`ollama serve` + `ollama pull llama3.2`) or set "
              "OPENROUTER_API_KEY, then re-run.")
        return

    tasks = load_gsm8k(args.data, n=args.n, seed=args.seed) if args.data else sample_tasks()
    source = args.data or "authored-sample"
    policies = {c: (lambda c=c: gsm8k_registry(chat, mode=c)) for c in configs}

    print(f"provider={provider}  data={source}  tasks={len(tasks)}  configs={configs}\n")
    results = run_eval(tasks, policies, max_workers=args.max_workers)
    print(gsm8k_report(results))


if __name__ == "__main__":
    main()
