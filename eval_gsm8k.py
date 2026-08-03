"""
eval_gsm8k.py — GSM8K accuracy vs the cost/quality frontier, measured live.

Runs GSM8K under four reasoning-strategy policies on ONE provider, so the comparison isolates
the STRATEGY (not the model) and ablates the two construction primitives (refine, merge):

  * single-pass  — neither. one solver, no critic
  * ssa          — refine only. solver + checker; the SSA loop (each retry a NEW generation)
  * sc           — merge only. K sampled solvers -> a φ majority-vote
  * combined     — both. K ADAPTIVE lineages -> a verifier-weighted φ over a DYNAMIC JOIN
                   (each lineage refines to its own critic's satisfaction; F2 runs the critics
                   greedy while the solvers sample)

The report prints the paper's five numbers (accuracy, $ per correct answer, mean generations,
mean wall-clock, replay-equivalence live) PLUS a paired McNemar analysis over the shared task
set — the right test when every policy runs the same problems.

Provider is auto-detected (Ollama at :11434, else OPENROUTER_API_KEY -> --cloud-model, default
DeepSeek V4 Flash); the run skips if neither is available. Data is a GSM8K test JSONL; with no
--data it falls back to the tiny AUTHORED sample set (a plumbing smoke, not a benchmark number).

    curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o test.jsonl
    OPENROUTER_API_KEY=... python eval_gsm8k.py --data test.jsonl --n 300 \
        --configs single,ssa,sc,combined --k 5 --dump results.json
"""

from __future__ import annotations

import argparse
import json
import os

from dyndag import (gsm8k_registry, gsm8k_report, load_gsm8k, paired_report,
                    per_problem_rows, run_eval, sample_tasks)
from dyndag.executors_openrouter import openrouter_chat

OLLAMA_V1 = "http://localhost:11434/v1"
OPENROUTER_V1 = "https://openrouter.ai/api/v1"
# sc = self-consistency (K samples, φ majority-vote); combined = φ-of-SSA (K refined lineages).
_KNOWN = ("single", "ssa", "sc", "combined")
_SAMPLED = ("sc", "combined")       # arms whose K lineages need temperature diversity


def _ollama_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _chat_factory(local_model: str, cloud_model: str):
    """Return (make_chat(temperature)->chat, provider_str) or (None, None). ONE provider for
    all policies (the comparison is strategy, not provider); self-consistency draws its `chat`
    with a temperature so its K samples differ, while single/ssa stay greedy."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return (lambda t=None: openrouter_chat(model=cloud_model, base_url=OPENROUTER_V1,
                                               api_key=key, temperature=t)), f"cloud:{cloud_model}"
    if _ollama_up():
        return (lambda t=None: openrouter_chat(model=local_model, base_url=OLLAMA_V1,
                                               api_key="ollama", temperature=t)), f"local:{local_model}"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="GSM8K eval on the self-constructing DAG.")
    ap.add_argument("--data", default=None, help="GSM8K test.jsonl (omit => authored sample set)")
    ap.add_argument("--n", type=int, default=250, help="random subset size (with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="single,ssa,sc,combined",
                    help="comma list of: single, ssa, sc, combined")
    ap.add_argument("--k", type=int, default=5, help="samples/lineages for sc and combined")
    ap.add_argument("--sc-temperature", type=float, default=0.8,
                    help="sampling temperature for sc/combined solvers (diversity)")
    ap.add_argument("--local-model", default="llama3.2")
    ap.add_argument("--cloud-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--max-workers", type=int, default=1, help="WIP cap per task (concurrency)")
    ap.add_argument("--dump", default=None, help="write per-problem results JSON to this path")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in _KNOWN]
    if unknown:
        print(f"[skip] unknown configs {unknown}; known: {list(_KNOWN)}.")
        return

    make_chat, provider = _chat_factory(args.local_model, args.cloud_model)
    if make_chat is None:
        print("[skip] no provider available.")
        print("       Start Ollama (`ollama serve` + `ollama pull llama3.2`) or set "
              "OPENROUTER_API_KEY, then re-run.")
        return

    tasks = load_gsm8k(args.data, n=args.n, seed=args.seed) if args.data else sample_tasks()
    source = args.data or "authored-sample"

    def _registry_for(c):
        if c == "combined":                              # F1/F2/F3: sampled solvers, GREEDY critics
            return gsm8k_registry(make_chat(args.sc_temperature), mode="combined", k=args.k,
                                  critic_chat=make_chat(None))
        if c == "sc":                                    # sampled solvers so the K samples differ
            return gsm8k_registry(make_chat(args.sc_temperature), mode="sc", k=args.k)
        return gsm8k_registry(make_chat(None), mode=c)   # greedy single / ssa
    policies = {c: (lambda c=c: _registry_for(c)) for c in configs}

    kinfo = f"  k={args.k}" if any(c in _SAMPLED for c in configs) else ""
    print(f"provider={provider}  data={source}  tasks={len(tasks)}  configs={configs}{kinfo}\n")
    results = run_eval(tasks, policies, max_workers=args.max_workers)
    print(gsm8k_report(results))
    if len(configs) >= 2:
        print()
        print(paired_report(results))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(per_problem_rows(results), fh, indent=2)
        print(f"\nper-problem results -> {args.dump}")


if __name__ == "__main__":
    main()
