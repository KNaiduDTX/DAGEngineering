"""
run_eval_parallel.py — time-efficient GSM8K / MATH eval (parallel, checkpointed, resumable).

Runs the selected arms x N tasks. Every `run_task` is independent and network-bound on Ollama
Cloud, which serves ~8-16 concurrent calls at near-flat latency (measured), so we overlap all
(config, task) runs in a flat thread pool of size POOL — the total concurrent model-call count
is exactly POOL (each run makes one call at a time). Robustness:
  * every completed run is checkpointed to JSONL immediately;
  * on restart the checkpoint is replayed and finished runs skipped (survives a kill);
  * errored runs are retried.

Arms (choose a subset with `--configs`): single / ssa / sc / combined (φ-of-SSA vote) /
combined_v2 (greedy-anchored SELECT — verify-first gate, never dilutes below single; `--margin`
sets the override threshold). `--task` picks GSM8K (exact-match) or MATH (LaTeX-equivalence
grader). Per-task wall_s is CONTENDED latency; accuracy / $-per-correct / generations / paired
McNemar are unaffected (each run is a deterministic, replayed unit).

Usage: python run_eval_parallel.py --task math --data aime2025.jsonl --n 30 --pool 8
       python run_eval_parallel.py --task math --data math500.jsonl --configs single,combined_v2
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))   # repo root = this script's directory
sys.path.insert(0, REPO)

from dyndag import (TaskResult, gsm8k_registry, gsm8k_report, load_gsm8k, load_math,  # noqa: E402
                    math_registry, paired_report, per_problem_rows, run_task)
from dyndag.executors_openrouter import openrouter_chat  # noqa: E402

OLLAMA_V1 = "http://localhost:11434/v1"
CONFIGS = ["single", "ssa", "sc", "combined", "combined_v2"]


def build(args):
    reg = {"gsm8k": gsm8k_registry, "math": math_registry}[args.task]
    load = {"gsm8k": load_gsm8k, "math": load_math}[args.task]

    def chat(temp):
        return openrouter_chat(model=args.model, base_url=OLLAMA_V1, api_key="ollama", temperature=temp)

    def registry_for(c):
        if c == "combined_v2":                     # greedy anchor + verify-gate + corroboration-select
            return reg(chat(None), mode="combined_v2", k=args.k, critic_chat=chat(None),
                       explore_chat=chat(args.sc_temperature), margin=args.margin)
        if c == "combined":                        # F1/F2/F3: sampled solvers, greedy critics
            return reg(chat(args.sc_temperature), mode="combined", k=args.k, critic_chat=chat(None))
        if c == "sc":
            return reg(chat(args.sc_temperature), mode="sc", k=args.k)
        return reg(chat(None), mode=c)             # greedy single / ssa

    load_kw = {"n": args.n, "seed": args.seed}
    if args.task == "math" and args.min_level:
        load_kw["min_level"] = args.min_level
    tasks = load(args.data, **load_kw)
    return tasks, registry_for


def _row(tr):
    return {"policy": tr.policy, "task": tr.task, "success": tr.success, "reason": tr.reason,
            "generations": tr.generations, "cost_usd": tr.cost_usd, "wall_s": tr.wall_s,
            "replay_ok": tr.replay_ok, "error": tr.error}


def _tr_from_row(r):
    return TaskResult(r["task"], r["policy"], r["success"], r["reason"], r["cost_usd"], 0, 0, 0, 0,
                      generations=r["generations"], wall_s=r["wall_s"], replay_ok=r["replay_ok"],
                      error=r["error"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "math"], default="math")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--sc-temperature", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="combined_v2: extra corroboration required to override the greedy anchor")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated arms to run (subset of %s)" % ",".join(CONFIGS))
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--min-level", type=int, default=None, help="MATH: keep only level >= this")
    ap.add_argument("--ckpt", default=f"{REPO}/results_checkpoint.jsonl")
    ap.add_argument("--out", default=f"{REPO}/results.json")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    tasks, registry_for = build(args)
    results = {c: {} for c in configs}
    by_name = {t.name: t for t in tasks}

    done = set()
    try:
        with open(args.ckpt, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["policy"] in results and r.get("error") is None:
                    results[r["policy"]][r["task"]] = _tr_from_row(r)
                    done.add((r["policy"], r["task"]))
    except FileNotFoundError:
        pass

    jobs = [(c, t) for c in configs for t in tasks if (c, t.name) not in done]
    print(f"task={args.task} model={args.model} n={len(tasks)} k={args.k} pool={args.pool} "
          f"resumed={len(done)} todo={len(jobs)}", flush=True)

    def run_one(c, task):
        tr = run_task(task, registry_for(c), max_workers=1)
        tr.policy = c
        return c, task.name, tr

    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.pool) as pool, open(args.ckpt, "a", encoding="utf-8") as cf:
        futs = [pool.submit(run_one, c, t) for c, t in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            c, name, tr = f.result()
            results[c][name] = tr
            with lock:
                cf.write(json.dumps(_row(tr)) + "\n")
                cf.flush()
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    for rnd in range(2):
        errs = [(c, n) for c in configs for n in results[c] if results[c][n].error is not None]
        errs += [(c, t.name) for c in configs for t in tasks if t.name not in results[c]]
        if not errs:
            break
        print(f"retry {rnd + 1}: {len(errs)} runs", flush=True)
        for c, name in errs:
            _, _, tr = run_one(c, by_name[name])
            results[c][name] = tr
            with open(args.ckpt, "a", encoding="utf-8") as cf:
                cf.write(json.dumps(_row(tr)) + "\n")

    remaining = sum(1 for c in configs for t in tasks
                    if t.name not in results[c] or results[c][t.name].error is not None)
    print(f"\nwall={time.time() - t0:.0f}s remaining_errors={remaining}\n", flush=True)
    print(gsm8k_report(results))
    print()
    print(paired_report(results))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(per_problem_rows(results), fh, indent=2)
    print(f"\nper-problem results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
