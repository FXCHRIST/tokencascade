#!/usr/bin/env python3
"""Dev-set runner: per-category accuracy + per-task latency.

This drives THE decision of the day: hybrid vs local-only final submission.
Gate: >=14/16 overall AND no task over 25s -> flip LOCAL_CATEGORIES to all 8.

Usage (runs main.py in-process against the dev set):
    # Local-only measurement (no remote needed, no cost):
    LOCAL_CATEGORIES=factual,sentiment,ner,summarization,math,logic,code_debug,code_gen \
    LOCAL_MODEL_PATH=./model.gguf python devset/check.py

    # Hybrid measurement (needs your own Fireworks key — dev only):
    FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
    ALLOWED_MODELS=<paste launch-day list> LOCAL_MODEL_PATH=./model.gguf \
    python devset/check.py

Gold matching is keyword-based (all keywords must appear, case-insensitive) —
stricter than the real LLM judge, so passing here is a good sign.
"""

import json
import os
import subprocess
import sys
import tempfile
import time


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "tasks.json")) as f:
        dev = json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "tasks.json")
        out_path = os.path.join(tmp, "results.json")
        log_path = os.path.join(tmp, "inference_log.json")
        with open(in_path, "w") as f:
            json.dump(
                [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in dev], f
            )

        env = dict(os.environ)
        env.update(INPUT_PATH=in_path, OUTPUT_PATH=out_path, LOG_PATH=log_path)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "..", "main.py")], env=env
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"main.py exited {proc.returncode}")
            sys.exit(1)

        with open(out_path) as f:
            results = {r["task_id"]: r["answer"] for r in json.load(f)}
        tokens = 0
        routes = {}
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
            tokens = log.get("fireworks_tokens", 0)
            routes = log.get("routes", {})

    by_cat: dict[str, list[bool]] = {}
    for t in dev:
        ans = (results.get(t["task_id"]) or "").lower()
        ok = all(g.lower() in ans for g in t["gold"])
        by_cat.setdefault(t["category"], []).append(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {t['task_id']:<12} [{routes.get(t['task_id'],'?')}]")

    total = sum(sum(v) for v in by_cat.values())
    n = sum(len(v) for v in by_cat.values())
    print("\nPer-category:")
    for cat, oks in sorted(by_cat.items()):
        print(f"  {cat:<15} {sum(oks)}/{len(oks)}")
    print(f"\nTOTAL: {total}/{n}   fireworks_tokens={tokens}   wall={elapsed:.1f}s")
    print("Gate reminder: need >=14/16 here (and 16/19 real) — and every task <25s.")


if __name__ == "__main__":
    main()
