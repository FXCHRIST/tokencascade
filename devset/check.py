#!/usr/bin/env python3
"""Devset runner and grader for TokenCascade.

Runs main.py as a subprocess against devset/tasks.json (exactly like the
harness runs the container), then grades every answer four ways:

  1. gold keywords     — every keyword must appear (case-insensitive)
  2. forbid_label      — the classification label must NOT be this value
                         (checks the label word, not mere word presence)
  3. format            — exact sentence count / exact bullet count /
                         max words per bullet, mirroring the official
                         T04/T04b pass rules
  4. exec_tests        — code answers are executed and each [expr, expected]
                         pair must evaluate to the expected value

Keyword matching is stricter than the real LLM judge, so passing here is a
good sign, not a guarantee.

Usage:
    LOCAL_MODEL_PATH=./model.gguf python devset/check.py
    THREADS=2 LOCAL_MODEL_PATH=./model.gguf python devset/check.py   # CI

Gate (exit code 1 on failure):
    GATE_MIN_CORRECT   default 20   (of 22)
    GATE_MAX_SECONDS   default 420  (total wall time of the agent run)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE_MIN = int(os.environ.get("GATE_MIN_CORRECT", "20"))
GATE_SEC = float(os.environ.get("GATE_MAX_SECONDS", "420"))


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def bullet_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^([-*\u2022]|\d+[.)])\s+", s):
            out.append(re.sub(r"^([-*\u2022]|\d+[.)])\s+", "", s).strip())
    return out


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    if "def " in text:
        return text[text.index("def "):].strip()
    return text.strip()


def check_keywords(answer, gold):
    low = answer.lower().replace(",", "")
    missing = [k for k in gold if k.lower() not in low]
    return (not missing), missing


def check_forbid_label(answer, forbidden):
    m = re.search(r"\b(positive|negative|neutral|mixed)\b", answer, re.I)
    if not m:
        return False, "no classification label found"
    if m.group(1).lower() == forbidden.lower():
        return False, f"label is {m.group(1)} (forbidden)"
    return True, ""


def check_format(answer, fmt):
    if "exact_bullets" in fmt:
        bl = bullet_lines(answer)
        if len(bl) != fmt["exact_bullets"]:
            return False, f"{len(bl)} bullets, need {fmt['exact_bullets']}"
        k = fmt.get("max_words_per_bullet")
        if k:
            over = [b for b in bl if len(b.split()) > k]
            if over:
                return False, f"bullet over {k} words: '{over[0][:50]}...'"
        return True, ""
    if "exact_sentences" in fmt:
        n = len(split_sentences(answer))
        if n != fmt["exact_sentences"]:
            return False, f"{n} sentences, need {fmt['exact_sentences']}"
    return True, ""


def check_exec(answer, tests):
    code = extract_code(answer)
    harness = code + "\nimport json as _j\n_r = []\n"
    for expr, _ in tests:
        harness += (f"try:\n    _r.append(_j.dumps({expr}))\n"
                    "except Exception as _e:\n"
                    "    _r.append('ERROR: ' + str(_e))\n")
    harness += "print(_j.dumps(_r))\n"
    try:
        r = subprocess.run([sys.executable, "-c", harness],
                           capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False, "execution timed out"
    if r.returncode != 0:
        return False, (r.stderr or "crashed").strip().splitlines()[-1][:120]
    try:
        results = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "could not parse test output"
    for (expr, expected), got in zip(tests, results):
        if isinstance(got, str) and got.startswith("ERROR:"):
            return False, f"{expr} -> {got[:100]}"
        try:
            got_val = json.loads(got)
        except Exception:
            got_val = got
        if got_val != expected:
            return False, f"{expr} -> {got_val!r}, expected {expected!r}"
    return True, ""


def grade(task, answer):
    reasons = []
    ok = True
    if not answer.strip():
        return False, ["empty answer"]
    if "gold" in task:
        k_ok, missing = check_keywords(answer, task["gold"])
        if not k_ok:
            ok = False
            reasons.append(f"missing keywords: {missing}")
    if "forbid_label" in task:
        f_ok, why = check_forbid_label(answer, task["forbid_label"])
        if not f_ok:
            ok = False
            reasons.append(why)
    if "format" in task:
        fm_ok, why = check_format(answer, task["format"])
        if not fm_ok:
            ok = False
            reasons.append(f"format: {why}")
    if "exec_tests" in task:
        e_ok, why = check_exec(answer, task["exec_tests"])
        if not e_ok:
            ok = False
            reasons.append(f"exec: {why}")
    return ok, reasons


def main():
    with open(os.path.join(HERE, "tasks.json")) as f:
        tasks = json.load(f)

    tmp = tempfile.mkdtemp(prefix="tokencascade-devset-")
    in_path = os.path.join(tmp, "tasks.json")
    out_path = os.path.join(tmp, "results.json")
    log_path = os.path.join(tmp, "inference_log.json")
    with open(in_path, "w") as f:
        json.dump([{"task_id": t["task_id"], "prompt": t["prompt"]}
                   for t in tasks], f)

    env = dict(os.environ)
    env.update({"INPUT_PATH": in_path, "OUTPUT_PATH": out_path,
                "LOG_PATH": log_path})
    env.setdefault("LOCAL_MODEL_PATH", os.path.join(ROOT, "model.gguf"))

    print(f"[check] running main.py on {len(tasks)} tasks "
          f"(threads={env.get('THREADS', 'auto')}, "
          f"model={env['LOCAL_MODEL_PATH']})")
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                          env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[check] FATAL: main.py exited {proc.returncode}")
        sys.exit(1)

    with open(out_path) as f:
        results = {r["task_id"]: r.get("answer", "") for r in json.load(f)}
    task_seconds = {}
    try:
        with open(log_path) as f:
            task_seconds = json.load(f).get("task_seconds", {})
    except Exception:
        pass

    missing_ids = [t["task_id"] for t in tasks if t["task_id"] not in results]
    if missing_ids:
        print(f"[check] FATAL: output missing task ids: {missing_ids}")
        sys.exit(1)

    correct = 0
    per_cat = {}
    report = {"tasks": [], "elapsed_s": round(elapsed, 1)}
    print(f"\n{'task':<18}{'cat':<15}{'sec':>6}  verdict")
    print("-" * 72)
    for t in tasks:
        tid, cat = t["task_id"], t.get("category", "?")
        ans = results.get(tid, "")
        ok, reasons = grade(t, ans)
        correct += ok
        c = per_cat.setdefault(cat, [0, 0])
        c[0] += ok
        c[1] += 1
        sec = task_seconds.get(tid, "")
        verdict = "PASS" if ok else "FAIL  " + "; ".join(reasons)[:90]
        print(f"{tid:<18}{cat:<15}{str(sec):>6}  {verdict}")
        report["tasks"].append({"task_id": tid, "category": cat, "pass": ok,
                                "seconds": sec, "reasons": reasons,
                                "answer": ans})

    print("-" * 72)
    for cat in sorted(per_cat):
        ok_n, tot = per_cat[cat]
        print(f"  {cat:<18}{ok_n}/{tot}")
    print(f"\n[check] TOTAL: {correct}/{len(tasks)} correct "
          f"in {elapsed:.1f}s (gate: >={GATE_MIN} and <={GATE_SEC:.0f}s)")

    report["correct"] = correct
    report["total"] = len(tasks)
    with open(os.path.join(os.getcwd(), "devset_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if correct < GATE_MIN or elapsed > GATE_SEC:
        print("[check] GATE FAILED")
        sys.exit(1)
    print("[check] GATE PASSED")


if __name__ == "__main__":
    main()
