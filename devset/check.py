#!/usr/bin/env python3
"""Devset gate for TokenCascade v6.

Runs the REAL pipeline (main.py handlers + the actual GGUF model) against
devset/tasks.json and grades every answer against the embedded spec:

  require_all / require_any / require_any_2 / require_any_3
        substrings that must appear (case-insensitive)
  label_any / label_forbid_leading
        sentiment label rules (label must appear early in the answer)
  sentences / bullets / bullet_max_words / max_words
        exact format constraints
  numeric
        the extracted final number must equal this value
  answer_line_contains / answer_line_regex
        checks against the text after 'Answer:'
  code_test
        the extracted code block is executed and the assertion (with `f`
        bound to the defined function) must pass

Usage:
  LOCAL_MODEL_PATH=./model.gguf python devset/check.py
  THREADS=2 LOCAL_MODEL_PATH=./model.gguf \
    GATE_MIN_CORRECT=20 GATE_MAX_SECONDS=420 python devset/check.py

Exit code 0 iff correct >= GATE_MIN_CORRECT and runtime <= GATE_MAX_SECONDS.
Writes devset_report.json either way.
"""
import ast
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# The pipeline never touches Fireworks; give it a generous local budget so
# the gate measures accuracy, and let GATE_MAX_SECONDS enforce time.
os.environ.setdefault("TIME_BUDGET_S", "3600")

import main as agent  # noqa: E402


def _extract_code(answer: str) -> str:
    return agent.extract_code(answer)


def _answer_line(answer: str) -> str:
    m = re.search(r"answer:\s*(.+)$", answer, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else answer.strip().splitlines()[-1] if answer.strip() else ""


def _run_code_test(answer: str, test: str, func_hint: str):
    code = _extract_code(answer)
    if not code:
        return False, "no code block"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return False, "no function defined"
    target = next((n for n in funcs if func_hint in n.lower()), funcs[0])
    harness = code + f"\nf = {target}\n" + test + "\nprint('CODE_TEST_PASS')"
    try:
        r = subprocess.run([sys.executable, "-c", harness],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and "CODE_TEST_PASS" in r.stdout:
            return True, "pass"
        return False, (r.stderr.strip().splitlines() or ["failed"])[-1]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def grade(answer: str, spec: dict):
    low = (answer or "").lower()
    reasons = []

    for key in ("require_all",):
        for s in spec.get(key, []):
            if s.lower() not in low:
                reasons.append(f"missing required '{s}'")
    for key in ("require_any", "require_any_2", "require_any_3"):
        opts = spec.get(key)
        if opts and not any(o.lower() in low for o in opts):
            reasons.append(f"none of {opts} present")

    if "label_any" in spec:
        head = low[:80]
        if not any(l in head for l in spec["label_any"]):
            reasons.append(f"no acceptable label in {spec['label_any']} near start")
    if "label_forbid_leading" in spec:
        head = low[:40]
        bad = spec["label_forbid_leading"]
        # a leading bare Negative fails; "negative aspects" later is fine
        if re.match(rf"\W*{bad}\b", head):
            reasons.append(f"leading forbidden label '{bad}'")

    if "sentences" in spec:
        n = len(agent.split_sentences(answer))
        if n != spec["sentences"]:
            reasons.append(f"{n} sentences, expected {spec['sentences']}")
    if "bullets" in spec:
        bullets = agent.split_bullets(answer)
        if len(bullets) != spec["bullets"]:
            reasons.append(f"{len(bullets)} bullets, expected {spec['bullets']}")
        mw = spec.get("bullet_max_words")
        if mw and any(len(b.split()) > mw for b in bullets):
            reasons.append(f"a bullet exceeds {mw} words")
    if "max_words" in spec:
        n = len((answer or "").split())
        if n > spec["max_words"]:
            reasons.append(f"{n} words, max {spec['max_words']}")

    if "numeric" in spec:
        v = agent.extract_number(answer)
        want = spec["numeric"]
        if v is None or not (abs(v - want) < 1e-6 or abs(round(v, 2) - round(want, 2)) < 1e-9):
            reasons.append(f"numeric {v} != {want}")

    if "answer_line_contains" in spec:
        if spec["answer_line_contains"].lower() not in _answer_line(answer).lower():
            reasons.append(f"answer line lacks '{spec['answer_line_contains']}'")
    if "answer_line_regex" in spec:
        if not re.search(spec["answer_line_regex"], _answer_line(answer), re.IGNORECASE):
            reasons.append("answer line regex mismatch")

    if "code_test" in spec:
        ok, why = _run_code_test(answer, spec["code_test"], spec.get("code_func_hint", ""))
        if not ok:
            reasons.append(f"code test: {why}")
    if "code_contains_any" in spec:
        code = _extract_code(answer)
        if not any(s in code for s in spec["code_contains_any"]):
            reasons.append(f"code lacks any of {spec['code_contains_any']}")

    return (len(reasons) == 0), reasons


def main():
    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        specs = json.load(f)

    gate_min = int(os.environ.get("GATE_MIN_CORRECT", "20"))
    gate_max_s = float(os.environ.get("GATE_MAX_SECONDS", "420"))

    t0 = time.time()
    local = agent.Local()

    report, correct = [], 0
    for spec in specs:
        tid, prompt = spec["task_id"], spec["prompt"]
        cat = agent.classify(prompt)
        ts = time.time()
        try:
            answer, info = agent.DISPATCH[cat](prompt, local, False)
        except Exception as exc:
            answer, info = "", {"verified": f"error: {exc}"}
        dt = time.time() - ts
        ok, reasons = grade(answer, spec["grade"])
        correct += ok
        report.append({"task_id": tid, "category": cat, "ok": ok,
                       "seconds": round(dt, 1), "reasons": reasons,
                       "verified": info.get("verified"), "answer": answer})
        print(f"{'PASS' if ok else 'FAIL'} {tid:<14} {cat:<14} {dt:6.1f}s "
              f"{'; '.join(reasons)[:100]}", flush=True)

    total_s = time.time() - t0
    summary = {"correct": correct, "total": len(specs),
               "seconds": round(total_s, 1),
               "gate_min_correct": gate_min, "gate_max_seconds": gate_max_s,
               "pass": correct >= gate_min and total_s <= gate_max_s}
    with open(os.path.join(ROOT, "devset_report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "tasks": report}, f, indent=2, ensure_ascii=False)

    print(f"\nGATE: {correct}/{len(specs)} correct in {total_s:.0f}s -> "
          f"{'PASS' if summary['pass'] else 'FAIL'}", flush=True)
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
