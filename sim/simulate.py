#!/usr/bin/env python3
"""End-to-end simulation suite for TokenCascade v6 (mock llama_cpp).

S1  classifier — all 8 official practice prompts route correctly, plus the
    18-prompt external validation set routes to sane categories
S2  happy path — full run on practice tasks: valid results.json, every
    task_id present, zero fallbacks, log written
S3  math self-consistency — NL derivation forced wrong; the executed code
    path must catch the disagreement and override the number
S4  code repair loop — first attempt forced to a SyntaxError; the retry
    with the error fed back must produce compiling code
S5  summarization format — constraint violation forced; retry/repair must
    deliver exactly one sentence
S6  NER repair — JSON wrapped in prose; output must still be a valid array
S7  sentiment surfacing — label buried mid-answer must be surfaced to front
S8  SIGTERM mid-run — results.json must already be complete (real answers
    for finished tasks, fallbacks for the rest) and process exits 0
S9  model load failure — run exits 0 with a complete all-fallback file

Run: python3 sim/simulate.py   (from repo root)
"""
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
OUT = "/tmp/v6sim"
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def run_agent(extra_env=None, timeout=120, input_path=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = SIM  # mock llama_cpp shadows the real one
    env["INPUT_PATH"] = input_path or os.path.join(ROOT, "practice", "tasks.json")
    env["OUTPUT_PATH"] = os.path.join(OUT, "results.json")
    env["LOG_PATH"] = os.path.join(OUT, "inference_log.json")
    env["LOCAL_MODEL_PATH"] = "/nonexistent-mock.gguf"
    env.update(extra_env or {})
    for f in ("results.json", "inference_log.json"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                       capture_output=True, text=True, timeout=timeout, env=env)
    results = None
    p = os.path.join(OUT, "results.json")
    if os.path.exists(p):
        with open(p) as f:
            results = json.load(f)
    return r, results


def by_id(results):
    return {r["task_id"]: r["answer"] for r in (results or [])}


# ---------------------------------------------------------------- S1
sys.path.insert(0, ROOT)
import main as agent  # noqa: E402

expected = {
    "practice-01": "factual", "practice-02": "math",
    "practice-03": "sentiment", "practice-04": "summarization",
    "practice-05": "ner", "practice-06": "code_debug",
    "practice-07": "logic", "practice-08": "code_gen",
}
practice = json.load(open(os.path.join(ROOT, "practice", "tasks.json")))
miss = [(t["task_id"], agent.classify(t["prompt"]))
        for t in practice if agent.classify(t["prompt"]) != expected[t["task_id"]]]
check("S1a classifier practice 8/8", not miss, str(miss))

external = [
    ("Explain why SSDs generally provide faster random access than HDDs.", "factual"),
    ("Why does HTTPS provide better security than HTTP? Briefly explain.", "factual"),
    ("Summarize the passage as exactly three bullet points:\n\nA wildlife...", "summarization"),
    ("Summarize in no more than 20 words:\n\nScientists created...", "summarization"),
    ("Summarize the passage in exactly two sentences:\n\nMany schools...", "summarization"),
    ("Return named entities as JSON:\n\nApple CEO Tim Cook met Prime Minister Narendra Modi in New Delhi on 20 June 2025.", "ner"),
    ("Four employees—Jack, Kim, Leo, and Maya—sit in a row. Kim sits immediately to the left of Leo. Maya sits somewhere to the right of Leo. Jack is not at either end. Determine the seating order.", "logic"),
    ("Write a Python function that returns True if a string is a palindrome, ignoring spaces and letter case.", "code_gen"),
    ("Implement a Python function that merges two already sorted lists into one sorted list.", "code_gen"),
    ("Write a Python function that counts the frequency of each word in a sentence.", "code_gen"),
    ("A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?", "math"),
    ("Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived late but support was great.'", "sentiment"),
]
ext_miss = [(p[:40], agent.classify(p), want)
            for p, want in external if agent.classify(p) != want]
check("S1b classifier external 12/12", not ext_miss, str(ext_miss))

# ---------------------------------------------------------------- S2
r, results = run_agent()
ids = by_id(results)
check("S2a exit 0 + valid results.json", r.returncode == 0 and results is not None)
check("S2b all 8 task_ids present, ordered",
      results is not None and [x["task_id"] for x in results] == [t["task_id"] for t in practice])
check("S2c zero fallbacks",
      results is not None and all(a != agent.FALLBACK for a in ids.values()),
      str([k for k, a in ids.items() if a == agent.FALLBACK]))
check("S2d math answer verified = 144", "144" in ids.get("practice-02", ""))
check("S2e log written with 0 fireworks tokens",
      os.path.exists(os.path.join(OUT, "inference_log.json")) and
      json.load(open(os.path.join(OUT, "inference_log.json")))["fireworks_tokens"] == 0)

# ---------------------------------------------------------------- S3
r, results = run_agent({"SIM_MATH_NL_WRONG": "1"})
ans = by_id(results).get("practice-02", "")
check("S3 math: executed code overrides wrong NL (144 not 150)",
      "144" in ans.split("Answer:")[-1] and "150" not in ans.split("Answer:")[-1], ans[-60:])

# ---------------------------------------------------------------- S4
r, results = run_agent({"SIM_CODE_BROKEN_FIRST": "1"})
ids = by_id(results)
code8 = agent.extract_code(ids.get("practice-08", ""))
code6 = agent.extract_code(ids.get("practice-06", ""))
ok8 = ok6 = False
import ast as _ast
try:
    _ast.parse(code8); ok8 = True
except SyntaxError:
    pass
try:
    _ast.parse(code6); ok6 = True
except SyntaxError:
    pass
check("S4 code repair loop yields compiling code (gen+debug)", ok8 and ok6)

# ---------------------------------------------------------------- S5
r, results = run_agent({"SIM_SUMM_VIOLATE": "1"})
summ = by_id(results).get("practice-04", "")
check("S5 summarization delivers exactly 1 sentence",
      len(agent.split_sentences(summ)) == 1, summ[:80])

# ---------------------------------------------------------------- S6
ner_tasks = os.path.join(OUT, "ner_tasks.json")
json.dump([{"task_id": "n1",
            "prompt": "Return named entities as JSON:\n\nMaria Sanchez joined Fireworks AI in Berlin last March."}],
          open(ner_tasks, "w"))
r, results = run_agent({"SIM_NER_GARBAGE": "1"}, input_path=ner_tasks)
ner_ans = by_id(results).get("n1", "")
ner_ok = False
try:
    parsed = json.loads(ner_ans)
    ner_ok = isinstance(parsed, list) and len(parsed) == 4
except json.JSONDecodeError:
    pass
check("S6 NER repair extracts valid 4-entity JSON from prose", ner_ok, ner_ans[:60])

# ---------------------------------------------------------------- S7
r, results = run_agent({"SIM_SENT_BURIED": "1"})
sent = by_id(results).get("practice-03", "")
check("S7 buried sentiment label surfaced to front",
      sent.lower().startswith("mixed"), sent[:60])

# ---------------------------------------------------------------- S8
env = dict(os.environ)
env["PYTHONPATH"] = SIM
env["INPUT_PATH"] = os.path.join(ROOT, "practice", "tasks.json")
env["OUTPUT_PATH"] = os.path.join(OUT, "results.json")
env["LOG_PATH"] = os.path.join(OUT, "inference_log.json")
env["SIM_SLOW_S"] = "0.8"
if os.path.exists(env["OUTPUT_PATH"]):
    os.remove(env["OUTPUT_PATH"])
proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "main.py")],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3.0)  # a few tasks in
proc.send_signal(signal.SIGTERM)
rc = proc.wait(timeout=15)
sig_results = json.load(open(env["OUTPUT_PATH"]))
sig_ids = by_id(sig_results)
real = sum(1 for a in sig_ids.values() if a != agent.FALLBACK)
check("S8 SIGTERM: exit 0, all 8 ids on disk, >=1 real answer banked",
      rc == 0 and len(sig_results) == 8 and real >= 1,
      f"rc={rc} ids={len(sig_results)} real={real}")

# ---------------------------------------------------------------- S9
r, results = run_agent({"SIM_LOAD_FAIL": "1"})
check("S9 model-load failure: exit 0 + complete all-fallback file",
      r.returncode == 0 and results is not None and len(results) == 8 and
      all(x["answer"] == agent.FALLBACK for x in results))

# ---------------------------------------------------------------- S10
r, results = run_agent({"SIM_LOGIC_TRUNCATED": "1"})
logic_ans = by_id(results).get("practice-07", "")
check("S10 truncated logic salvaged via conclude call",
      "answer:" in logic_ans.lower() and "sam" in logic_ans.lower().split("answer:")[-1],
      logic_ans[-50:])

# ---------------------------------------------------------------- S11
r, results = run_agent({"SIM_FACT_WRONG": "1"})
fact_ans = by_id(results).get("practice-01", "")
check("S11 factual self-check flags swap and ships correction",
      "burley griffin" in fact_ans.lower(), fact_ans[:70])

# ----------------------------------------------------------------
print(f"\nOVERALL: {len(PASS)} pass, {len(FAIL)} fail "
      f"-> {'ALL SCENARIOS PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(0 if not FAIL else 1)
