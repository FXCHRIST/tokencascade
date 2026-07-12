#!/usr/bin/env python3
"""TokenCascade — AMD Developer Hackathon ACT II, Track 1.

Pure local, zero-Fireworks-token agent. Every task in /input/tasks.json is
answered by a bundled Qwen3-4B-Instruct-2507 GGUF running on CPU via
llama.cpp. No remote inference exists in this codebase, so the scored token
count is 0 by construction.

Design laws (carried over from the v4 post-mortem, still binding):

  L1. No native-code interruption. Local generations run to completion,
      bounded by small max_tokens. Never stream-and-abandon llama.cpp.
  L2. No threads. Strictly sequential; every line debuggable from a log.
  L3. Results are flushed to disk ATOMICALLY AFTER EVERY TASK; SIGTERM
      converts to flush-and-exit-0. A crash at task 12 leaves 12 answers,
      and every task_id is always present in the output.
  L4. Exact pins. The dependency set is the one that demonstrably ran.
  L5. Free compute does verification: math is re-computed by Python from a
      model-proposed expression, code must compile AND execute, and
      constrained formats (exact sentence/bullet counts) are validated and
      regenerated on violation. Zero tokens, real accuracy.
"""

import ast
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Configuration (env-tunable, sane defaults for the judging harness)
# --------------------------------------------------------------------------

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
LOG_PATH = os.environ.get("LOG_PATH", "/output/inference_log.json")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")

TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "520"))
RESERVE_S = float(os.environ.get("RESERVE_S", "40"))
THREADS = int(os.environ.get("THREADS", str(os.cpu_count() or 2)))
N_CTX = int(os.environ.get("N_CTX", "3072"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "6000"))

START = time.time()

SYS_LOCAL = (
    "You are a precise assistant. Answer directly with no preamble, no "
    "self-reference, and no markdown headers. Follow every format "
    "instruction in the task exactly."
)

# Generation caps per category (output tokens). Tokens are free locally;
# these caps exist purely to protect the runtime budget.
CAP = {
    "factual": 220,
    "sentiment": 160,
    "summarization": 260,
    "ner": 260,
    "math": 420,
    "logic": 420,
    "code_debug": 420,
    "code_gen": 420,
}

# Cheap categories run first so a time-out late in the run costs the fewest
# answers (L3 guarantees everything answered so far is already on disk).
CATEGORY_ORDER = ["factual", "sentiment", "ner", "summarization",
                  "math", "logic", "code_debug", "code_gen"]

CANONICAL_CATEGORY = {
    "factual": "factual", "factual_knowledge": "factual",
    "math": "math", "mathematical_reasoning": "math",
    "sentiment": "sentiment", "sentiment_classification": "sentiment",
    "summarization": "summarization", "text_summarization": "summarization",
    "ner": "ner", "named_entity_recognition": "ner",
    "logic": "logic", "logical_reasoning": "logic", "logic_puzzle": "logic",
    "code_gen": "code_gen", "code_generation": "code_gen",
    "code_debug": "code_debug", "code_debugging": "code_debug",
}

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def remaining() -> float:
    return TIME_BUDGET_S - (time.time() - START)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Crash-proof result sink (L3)
# --------------------------------------------------------------------------

class Sink:
    def __init__(self, task_ids):
        self.results = {tid: "" for tid in task_ids}
        self.meta = {"fireworks_tokens": 0, "routes": {},
                     "task_seconds": {}, "notes": []}
        atexit.register(self.flush)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except Exception:
                pass

    def _on_signal(self, signum, frame):
        log(f"[sink] signal {signum} — flushing and exiting 0")
        self.flush()
        os._exit(0)

    def set(self, tid, answer, route, seconds):
        self.results[tid] = answer
        self.meta["routes"][tid] = route
        self.meta["task_seconds"][tid] = round(seconds, 1)
        self.flush()

    def flush(self):
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            payload = [{"task_id": t, "answer": a}
                       for t, a in self.results.items()]
            tmp = OUTPUT_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, OUTPUT_PATH)
            self.meta["elapsed_s"] = round(time.time() - START, 1)
            self.meta["empty_answers"] = [t for t, a in self.results.items()
                                          if not a]
            with open(LOG_PATH, "w") as f:
                json.dump(self.meta, f, indent=2)
        except Exception as e:
            log(f"[sink] flush failed: {e}")


# --------------------------------------------------------------------------
# Task loading and classification
# --------------------------------------------------------------------------

def load_tasks(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", data.get("data", []))
    tasks = []
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            t = {"prompt": str(t)}
        tid = str(t.get("task_id", t.get("id", f"task-{i + 1}")))
        prompt = str(t.get("prompt", t.get("input", t.get("question",
                     t.get("task", ""))))).strip()
        cat_raw = str(t.get("category", t.get("type", ""))).strip().lower()
        tasks.append({"task_id": tid, "prompt": prompt,
                      "category": CANONICAL_CATEGORY.get(cat_raw, "")})
    return tasks


CATEGORY_RULES = [
    ("sentiment", r"\bsentiment\b|classify .{0,40}(review|tweet|comment)"
                  r"|\bpositive, negative"),
    ("ner", r"named entit|entities and their types|extract .{0,60}entit"
            r"|label each as|\bidentify the (people|persons|organizations)\b"),
    ("summarization", r"\bsummar(y|ise|ize|iz)|\bTL;DR\b|\bshorten\b"
                      r"|\bmain point\b|\bbullet point"),
    ("code_debug", r"(bug|fix|broken|incorrect|error|exception|crash)"
                   r".{0,160}(def |function|code|```)"
                   r"|(def |function|```).{0,200}(bug|fix|broken|crash)"),
    ("code_gen", r"\bwrite (a |an )?\w{0,14}\s?(python )?"
                 r"(function|program|script|class|method)\b"
                 r"|\bcreate a (python|script|function)\b"
                 r"|\bimplement\b.{0,40}\b(function|algorithm)\b"),
    ("logic", r"each own|who owns|exactly one|puzzle|deduce|riddle"
              r"|all (the )?conditions|must be satisfied"
              r"|taller than|shorter than|older than|younger than"
              r"|finished (before|after|first|last)"
              r"|who is the (shortest|tallest|oldest|youngest|first|last)"),
    ("math", r"\bhow (many|much)\b.*\d|\d+\s*%|\bpercent|\bcalculate\b"
             r"|\baverage\b.*\d|\bremain\b.*\d|\d.*\bremain\b"
             r"|\bcost\b.*\d|\bprice\b.*\d|\bliters?\b.*\d|\btotal\b.*\d"),
]


def classify(task) -> str:
    if task["category"]:
        return task["category"]
    p = task["prompt"].lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, p, re.DOTALL):
            return cat
    if re.search(r"\d", p) and re.search(
            r"total|left|per hour|speed|discount|sold|sells", p):
        return "math"
    return "factual"


# --------------------------------------------------------------------------
# Deterministic helpers (L5)
# --------------------------------------------------------------------------

_ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                ast.Mod, ast.Pow, ast.USub, ast.UAdd)


def safe_eval(expr):
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST):
                return None
            if isinstance(node, ast.Constant) and not isinstance(
                    node.value, (int, float)):
                return None
        return eval(compile(tree, "<expr>", "eval"))
    except Exception:
        return None


def fmt_num(v):
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, float):
        return f"{round(v, 4):g}"
    return str(v)


def strip_think(text):
    """Safety net only — the bundled model is a non-thinking instruct model,
    but this keeps a model swap from ever leaking <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned if cleaned else text.strip()


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    if "def " in text:
        return text[text.index("def "):].strip()
    return ""


def undefined_names(code):
    """Conservative AST lint: names that are loaded somewhere but bound
    nowhere in the module and are not builtins. Catches NameErrors hiding
    inside function bodies, which mere compilation and definition miss."""
    import builtins
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    bound = set(dir(builtins)) | {"self", "cls"}
    loaded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            else:
                bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            a = node.args
            for arg in (a.args + a.posonlyargs + a.kwonlyargs
                        + ([a.vararg] if a.vararg else [])
                        + ([a.kwarg] if a.kwarg else [])):
                bound.add(arg.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return sorted(loaded - bound)


def python_runs(code, timeout=8):
    """Compile, lint, AND execute the candidate in a subprocess.
    Returns (ok, err)."""
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    undef = undefined_names(code)
    if undef:
        return False, f"undefined name(s): {', '.join(undef)}"
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return False, (r.stderr or "runtime error").strip()[-400:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "execution timed out"
    except Exception as e:
        return False, str(e)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def bullet_lines(text):
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^([-*•]|\d+[.)])\s+", s):
            lines.append(re.sub(r"^([-*•]|\d+[.)])\s+", "", s).strip())
    return lines


def parse_format_constraints(prompt):
    p = prompt.lower()
    c = {}
    m = re.search(r"exactly\s+(\w+)\s+sentence", p)
    if m:
        c["sentences"] = WORD_NUM.get(m.group(1), None) or (
            int(m.group(1)) if m.group(1).isdigit() else None)
    m = re.search(r"exactly\s+(\w+)\s+bullet", p)
    if m:
        c["bullets"] = WORD_NUM.get(m.group(1), None) or (
            int(m.group(1)) if m.group(1).isdigit() else None)
    m = re.search(r"no (?:longer|more) than\s+(\d+)\s+words", p)
    if m:
        c["max_words"] = int(m.group(1))
    m = re.search(r"(\d+)\s+words or (?:fewer|less)", p)
    if m:
        c["max_words"] = int(m.group(1))
    return {k: v for k, v in c.items() if v}


def format_ok(answer, c):
    if "bullets" in c:
        bl = bullet_lines(answer)
        if len(bl) != c["bullets"]:
            return False
        if "max_words" in c and any(len(b.split()) > c["max_words"]
                                    for b in bl):
            return False
        return True
    if "sentences" in c:
        return len(split_sentences(answer)) == c["sentences"]
    return True


def format_repair(answer, c):
    """Last-resort deterministic repair after regeneration attempts."""
    if "bullets" in c:
        bl = bullet_lines(answer) or split_sentences(answer)
        n, k = c["bullets"], c.get("max_words", 0)
        bl = bl[:n]
        while len(bl) < n:
            bl.append(bl[-1] if bl else "See passage.")
        if k:
            bl = [" ".join(b.split()[:k]).rstrip(",;") for b in bl]
        return "\n".join("- " + b for b in bl)
    if "sentences" in c:
        sents = split_sentences(answer)
        n = c["sentences"]
        if len(sents) > n:
            head = sents[:n - 1] if n > 1 else []
            tail = " ".join(s.rstrip(".!?") + ";" for s in sents[n - 1:-1])
            last = (tail + " " + sents[-1]).strip() if tail else sents[-1]
            return " ".join(head + [last])
    return answer


# --------------------------------------------------------------------------
# Local model (the only inference path — zero Fireworks tokens)
# --------------------------------------------------------------------------

class Local:
    def __init__(self):
        from llama_cpp import Llama
        t0 = time.time()
        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            n_batch=256,
            verbose=False,
        )
        # Warm-up primes caches so the first real task isn't penalised.
        self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
        self.avg_task_s = 15.0
        log(f"[local] model loaded in {time.time() - t0:.1f}s, "
            f"threads={THREADS}, ctx={N_CTX}")

    def gen(self, user, cap, system=SYS_LOCAL):
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            repeat_penalty=1.05,
            max_tokens=cap,
        )
        return strip_think(
            (out["choices"][0]["message"]["content"] or "").strip())


# --------------------------------------------------------------------------
# Category pipelines
# --------------------------------------------------------------------------

class Agent:
    def __init__(self, local):
        self.local = local
        self.fast = False  # set True by the governor when time runs low

    def _cap(self, cat):
        cap = CAP.get(cat, 260)
        return min(cap, 140) if self.fast else cap

    def factual(self, prompt):
        return self.local.gen(
            prompt + "\n\nAnswer every part of the question directly and "
            "name the exact entities or facts requested.", self._cap("factual"))

    def sentiment(self, prompt):
        ask = (prompt + "\n\nRules: reply with exactly one label "
               "(Positive, Negative, Neutral, or Mixed) followed by a "
               "one-sentence reason. If the text contains BOTH good and bad "
               "points, the label must be Mixed and the reason must name one "
               "specific positive detail and one specific negative detail.")
        ans = self.local.gen(ask, self._cap("sentiment"))
        has_contrast = re.search(r"\bbut\b|\bhowever\b|\balthough\b|\byet\b",
                                 prompt, re.I)
        label = re.search(r"\b(positive|negative|neutral|mixed)\b", ans, re.I)
        if (has_contrast and label and label.group(1).lower() == "negative"
                and not self.fast):
            ans = self.local.gen(
                ask + "\n\nIMPORTANT: this text mixes good and bad points, "
                "so do NOT label it Negative. Use Mixed (or Neutral) and "
                "mention one detail from each side.",
                self._cap("sentiment"))
        return ans

    def summarization(self, prompt):
        c = parse_format_constraints(prompt)
        ans = self.local.gen(
            prompt + "\n\nObey the length/format constraint exactly. Cover "
            "both the positives/opportunities and the "
            "challenges/concerns in the passage.",
            self._cap("summarization"))
        tries = 0
        while c and not format_ok(ans, c) and tries < 2 and not self.fast:
            tries += 1
            fix = []
            if "sentences" in c:
                fix.append(f"exactly {c['sentences']} sentence(s) — count "
                           "them before answering")
            if "bullets" in c:
                fix.append(f"exactly {c['bullets']} bullet points, each "
                           "starting with '- '")
            if "max_words" in c:
                fix.append(f"each bullet at most {c['max_words']} words")
            ans = self.local.gen(
                prompt + "\n\nYour previous attempt violated the format. "
                "Produce " + " and ".join(fix) + ". Nothing else.",
                self._cap("summarization"))
        if c and not format_ok(ans, c):
            ans = format_repair(ans, c)
        return ans

    def ner(self, prompt):
        ask = (prompt + "\n\nOutput one entity per line in exactly this "
               "format: Entity - LABEL\nAllowed labels: PERSON, "
               "ORGANIZATION, LOCATION, DATE. Include EVERY entity in the "
               "text. No other text.")
        ans = self.local.gen(ask, self._cap("ner"))
        pat = r"-\s*(PERSON|ORGANIZATION|LOCATION|DATE)\b"
        if len(re.findall(pat, ans, re.I)) < 2 and not self.fast:
            ans = self.local.gen(
                ask + "\n\nYour previous output was not in the required "
                "'Entity - LABEL' line format. Redo it correctly.",
                self._cap("ner"))
        return ans

    def math(self, prompt):
        ans = self.local.gen(
            prompt + "\n\nShow the calculation briefly, then end with "
            "'Final answer:' followed by the value(s).", self._cap("math"))
        if self.fast or remaining() < RESERVE_S + self.local.avg_task_s:
            return ans
        expr_raw = self.local.gen(
            f"Problem:\n{prompt}",
            120,
            system=("Output ONLY the arithmetic expression(s) that compute "
                    "the final numeric answer(s), separated by ';'. Python "
                    "syntax, numbers and + - * / % ( ) only. No words."))
        exprs = [e.strip() for e in expr_raw.split(";") if e.strip()]
        values = [v for v in (safe_eval(e) for e in exprs[:4]) if v is not None]
        if not values:
            return ans
        wanted = [fmt_num(v) for v in values]
        norm = ans.replace(",", "").replace("$", "")
        if all(w in norm for w in wanted):
            return ans
        log(f"[math] mismatch — model text vs computed {wanted}; regenerating")
        hinted = self.local.gen(
            prompt + f"\n\n(The correct computed value(s): "
            f"{', '.join(wanted)}. Show brief working and state them "
            "plainly, ending with 'Final answer:'.)", self._cap("math"))
        if all(w in hinted.replace(",", "").replace("$", "") for w in wanted):
            return hinted
        return "Final answer: " + " and ".join(wanted)

    def logic(self, prompt):
        ans = self.local.gen(
            prompt + "\n\nReason step by step briefly, checking every "
            "condition. End with 'Final answer:' followed by the complete "
            "assignment or ordering.", self._cap("logic"))
        m = re.search(r"final answer\s*:\s*(.+)", ans, re.I | re.DOTALL)
        if m and len(m.group(1).strip()) >= 10:
            return m.group(1).strip()
        return ans

    def _code(self, prompt, cat, instruction):
        ask = prompt + "\n\n" + instruction
        ans = self.local.gen(ask, self._cap(cat))
        code = extract_code(ans)
        ok, err = python_runs(code) if code else (False, "no code block")
        if not ok and not self.fast:
            retry = self.local.gen(
                ask + f"\n\nYour previous attempt failed with: {err}\n"
                "Return the complete corrected code in one ```python block.",
                self._cap(cat))
            rcode = extract_code(retry)
            rok, _ = python_runs(rcode) if rcode else (False, "")
            if rok:
                return rcode
            code = rcode or code
        return code if code else ans

    def code_gen(self, prompt):
        return self._code(
            prompt, "code_gen",
            "Write ONLY the Python code in a single ```python code block. "
            "Include the exact function name requested. No explanation.")

    def code_debug(self, prompt):
        return self._code(
            prompt, "code_debug",
            "Return the fully corrected code in a single ```python code "
            "block. Fix the bug, change nothing else. No explanation.")

    def answer(self, prompt, cat):
        prompt = prompt[:MAX_PROMPT_CHARS]
        fn = {"factual": self.factual, "sentiment": self.sentiment,
              "summarization": self.summarization, "ner": self.ner,
              "math": self.math, "logic": self.logic,
              "code_gen": self.code_gen, "code_debug": self.code_debug}
        return fn.get(cat, self.factual)(prompt)


# --------------------------------------------------------------------------
# Main run loop
# --------------------------------------------------------------------------

def run() -> int:
    tasks = load_tasks(INPUT_PATH)
    sink = Sink([t["task_id"] for t in tasks])
    log(f"[run] {len(tasks)} tasks loaded")

    try:
        local = Local()
    except Exception as e:
        log(f"[fatal] local model unavailable: {e}")
        sink.meta["notes"].append(f"model load failed: {e}")
        sink.flush()
        return 0

    agent = Agent(local)
    cats = {t["task_id"]: classify(t) for t in tasks}

    ordered = sorted(
        tasks, key=lambda t: CATEGORY_ORDER.index(cats[t["task_id"]])
        if cats[t["task_id"]] in CATEGORY_ORDER else 99)

    done = 0
    for t in ordered:
        tid = t["task_id"]
        cat = cats[tid]
        left = remaining()
        if left < 8:
            log(f"[governor] {left:.0f}s left — stopping generation")
            break
        if not agent.fast and left - RESERVE_S < local.avg_task_s * 1.5:
            agent.fast = True
            log(f"[governor] fast mode ON at {left:.0f}s remaining")
        t0 = time.time()
        try:
            text = agent.answer(t["prompt"], cat)
        except Exception as e:
            log(f"[task] {tid} ({cat}) failed: {e}")
            text = ""
        dt = time.time() - t0
        done += 1
        local.avg_task_s = local.avg_task_s * 0.6 + dt * 0.4
        sink.set(tid, text, f"local:{cat}", dt)
        log(f"[task] {tid} ({cat}) done in {dt:.1f}s — "
            f"{remaining():.0f}s remaining")

    sink.meta["fireworks_tokens"] = 0
    sink.flush()
    log(f"[done] {done}/{len(tasks)} generated, 0 fireworks tokens, "
        f"{time.time() - START:.1f}s")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as e:
        log(f"[fatal] {type(e).__name__}: {e} — writing salvage output")
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            if not os.path.exists(OUTPUT_PATH):
                with open(OUTPUT_PATH, "w") as f:
                    json.dump([], f)
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
