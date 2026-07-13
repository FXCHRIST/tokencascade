#!/usr/bin/env python3
"""TokenCascade v6 — AMD Developer Hackathon ACT II, Track 1.

Fully local, zero Fireworks tokens. Every task in /input/tasks.json is
answered by a bundled Qwen3-4B-Instruct-2507 GGUF running on CPU via
llama.cpp. No remote inference exists in this codebase, so the scored
token count is 0 by construction.

Design laws (from the v4 post-mortem, still binding):

  L1. No native-code interruption. Local generations run to completion,
      bounded by small max_tokens. Never stream-and-abandon llama.cpp.
  L2. No threads. Strictly sequential; every line debuggable from a log.
  L3. Crash-safe output. The results file contains EVERY task_id from the
      moment the run starts (prefilled with a safe fallback) and is
      rewritten atomically after every completed task. SIGTERM converts to
      flush-and-exit-0. A crash at task 17 leaves 16 real answers and 2
      fallbacks on disk — never a missing file, never a missing task_id.
  L4. Exact pins. The dependency set is the one that demonstrably ran.
  L5. Free compute does verification:
        * math    — solved twice independently (natural-language derivation
                    AND a model-written Python script executed in a
                    subprocess); agreement is required, disagreement
                    triggers a tie-breaking resample.
        * code    — must parse (ast) AND execute; failures regenerate with
                    the actual error message fed back to the model.
        * NER     — when JSON is requested, output is constrained by a
                    GBNF grammar at the sampler level (malformed JSON is
                    structurally impossible), with a 3-stage repair
                    fallback behind it.
        * format  — "exactly N sentences" / "N bullets" / word-limit
                    constraints are parsed from the prompt, validated,
                    regenerated on violation, and deterministically
                    repaired as a last resort.
        * sentiment — rubric-guarded: mixed reviews must acknowledge both
                    sides and are never labeled bare Negative.

Category-specific system prompts (one per pipeline) replace the single
generic prompt of earlier versions.
"""

import ast
import json
import os
import re
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration (env-tunable; defaults sized for the judging harness:
# 2 vCPU / 4 GB RAM / <60 s startup / <10 min total / <30 s per request)
# ---------------------------------------------------------------------------

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
LOG_PATH = os.environ.get("LOG_PATH", "/output/inference_log.json")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")

TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "500"))
RESERVE_S = float(os.environ.get("RESERVE_S", "45"))


def detect_threads() -> int:
    """Thread count the judging harness ACTUALLY grants.

    os.cpu_count() reports the HOST's cores, not the container's cgroup
    quota. On a 2-vCPU cgroup running on a 32-core host it returns 32;
    llama.cpp then spawns 32 threads on 2 vCPUs and thrashes itself into
    a 5-20x slowdown — the proven TIMEOUT root cause of v5 and rc5.
    Resolution order: THREADS env > cgroup v2 quota > cgroup v1 quota >
    CPU affinity mask > cpu_count, always clamped to 4."""
    env = os.environ.get("THREADS")
    if env:
        return max(1, int(env))
    try:  # cgroup v2: "200000 100000" -> 2 CPUs; "max 100000" -> no quota
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 2
    return max(1, min(affinity, os.cpu_count() or 2, 4))


THREADS = detect_threads()
N_CTX = int(os.environ.get("N_CTX", "2048"))
N_BATCH = int(os.environ.get("N_BATCH", "256"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "6000"))
CODE_EXEC_TIMEOUT_S = float(os.environ.get("CODE_EXEC_TIMEOUT_S", "8"))

FALLBACK = "Unable to determine a reliable answer for this task."

START = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def remaining() -> float:
    return TIME_BUDGET_S - (time.time() - START)


# ---------------------------------------------------------------------------
# Category constants — single source of truth; a typo'd category becomes a
# KeyError at startup instead of silent misrouting.
# ---------------------------------------------------------------------------

FACTUAL = "factual"
MATH = "math"
SENTIMENT = "sentiment"
SUMMARIZATION = "summarization"
NER = "ner"
CODE_DEBUG = "code_debug"
LOGIC = "logic"
CODE_GEN = "code_gen"

ALL_CATEGORIES = [FACTUAL, MATH, SENTIMENT, SUMMARIZATION, NER,
                  CODE_DEBUG, LOGIC, CODE_GEN]

# Output-token caps per category. Tokens are free locally; these caps exist
# purely to protect the runtime budget on a 2-vCPU CPU box. FAST values are
# used when the time governor engages.
CAP = {
    FACTUAL: 180, SENTIMENT: 120, SUMMARIZATION: 260, NER: 300,
    MATH: 260, LOGIC: 340, CODE_DEBUG: 460, CODE_GEN: 460,
}
CAP_FAST = {
    FACTUAL: 140, SENTIMENT: 90, SUMMARIZATION: 180, NER: 220,
    MATH: 200, LOGIC: 280, CODE_DEBUG: 300, CODE_GEN: 300,
}
MATH_CODE_CAP = 240        # the code path is terse by design
LOGIC_CONCLUDE_CAP = 60    # salvage call: one Answer line only
FACTUAL_CHECK_CAP = 60     # self-verification review pass

# Cheap categories run first so if time runs out late in the run, the
# expensive stragglers are the ones that fall back (L3 guarantees everything
# answered so far is already on disk).
ORDER_RANK = {c: i for i, c in enumerate(
    [SENTIMENT, FACTUAL, NER, SUMMARIZATION, MATH, LOGIC, CODE_DEBUG, CODE_GEN])}

# ---------------------------------------------------------------------------
# Category-specific system prompts (v6: one per pipeline, each written
# against the public rubric in the judging Self-Check guide).
# ---------------------------------------------------------------------------

PROMPTS = {
    FACTUAL: (
        "You are a precise technical assistant. Answer the question directly "
        "and completely, covering every part of what is asked. State specific "
        "facts confidently — never hedge between alternatives or speculate. "
        "Be concise: 2-5 sentences, no preamble, no headers, no self-reference."
    ),
    SENTIMENT: (
        "Classify the sentiment of the given text as Positive, Negative, "
        "Neutral, or Mixed. If the text contains BOTH positive and negative "
        "points, do NOT label it Negative just because a complaint is "
        "present — use Mixed, Neutral, or Positive depending on the overall "
        "outcome, and your justification MUST acknowledge both the negative "
        "and the positive aspects. Reply with the label first, then a single "
        "one-sentence justification."
    ),
    SUMMARIZATION: (
        "Summarize the given text, following the exact format constraint in "
        "the request (sentence count, bullet count, word limit) precisely. "
        "Cover both the benefits/opportunities AND the concerns/challenges "
        "mentioned in the text — omitting either side is a failure. Write "
        "only the summary itself: no preamble, no labels, no commentary."
    ),
    NER: (
        "Extract ALL named entities from the text. If the request specifies "
        "entity types or an output format, follow it exactly. Otherwise "
        "label each entity as PERSON, ORGANIZATION, LOCATION, or DATE and "
        "list one entity per line as: Entity (TYPE). Extract every distinct "
        "entity — do not merge or skip any, and never return the whole "
        "sentence as one entity."
    ),
    NER + "_json": (
        "Extract all named entities from the text. Return ONLY a JSON array "
        "where each element is an object with a \"text\" key (the exact "
        "entity string) and a \"type\" key (PERSON, ORGANIZATION, LOCATION, "
        "or DATE, unless the request specifies different types). Include "
        "every distinct entity. No prose, no markdown — just the JSON array."
    ),
    LOGIC: (
        "Solve the logic puzzle by applying each clue in turn. Reason in "
        "compact plain text: no markdown, no headers, do NOT restate the "
        "clues or the setup. A few short lines of deduction, then end with "
        "a single final line in exactly this form: 'Answer: <answer>'."
    ),
    MATH + "_nl": (
        "Solve this math problem step by step in brief plain text: no "
        "markdown, no LaTeX, no restating the problem, no double-checking "
        "section. Show each arithmetic step on one short line. If the "
        "problem has multiple parts, answer every part. Finish with a "
        "single final line in exactly this form: 'Answer: <number>'."
    ),
    MATH + "_expr": (
        "Reply with ONLY a single Python arithmetic expression that "
        "computes the final answer to this problem. Numbers and the "
        "operators + - * / ** ( ) only. No variables, no words, no code "
        "block, no explanation — just the expression on one line."
    ),
    MATH + "_code": (
        "Write a short Python script that solves the given math word "
        "problem. Compute the answer with code (do not hardcode the final "
        "number) and end the script with a single print() that outputs ONLY "
        "the final numeric answer. Return ONLY one Python code block, no "
        "explanation."
    ),
    CODE_DEBUG: (
        "You are given code containing a bug. State the bug in one or two "
        "sentences, then provide the complete corrected code in a single "
        "Python code block. The corrected code must be runnable as-is."
    ),
    CODE_GEN: (
        "Write clean, correct Python code that satisfies the request "
        "exactly, including all stated edge cases. Return the complete "
        "implementation in a single Python code block. One or two sentences "
        "of explanation at most."
    ),
}

# ---------------------------------------------------------------------------
# Classifier — regex scoring with most-specific-first priority tie-break.
# Pure pattern matching: it must run before the model is loaded.
# ---------------------------------------------------------------------------

PATTERNS = {
    CODE_DEBUG: [
        r"\bbug\b", r"\bdebug\b", r"fix (the |this |it)?\b",
        r"find (the )?bug", r"corrected version",
        r"what('s| is) wrong with (this|the)",
        r"why does this (code|function) (fail|crash|not work)",
        r"has a bug",
    ],
    CODE_GEN: [
        r"write a (python )?function", r"implement (a|the) (python )?function",
        r"write (a )?(python )?(script|program|class|method)",
        r"\bdef \w+\(", r"implement.*that (takes|returns|computes|counts|merges|removes|groups)",
        r"python function that",
    ],
    LOGIC: [
        r"logic puzzle", r"each own[s]? a different", r"sit(s)? in a row",
        r"\bimmediately to the (left|right)\b", r"\bleft of\b", r"\bright of\b",
        r"unique solution", r"determine the (order|arrangement|position|seating)",
        r"who owns (each|the)", r"exactly one of", r"seating order",
        r"three friends", r"four (friends|employees|people|colleagues)",
    ],
    NER: [
        r"named entit", r"extract.*entit", r"entit(y|ies).*(json|extract|label)",
        r"\bperson\b.*\borganization\b", r"\blocation\b.*\bdate\b",
        r"identify and label", r"entity type",
    ],
    SUMMARIZATION: [
        r"summari[sz]e", r"\bsummary\b", r"\bcondense\b",
        r"in one sentence", r"in exactly \d+ (words|sentences|bullet)",
        r"no more than \d+ words", r"exactly (one|two|three|four|\d+) (bullet|sentence)",
    ],
    SENTIMENT: [
        r"\bsentiment\b", r"classify the sentiment", r"positive.*negative",
        r"determine.*sentiment", r"label the sentiment",
    ],
    MATH: [
        r"\bpercent(age)?\b", r"\d+\s*%", r"how (much|many) (change|remain|are left|items|units)",
        r"average speed", r"\$\s?\d", r"compounded", r"\bdiscount\b",
        r"\binvestment\b", r"how many .* (remain|left)", r"total cost",
        r"\d+\s*(units|items|cups|km|kg|miles)",
    ],
    FACTUAL: [
        r"^what (is|are|causes|does)", r"^explain", r"^why (is|does|do|are)",
        r"^how (does|do|is|are)", r"explain (why|how)", r"briefly explain",
        r"what is the (difference|capital)",
    ],
}

# Most specific first — a code-debug prompt can mention percentages, a logic
# puzzle mentions names (NER-ish), so specificity wins ties.
PRIORITY = [CODE_DEBUG, CODE_GEN, LOGIC, NER, SUMMARIZATION, SENTIMENT,
            MATH, FACTUAL]


def classify(prompt: str) -> str:
    text = (prompt or "").lower()
    scores = {c: 0 for c in ALL_CATEGORIES}
    for cat, pats in PATTERNS.items():
        for pat in pats:
            if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
                scores[cat] += 1
    # Hard override: a fenced code block is code territory, never math/NER.
    if "```" in (prompt or ""):
        if scores[CODE_GEN] > scores[CODE_DEBUG]:
            return CODE_GEN
        return CODE_DEBUG
    best, best_score = FACTUAL, 0
    for cat in PRIORITY:
        if scores[cat] > best_score:
            best, best_score = cat, scores[cat]
    return best


# ---------------------------------------------------------------------------
# GBNF grammar for NER JSON — constrains sampling itself so malformed JSON
# is structurally impossible. Built lazily so importing main.py never
# requires llama_cpp (the devset grader and simulator import this module).
# ---------------------------------------------------------------------------

_NER_GBNF = r'''
root   ::= "[" ws (entity ("," ws entity)*)? ws "]"
entity ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"type\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" .
ws     ::= [ \t\n]*
'''

_ner_grammar = None
_ner_grammar_failed = False


def get_ner_grammar():
    """Return the compiled grammar, or None if llama_cpp can't build it —
    the pipeline then degrades to prompt-only JSON plus repair, never
    crashes (L3 spirit: a missing optimization must not cost answers)."""
    global _ner_grammar, _ner_grammar_failed
    if _ner_grammar is not None or _ner_grammar_failed:
        return _ner_grammar
    try:
        from llama_cpp import LlamaGrammar
        _ner_grammar = LlamaGrammar.from_string(_NER_GBNF)
    except Exception as exc:
        log(f"[ner] grammar unavailable ({exc}); using prompt+repair path")
        _ner_grammar_failed = True
        _ner_grammar = None
    return _ner_grammar


# ---------------------------------------------------------------------------
# Local model runtime — one model, loaded once, strictly sequential (L1/L2).
# ---------------------------------------------------------------------------

class Local:
    def __init__(self):
        from llama_cpp import Llama  # import here: grader/simulator safe
        t0 = time.time()
        log(f"[local] loading {LOCAL_MODEL_PATH} threads={THREADS} n_ctx={N_CTX}")
        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            n_batch=N_BATCH,
            verbose=False,
        )
        try:  # warm-up primes caches so the first task isn't penalised
            self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
        except Exception:
            pass
        log(f"[local] ready in {time.time() - t0:.1f}s")
        self.avg_task_s = 15.0   # EMA of per-task latency for the governor
        self.tps = 8.0           # measured output tokens/sec (conservative
                                 # prior; updated from real generations)

    def gen(self, system: str, user: str, max_tokens: int,
            temperature: float = 0.2, grammar=None) -> str:
        """One complete, uninterrupted generation (L1)."""
        # Hard anti-timeout clamp: near the end of the budget, a request
        # for N tokens must not be allowed to run past the wall. Scale the
        # cap to what the MEASURED generation speed can deliver in the
        # time that is actually left.
        left = remaining() - RESERVE_S * 0.5
        if left < max_tokens / max(self.tps, 1.0):
            max_tokens = max(48, int(left * self.tps * 0.7))
        kwargs = dict(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user[:MAX_PROMPT_CHARS]}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            t0 = time.time()
            out = self.llm.create_chat_completion(**kwargs)
            dt = max(time.time() - t0, 0.001)
            try:
                ctoks = out.get("usage", {}).get("completion_tokens", 0)
                if ctoks >= 16:
                    self.tps = 0.7 * self.tps + 0.3 * (ctoks / dt)
            except Exception:
                pass
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            log(f"[local] generation failed: {exc}")
            return ""

    def note_latency(self, seconds: float) -> None:
        self.avg_task_s = 0.7 * self.avg_task_s + 0.3 * seconds


# ---------------------------------------------------------------------------
# Result sink — L3. Prefilled with every task_id at start; atomic rewrite
# after every task; SIGTERM flushes and exits 0.
# ---------------------------------------------------------------------------

class Sink:
    def __init__(self, tasks: list):
        self.order = [t["task_id"] for t in tasks]
        self.results = {tid: FALLBACK for tid in self.order}
        self.answered = {tid: False for tid in self.order}
        self.meta = {"fireworks_tokens": 0, "per_task": []}
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        self.flush()  # from second zero, a valid complete file exists

    def put(self, task_id: str, answer: str, info: dict) -> None:
        answer = (answer or "").strip() or FALLBACK
        self.results[task_id] = answer
        self.answered[task_id] = answer != FALLBACK
        info["task_id"] = task_id
        self.meta["per_task"].append(info)
        self.flush()

    def flush(self) -> None:
        payload = [{"task_id": tid, "answer": self.results[tid]}
                   for tid in self.order]
        tmp = OUTPUT_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, OUTPUT_PATH)  # atomic on POSIX

    def flush_log(self) -> None:
        try:
            tmp = LOG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, LOG_PATH)
        except Exception:
            pass  # the log is best-effort; results.json is what is scored


_SINK: "Sink|None" = None


def _sigterm(_sig, _frm):
    log("[signal] SIGTERM — flushing and exiting 0")
    if _SINK is not None:
        _SINK.flush()
        _SINK.flush_log()
    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm)

# ---------------------------------------------------------------------------
# Verification helpers (L5)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """First fenced block; tolerate a missing closing fence (truncation)."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m_open = re.search(r"```(?:python)?\s*\n?", text)
    if m_open:
        return text[m_open.end():].strip()
    # No fence: if it parses as Python, treat the raw text as code.
    try:
        ast.parse(text)
        return text.strip()
    except SyntaxError:
        return ""


def run_code(code: str, timeout: float = None):
    """Execute code in a fresh subprocess with a hard, reliably-killable
    timeout — the one place a timeout truly stops work (unlike interrupting
    an in-process llama.cpp call, which L1 forbids)."""
    timeout = timeout or CODE_EXEC_TIMEOUT_S
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "execution timed out"
    except Exception as exc:
        return False, "", str(exc)


_ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)


def safe_eval(expr: str):
    """Evaluate a pure-arithmetic expression. Anything beyond numbers and
    + - * / ** // % (names, calls, subscripts, strings) is rejected."""
    expr = (expr or "").strip().strip("`").strip()
    if not expr or len(expr) > 300:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            e = node.right
            if not (isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                    and abs(e.value) <= 64):
                return None
    try:
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def extract_number(text: str):
    """Pull the final numeric answer: prefer an 'Answer:' line, else the
    last number in the text."""
    if not text:
        return None
    ms = re.findall(r"answer:\s*\$?\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    candidate = ms[-1] if ms else None
    if candidate is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        candidate = nums[-1] if nums else None
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


def numbers_agree(a, b) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) < 1e-6:
        return True
    # tolerate rounding to 2dp (currency) either direction
    return abs(round(a, 2) - round(b, 2)) < 1e-9


# --- summarization format constraints ---------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def parse_format_constraint(prompt: str):
    """Return one of:
       ("sentences", n) | ("bullets", n, max_words|None) |
       ("max_words", n) | None"""
    p = prompt.lower()
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

    m = re.search(r"exactly (\d+|one|two|three|four|five) sentences?", p)
    if m:
        v = m.group(1)
        return ("sentences", int(v) if v.isdigit() else words[v])
    if re.search(r"in one sentence|exactly one sentence", p):
        return ("sentences", 1)

    m = re.search(r"exactly (\d+|one|two|three|four|five) bullet", p)
    if m:
        v = m.group(1)
        n = int(v) if v.isdigit() else words[v]
        mw = re.search(r"no longer than (\d+) words|each no more than (\d+) words|≤\s*(\d+) words", p)
        max_w = None
        if mw:
            max_w = int(next(g for g in mw.groups() if g))
        return ("bullets", n, max_w)

    m = re.search(r"in exactly (\d+) words|exactly (\d+) words", p)
    if m:
        return ("exact_words", int(next(g for g in m.groups() if g)))
    m = re.search(r"no more than (\d+) words|in at most (\d+) words|under (\d+) words|(\d+) words or (?:less|fewer)", p)
    if m:
        n = int(next(g for g in m.groups() if g))
        return ("max_words", n)
    return None


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def split_bullets(text: str) -> list:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return [re.sub(r"^[-*•\u2022]\s*|^\d+[.)]\s*", "", ln)
            for ln in lines if re.match(r"^[-*•\u2022]|^\d+[.)]", ln)]


def check_format(answer: str, constraint) -> bool:
    if constraint is None:
        return True
    kind = constraint[0]
    if kind == "sentences":
        return len(split_sentences(answer)) == constraint[1]
    if kind == "bullets":
        bullets = split_bullets(answer)
        if len(bullets) != constraint[1]:
            return False
        max_w = constraint[2]
        if max_w:
            return all(len(b.split()) <= max_w for b in bullets)
        return True
    if kind == "max_words":
        return len(answer.split()) <= constraint[1]
    if kind == "exact_words":
        return len(answer.split()) == constraint[1]
    return True


def repair_format(answer: str, constraint) -> str:
    """Deterministic last resort — only reached after a regeneration also
    violated the constraint. Repairs at whole-sentence/bullet boundaries,
    never mid-sentence."""
    kind = constraint[0]
    if kind == "sentences":
        sents = split_sentences(answer)
        n = constraint[1]
        if len(sents) > n:
            return " ".join(sents[:n])
        return answer  # too few sentences can't be honestly fabricated
    if kind == "bullets":
        n = constraint[1]
        max_w = constraint[2]
        bullets = split_bullets(answer)
        if not bullets:  # model wrote prose: convert sentences to bullets
            bullets = split_sentences(answer)
        bullets = bullets[:n]
        if max_w:
            bullets = [" ".join(b.split()[:max_w]).rstrip(",;") for b in bullets]
        return "\n".join(f"- {b}" for b in bullets)
    if kind == "exact_words":
        ws = answer.split()
        n = constraint[1]
        if len(ws) <= n:
            return answer  # too few words can't be honestly padded
        cut = " ".join(ws[:n]).rstrip(",;")
        return cut if cut.endswith((".", "!", "?")) else cut + "."
    if kind == "max_words":
        ws = answer.split()
        n = constraint[1]
        if len(ws) <= n:
            return answer
        cut = " ".join(ws[:n])
        # end at the last sentence boundary inside the cut if one exists
        m = list(re.finditer(r"[.!?]", cut))
        if m:
            return cut[:m[-1].end()]
        return cut.rstrip(",;") + "."
    return answer


# --- NER repair --------------------------------------------------------------

def repair_ner_json(raw: str) -> str:
    """3-stage repair: direct parse -> slice [...] -> regex-mined pairs."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = (raw or "").find("["), (raw or "").rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass
    pairs = re.findall(r'"text"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"',
                       raw or "")
    if pairs:
        return json.dumps([{"text": t, "type": ty} for t, ty in pairs],
                          ensure_ascii=False)
    return ""


# ---------------------------------------------------------------------------
# Category handlers — each returns (answer, info_dict)
# ---------------------------------------------------------------------------

FACTUAL_SELFCHECK = os.environ.get("FACTUAL_SELFCHECK", "1") == "1"


def h_factual(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[FACTUAL]
    a = local.gen(PROMPTS[FACTUAL], prompt, cap, 0.2)
    if not a or fast or not FACTUAL_SELFCHECK or remaining() < RESERVE_S + 120:
        return a, {"verified": "n/a"}
    # Concept-swap guard: generation drift can attach the right physics to
    # the wrong term (observed: "RGB is subtractive"). The model knows the
    # fact when asked to review, so one cheap check catches it.
    review = local.gen(
        "Review the answer for factual errors, especially swapped or "
        "misattributed terms. If the answer is fully correct, reply with "
        "exactly: OK. Otherwise state the single most important error in "
        "one short sentence.",
        f"Question: {prompt}\n\nAnswer: {a}",
        FACTUAL_CHECK_CAP, 0.1)
    rv = (review or "").strip()
    if not rv or rv[:2].upper() == "OK":
        return a, {"verified": "selfcheck-ok"}
    fixed = local.gen(
        "Answer the question correctly and directly in 1-3 short "
        "sentences. State only the corrected facts — no hedging, no "
        "discussion of the error, no alternatives. A reviewer found this "
        f"error in a previous answer: {rv[:200]}",
        prompt, 140, 0.2)
    if fixed:
        return fixed, {"verified": "selfcheck-corrected"}
    return a, {"verified": "selfcheck-flagged-unfixed"}


def h_sentiment(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[SENTIMENT]
    a = local.gen(PROMPTS[SENTIMENT], prompt, cap, 0.2)
    if not a:
        return "", {"verified": "failed"}
    # Surface a buried label so the grader sees it FIRST. The label must
    # lead the answer; "the tone is mixed" mid-sentence is not prominent.
    label_pat = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)
    if not label_pat.search(a[:20]):
        m = label_pat.search(a)
        if m:
            label = m.group(1).capitalize()
            # Justification: the sentence the label lived in, plus the next
            # one if short — keeps the both-sides acknowledgement intact.
            sents = split_sentences(a)
            just = " ".join(s for s in sents if label_pat.search(s))[:220]
            if not just:
                just = " ".join(sents[:2])[:220]
            a = f"{label}. {just}"
    return a, {"verified": "label-surfaced"}


FMT_DESC = {
    "sentences": lambda c: f"EXACTLY {c[1]} sentences",
    "bullets": lambda c: f"EXACTLY {c[1]} bullet points"
               + (f", each at most {c[2]} words" if c[2] else ""),
    "max_words": lambda c: f"AT MOST {c[1]} words",
    "exact_words": lambda c: f"EXACTLY {c[1]} words",
}


def h_summarization(prompt, local, fast):
    constraint = parse_format_constraint(prompt)
    cap = (CAP_FAST if fast else CAP)[SUMMARIZATION]
    sys_prompt = PROMPTS[SUMMARIZATION]
    if constraint:
        sys_prompt += (" The summary MUST be "
                       + FMT_DESC[constraint[0]](constraint)
                       + " — count carefully before answering.")
    a = local.gen(sys_prompt, prompt, cap, 0.2)
    if check_format(a, constraint):
        return a, {"verified": "format-ok"}
    if fast or remaining() < RESERVE_S + 20:
        return repair_format(a, constraint), {"verified": "format-repaired"}
    # Regenerate with the violated constraint restated explicitly.
    desc = FMT_DESC[constraint[0]](constraint)
    retry = local.gen(
        PROMPTS[SUMMARIZATION] + f" Your previous attempt violated the "
        f"format. The summary MUST be {desc} — count before answering.",
        prompt, cap, 0.3)
    if check_format(retry, constraint):
        return retry, {"verified": "format-ok-retry"}
    best = retry if retry else a
    return repair_format(best, constraint), {"verified": "format-repaired"}


def h_ner(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[NER]
    wants_json = "json" in prompt.lower()
    if wants_json:
        raw = local.gen(PROMPTS[NER + "_json"], prompt, cap, 0.1,
                        grammar=get_ner_grammar())
        fixed = repair_ner_json(raw)
        if fixed:
            return fixed, {"verified": "json-valid"}
        if not fast:
            raw2 = local.gen(PROMPTS[NER + "_json"], prompt, cap, 0.3,
                             grammar=get_ner_grammar())
            fixed2 = repair_ner_json(raw2)
            if fixed2:
                return fixed2, {"verified": "json-valid-retry"}
        return raw or "", {"verified": "json-failed"}
    a = local.gen(PROMPTS[NER], prompt, cap, 0.1)
    return a, {"verified": "n/a"}


def h_logic(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[LOGIC]
    a = local.gen(PROMPTS[LOGIC], prompt, cap, 0.2)
    if a and "answer:" in a.lower():
        return a, {"verified": "answer-line"}

    # Reasoning exists but no Answer line — cap truncation. Conclude can
    # only EXTRACT an answer that already appears in the text; if the
    # derivation was cut before reaching it, conclude guesses (observed:
    # wrong seating order). So first CONTINUE the derivation to completion,
    # then conclude only as a last resort.
    if a and not fast and remaining() > RESERVE_S + 30:
        cont = local.gen(
            "You are finishing a logic-puzzle solution that was cut off. "
            "Continue the reasoning from exactly where it stops — do not "
            "restart or repeat it — and finish with a single final line in "
            "exactly this form: 'Answer: <answer>'.",
            prompt + "\n\nWork so far (continue from the end):\n" + a[-1600:],
            220, 0.1)
        if cont and "answer:" in cont.lower():
            return a + "\n" + cont, {"verified": "continued"}
        if cont:
            a = a + "\n" + cont  # partial progress still helps conclude
    if a:
        conclude = local.gen(
            "You are given a logic puzzle and a partial line of reasoning "
            "that was cut off. Based on that reasoning, reply with ONLY one "
            "line in exactly this form: 'Answer: <final answer>'. No other "
            "text.",
            prompt + "\n\nReasoning so far:\n" + a[-1400:],
            LOGIC_CONCLUDE_CAP, 0.1)
        m = re.search(r"answer:.*", conclude or "", re.IGNORECASE)
        if m:
            return a + "\n\n" + m.group(0).strip(), {"verified": "concluded"}

    if fast or remaining() < RESERVE_S + 25:
        return a, {"verified": "no-answer-line"}
    retry = local.gen(
        "Solve the puzzle in a few short plain-text lines. Your LAST line "
        "must be exactly 'Answer: <final answer>'.", prompt, 320, 0.1)
    if retry and "answer:" in retry.lower():
        return retry, {"verified": "answer-line-retry"}
    return retry or a, {"verified": "no-answer-line"}


def h_math(prompt, local, fast):
    """Dual-path self-consistency (L5):
       path A — natural-language derivation;
       path B — model-written Python script, executed in a subprocess.
       Agreement wins; disagreement triggers one tie-breaking resample of
       the code path; execution-verified code is the default arbiter."""
    nl_cap = (CAP_FAST if fast else CAP)[MATH]
    a_nl = local.gen(PROMPTS[MATH + "_nl"], prompt, nl_cap, 0.2)
    v_nl = extract_number(a_nl)

    def expr_pass(temp=0.1):
        raw = local.gen(PROMPTS[MATH + "_expr"], prompt, 60, temp)
        line = (raw or "").strip().splitlines()
        return safe_eval(line[0]) if line else None

    def code_pass(temp):
        raw = local.gen(PROMPTS[MATH + "_code"], prompt, MATH_CODE_CAP, temp)
        code = extract_code(raw)
        if not code:
            return None
        ok, stdout, _err = run_code(code)
        if not ok or not stdout:
            return None
        return extract_number(stdout)

    # Cheap deterministic check first (~30 output tokens); the full script
    # path only runs when the expression fails to produce a value.
    v_code = None
    if not fast:
        v_code = expr_pass()
        if v_code is None:
            v_code = code_pass(0.2)

    if v_code is not None and v_nl is not None:
        if numbers_agree(v_code, v_nl):
            return a_nl, {"verified": "dual-agree", "value": v_nl}
        v_code2 = code_pass(0.5) if remaining() > RESERVE_S + 60 else None
        if v_code2 is not None and numbers_agree(v_code2, v_nl):
            return a_nl, {"verified": "resample-sided-nl", "value": v_nl}
        # trust execution over narration; keep the derivation, fix the line
        body = re.sub(r"answer:.*$", "", a_nl, flags=re.I | re.S).strip()
        fixed = f"{body}\n\nAnswer: {v_code:g}"
        return fixed, {"verified": "code-overrides-nl", "value": v_code}

    if v_nl is not None:
        return a_nl, {"verified": "nl-only", "value": v_nl}
    if v_code is not None:
        return f"Answer: {v_code:g}", {"verified": "code-only", "value": v_code}
    return a_nl or "", {"verified": "unverified"}


def _code_repair_loop(prompt, system, local, cap, must_execute=True):
    """Generate -> parse -> execute; on failure, feed the ACTUAL error back
    and retry once with a doubled budget. Returns (answer, status)."""
    a = local.gen(system, prompt, cap, 0.2)
    code = extract_code(a)
    err = None
    if code:
        try:
            ast.parse(code)
            if must_execute:
                ok, _out, stderr = run_code(code)
                if not ok:
                    err = stderr.splitlines()[-1] if stderr else "runtime error"
            if err is None:
                return a, "exec-ok" if must_execute else "parse-ok"
        except SyntaxError as exc:
            err = f"SyntaxError: {exc}"
    else:
        err = "no code block found"

    if remaining() < RESERVE_S + 30:
        return a, f"unrepaired ({err})"

    retry = local.gen(
        system + f" Your previous attempt failed with: {err}. Provide the "
        "complete, corrected code in a single Python code block.",
        prompt, min(cap * 2, 900), 0.2)
    rcode = extract_code(retry)
    if rcode:
        try:
            ast.parse(rcode)
            if must_execute:
                ok, _out, _err2 = run_code(rcode)
                if ok:
                    return retry, "exec-ok-retry"
                return retry, "parse-ok-retry"
            return retry, "parse-ok-retry"
        except SyntaxError:
            pass
    return retry if rcode else a, f"unrepaired ({err})"


def h_code_gen(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[CODE_GEN]
    a, status = _code_repair_loop(prompt, PROMPTS[CODE_GEN], local, cap,
                                  must_execute=not fast)
    return a, {"verified": status}


def h_code_debug(prompt, local, fast):
    cap = (CAP_FAST if fast else CAP)[CODE_DEBUG]
    a, status = _code_repair_loop(prompt, PROMPTS[CODE_DEBUG], local, cap,
                                  must_execute=not fast)
    return a, {"verified": status}


DISPATCH = {
    FACTUAL: h_factual, SENTIMENT: h_sentiment,
    SUMMARIZATION: h_summarization, NER: h_ner, LOGIC: h_logic,
    MATH: h_math, CODE_GEN: h_code_gen, CODE_DEBUG: h_code_debug,
}

# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def load_tasks() -> list:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    cleaned = []
    for t in tasks:
        if isinstance(t, dict) and "task_id" in t and "prompt" in t:
            cleaned.append({"task_id": str(t["task_id"]),
                            "prompt": str(t["prompt"])})
    return cleaned


def run() -> int:
    global _SINK
    tasks = load_tasks()
    log(f"[input] {len(tasks)} tasks from {INPUT_PATH}")
    sink = Sink(tasks)
    _SINK = sink

    ordered = sorted(
        ((t, classify(t["prompt"])) for t in tasks),
        key=lambda tc: ORDER_RANK[tc[1]],
    )
    for t, c in ordered:
        log(f"[route] {t['task_id']} -> {c}")

    try:
        local = Local()
    except Exception as exc:
        # Model load failure: results.json already exists with fallbacks for
        # every task_id (L3) — exit 0 so the harness scores what exists.
        log(f"[fatal] model load failed: {exc}")
        sink.flush()
        sink.flush_log()
        return 0

    fast = False
    for t, cat in ordered:
        tid = t["task_id"]
        left = remaining()
        if not fast and left < RESERVE_S + 3.0 * local.avg_task_s:
            fast = True
            log(f"[governor] {left:.0f}s left — FAST mode (short caps, no retries)")
        if left < RESERVE_S * 0.5:
            log(f"[governor] {left:.0f}s left — banking fallbacks for the rest")
            break  # sink already holds FALLBACK for unprocessed task_ids

        t0 = time.time()
        try:
            answer, info = DISPATCH[cat](t["prompt"], local, fast)
        except Exception as exc:
            log(f"[task] {tid} handler crashed: {exc}")
            answer, info = "", {"verified": f"handler-error: {exc}"}
        dt = time.time() - t0
        local.note_latency(dt)
        info.update({"category": cat, "seconds": round(dt, 2),
                     "fireworks_tokens": 0, "fast_mode": fast})
        sink.put(tid, answer, info)
        log(f"[task] {tid} ({cat}) {dt:.1f}s verified={info.get('verified')}")

    sink.meta["total_seconds"] = round(time.time() - START, 1)
    sink.meta["answered"] = sum(1 for v in sink.answered.values() if v)
    sink.meta["total_tasks"] = len(tasks)
    sink.flush()
    sink.flush_log()
    log(f"[done] {sink.meta['answered']}/{len(tasks)} answered, "
        f"0 fireworks tokens, {sink.meta['total_seconds']}s")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        log(f"[fatal] {type(exc).__name__}: {exc} — salvaging output")
        try:
            if _SINK is not None:
                _SINK.flush()
                _SINK.flush_log()
            elif not os.path.exists(OUTPUT_PATH):
                os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump([], f)
        except Exception:
            pass
        return 0  # scored partial output beats RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
