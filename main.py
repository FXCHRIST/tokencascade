#!/usr/bin/env python3
"""TokenCascade v5 — AMD ACT II Track 1. Clean-room rebuild.

Post-mortem of v4's RUNTIME_ERROR drove five design laws:

  L1. NO native-code interruption. Local generations run to completion
      (bounded by small max_tokens), never streamed-and-abandoned.
      Interrupting llama.cpp mid-sample is the prime segfault suspect.
  L2. NO threads. Strictly sequential. Nothing shares state; teardown is
      trivial; every line is debuggable from a log.
  L3. Results are flushed to disk ATOMICALLY AFTER EVERY TASK, and SIGTERM
      converts to flush-and-exit-0. A crash at task 12 leaves 12 answers.
  L4. Exact pins. The dependency set is the one that demonstrably ran.
  L5. Free compute does verification: local math answers are checked by
      ACTUAL ARITHMETIC (model proposes the expression, Python computes),
      local code is compile-gated. Zero tokens, real accuracy.

Routing (env-tunable, no rebuild):
  LOCAL_CATEGORIES default: factual,sentiment,ner,summarization,math,code_debug
  Zero-token mode: add logic,code_gen to LOCAL_CATEGORIES.
"""
"""TokenCascade v6 — AMD ACT II Track 1. Flawless revision."""

import ast
import atexit
import json
import os
import re
import signal
import sys
import time

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
LOG_PATH = os.environ.get("LOG_PATH", "/output/inference_log.json")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")
LOCAL_CATEGORIES = {
    c.strip()
    for c in os.environ.get(
        "LOCAL_CATEGORIES",
        "factual,sentiment,ner,summarization,math,code_debug",
    ).split(",")
    if c.strip()
}
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "520"))
RESERVE_S = float(os.environ.get("RESERVE_S", "60"))
PROBE_TIMEOUT_S = float(os.environ.get("PROBE_TIMEOUT_S", "8"))
LOCAL_MAX_PROMPT_CHARS = int(os.environ.get("LOCAL_MAX_PROMPT_CHARS", "1600"))
START = time.time()

SYS_REMOTE = "Answer only what is asked. No preamble, no markdown."
SYS_LOCAL = (
    "You are a concise assistant. Do NOT use a <think> block. Do NOT include internal reasoning. "
    "Answer concisely and directly without any introductory text, preambles, or markdown formatting. "
    "State the final answer immediately."
)

# Speed-optimized caps to prevent dual-core CPU generation timeouts
# Expanded caps to accommodate DeepSeek-R1's <think> phase without truncation
CAP_LOCAL = {
    "factual": 140,       
    "sentiment": 140,     
    "summarization": 160, 
    "ner": 160,           
    "math": 320,          # High cap to allow deep mathematical reasoning
    "code_debug": 300,   
    "logic": 280,         # High cap to allow step-by-step logic chains
    "code_gen": 380       
}
CAP_REMOTE = {"math": 300, "logic": 170, "code_debug": 400, "code_gen": 380,
              "factual": 150, "sentiment": 80, "ner": 180, "summarization": 120}

def remaining() -> float:
    return TIME_BUDGET_S - (time.time() - START)

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

class Sink:
    def __init__(self, task_ids):
        self.results = {tid: "" for tid in task_ids}
        self.meta = {"fireworks_tokens": 0, "routes": {}, "notes": []}
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

    def set(self, tid: str, answer: str, route: str) -> None:
        self.results[tid] = answer
        self.meta["routes"][tid] = route
        self.flush()

    def flush(self) -> None:
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            payload = [{"task_id": t, "answer": a} for t, a in self.results.items()]
            tmp = OUTPUT_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, OUTPUT_PATH)
            self.meta["elapsed_s"] = round(time.time() - START, 1)
            self.meta["empty_answers"] = [t for t, a in self.results.items() if not a]
            with open(LOG_PATH, "w") as f:
                json.dump(self.meta, f, indent=2)
        except Exception as e:
            log(f"[sink] flush failed: {e}")

CATEGORY_RULES = [
    ("sentiment", r"\bsentiment\b|\bfeel(ing)?\b|\bemotion(s)?\b|\btone\b"),
    ("ner", r"named entit|entities and their types|extract .{0,40}entit|\bidentify the people\b"),
    ("summarization", r"\bsummar(y|ise|ize|iz)|\bTL;DR\b|\bshorten\b|\bmain point\b"),
    ("code_debug", r"(bug|fix|broken|incorrect|error|exception).{0,120}(def |function|code|```)"
                   r"|(def |function|```).{0,160}(bug|fix|broken)"),
    ("code_gen", r"\bwrite (a |an )?\w{0,12}\s?(function|program|script|class|method)\b|\bcreate a (python|script)\b"),
    ("logic", r"each own|who owns|exactly one|three friends|puzzle|deduce"
              r"|all (the )?conditions|must be satisfied"
              r"|taller than|shorter than|older than|younger than"
              r"|who is the (shortest|tallest|oldest|youngest)"),
    ("math", r"\bhow (many|much)\b.*\d|\d+\s*%|\bpercent|\bcalculate\b"
             r"|\baverage\b.*\d|\bremain\b.*\d|\d.*\bremain\b|\bcost\b|\bprice\b"),
]

def strip_think(text: str) -> str:
    """Removes DeepSeek thinking blocks to isolate the final answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def classify(prompt: str) -> str:
    p = prompt.lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, p, re.DOTALL):
            return cat
    if re.search(r"\d", p) and re.search(r"total|left|per hour|speed|cost|price|discount", p):
        return "math"
    return "factual"

_ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                ast.Mod, ast.Pow, ast.USub, ast.UAdd)

def safe_eval(expr: str):
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST):
                raise ValueError(f"disallowed node: {type(node).__name__}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric constant")
        return eval(compile(tree, "<expr>", "eval"))
    except Exception:
        return None

def fmt_num(v) -> str:
    if v is None: return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, float):
        return f"{round(v, 4):g}"
    return str(v)

def extract_code_blocks(text: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not blocks and "def " in text:
        blocks = [text[text.index("def "):]]
    return blocks

def code_compiles(text: str) -> bool:
    blocks = extract_code_blocks(text)
    if not blocks:
        return False
    try:
        for b in blocks:
            compile(b.strip(), "<candidate>", "exec")
        return True
    except SyntaxError:
        return False

class Local:
    def __init__(self):
        from llama_cpp import Llama
        t0 = time.time()
        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,           
            n_threads=2,          # Changed from 3 to 2 to eliminate CPU contention
            n_batch=512,          
            verbose=False,
        )
        self.avg_task_s = 10.0
        log(f"[local] model loaded in {time.time()-t0:.1f}s")
      
    def gen(self, system: str, user: str, cap: int) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=cap,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def answer(self, prompt: str, cat: str) -> str:
        injected_prompt = prompt
        
        if cat == "sentiment":
            injected_prompt += "\n\nCRITICAL: After your internal reasoning, your final response MUST explicitly name the specific positive AND negative details mentioned."
        elif cat == "summarization":
            injected_prompt += "\n\nCRITICAL: After your internal reasoning, you must strictly obey the sentence length constraints."
        elif cat == "ner":
            injected_prompt += "\n\nCRITICAL: After your internal reasoning, extract EVERY entity with the exact required labels."
        elif cat == "math":
            injected_prompt += "\n\nCRITICAL: After your internal reasoning, make sure to explicitly provide the final answer values."
        elif cat == "factual":
            injected_prompt += "\n\nCRITICAL: State the exact names or requested entities directly in your final conclusion."
        elif cat == "logic":
            injected_prompt += "\n\nCRITICAL: State the exact names of the subjects requested in your final conclusion."

        return self.gen(SYS_LOCAL, injected_prompt, CAP_LOCAL.get(cat, 300))

    def math_verified(self, prompt: str) -> str:
        answer = self.answer(prompt, "math")
        if remaining() < RESERVE_S + 2 * self.avg_task_s:
            return answer
        try:
            expr_raw = self.gen(
                "Output ONLY the arithmetic expression(s) that compute the final numeric answer(s), separated by ';'. "
                "Python syntax. Numbers and + - * / % ( ) only. No words.",
                f"Problem:\n{prompt}", 120, # Increased cap slightly to allow safe thinking room
            )
            # STRIP THINK TAGS FIRST!
            expr_clean = strip_think(expr_raw)
            exprs = [e for e in (s.strip() for s in expr_clean.split(";")) if e]
            values = [safe_eval(e) for e in exprs[:4] if safe_eval(e) is not None]
            if not values:
                return answer
        except Exception as e:
            log(f"[math] extraction/eval skipped: {e}")
            return answer
            
        wanted = [fmt_num(v) for v in values if v is not None]
        norm = answer.replace(",", "")
        if wanted and all(w in norm for w in wanted):
            return answer
            
        log(f"[math] mismatch — regenerating with hint")
        if wanted and remaining() > RESERVE_S + self.avg_task_s:
            hinted = self.gen(
                SYS_LOCAL,
                f"{prompt}\n\n(The correct computed value(s): {', '.join(wanted)}. State them plainly in your answer.)",
                CAP_LOCAL["math"],
            )
            if all(w in hinted.replace(",", "") for w in wanted):
                return hinted
        return ("The answer is " + " and ".join(wanted) + ".") if wanted else answer

    def code_debug_gated(self, prompt: str):
        answer = self.answer(prompt, "code_debug")
        blocks = extract_code_blocks(answer)
        
        # FIXED: Extract and return ONLY the raw python block if it compiles
        if blocks and code_compiles(answer):
            return blocks[0].strip(), True
            
        if remaining() > RESERVE_S + self.avg_task_s:
            retry = self.gen(
                SYS_LOCAL,
                prompt + "\n\n(Provide the corrected function as a complete, valid Python code block.)",
                CAP_LOCAL["code_debug"],
            )
            r_blocks = extract_code_blocks(retry)
            if r_blocks and code_compiles(retry):
                return r_blocks[0].strip(), True
            if retry:
                answer = retry
                
        # Fallback: Extract the first block even if compilation checks are risky
        blocks = extract_code_blocks(answer)
        if blocks:
            return blocks[0].strip(), False
        return answer, False

class Remote:
    def __init__(self):
        from openai import OpenAI

        base = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
        key = os.environ["FIREWORKS_API_KEY"]
        self.allowed = [m.strip() for m in
                        os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
        if not self.allowed:
            raise RuntimeError("ALLOWED_MODELS is empty")
        
        variants = [base]
        if not base.endswith("/v1") and not base.endswith("/inference/v1"):
            variants.append(base + "/v1")
            variants.append(base + "/inference/v1")
        elif base.endswith("/v1") and not base.endswith("/inference/v1"):

            variants.append(base[:-3])
            
        self.bases = list(dict.fromkeys(variants))
        
        self._probe = {b: OpenAI(base_url=b, api_key=key,
                                 timeout=PROBE_TIMEOUT_S, max_retries=0)
                       for b in self.bases}
        self._live = {b: OpenAI(base_url=b, api_key=key, timeout=25, max_retries=1)
                      for b in self.bases}
        self.tokens = 0
        self.locked = None
        self.alive = True

    def _candidates(self, cat: str):
        prefs = ["kimi"] if cat in ("code_debug", "code_gen") else ["minimax"]
        ordered = []
        for kw in prefs:
            ordered += [m for m in self.allowed if kw in m.lower() and "gemma" not in m.lower()]
        ordered += [m for m in self.allowed if "gemma" not in m.lower() and m not in ordered]
        ordered += [m for m in self.allowed if m not in ordered]
        
        out = []
        for m in ordered:
            bare = m.split("/")[-1]
            pref = m if m.startswith("accounts/") else f"accounts/fireworks/models/{bare}"
            for v in (m, pref, bare):
                if v not in out:
                    out.append(v)
        return out

    def _styled(self, cat: str, style: str) -> str:
        for m in self._candidates(cat):
            if ("/" in m) == (style == "prefixed"):
                return m
        return self._candidates(cat)[0]

    def _chat(self, client, model: str, prompt: str, cat: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0,
            max_tokens=CAP_REMOTE.get(cat, 200),
            messages=[{"role": "system", "content": SYS_REMOTE},
                      {"role": "user", "content": prompt}],
        )
        if r.usage:
            self.tokens += (r.usage.prompt_tokens or 0) + (r.usage.completion_tokens or 0)
        text = (r.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty completion")
        return text

    def answer(self, prompt: str, cat: str) -> str:
        if self.locked:
            base, style = self.locked
            try:
                return self._chat(self._live[base], self._styled(cat, style), prompt, cat)
            except Exception as e:
                log(f"[remote] locked path failed ({e}); re-probing")
                self.locked = None
        last = None
        for base in self.bases:
            for model in self._candidates(cat):
                if remaining() < 15:
                    raise RuntimeError("time exhausted while probing")
                try:
                    text = self._chat(self._probe[base], model, prompt, cat)
                    self.locked = (base, "prefixed" if "/" in model else "bare")
                    log(f"[remote] LOCKED base={base} model={model}")
                    return text
                except Exception as e:
                    last = e
        self.alive = False
        raise RuntimeError(f"all remote combos failed: {last}")

def run() -> int:
    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    for i, t in enumerate(tasks):
        t["task_id"] = str(t.get("task_id", f"task-{i+1}"))
    sink = Sink([t["task_id"] for t in tasks])

    remote = None
    try:
        remote = Remote()
    except Exception as e:
        log(f"[remote] unavailable: {e}")
    local = None
    try:
        local = Local()
    except Exception as e:
        log(f"[local] unavailable: {e}")

    cats = {t["task_id"]: classify(t.get("prompt", "")) for t in tasks}

    def do_remote(t) -> bool:
        if not remote or not remote.alive: return False
        try:
            sink.set(t["task_id"], remote.answer(t["prompt"], cats[t["task_id"]]), f"remote:{cats[t['task_id']]}")
            return True
        except Exception as e:
            log(f"[remote] {t['task_id']}: {e}")
            return False

    def do_local(t) -> bool:
        if not local: return False
        tid, cat = t["task_id"], cats[t["task_id"]]
        try:
            if cat == "math":
                text = local.math_verified(t["prompt"])
            elif cat == "code_debug":
                text, gate_ok = local.code_debug_gated(t["prompt"])
                if not gate_ok and remote is not None and remote.alive:
                    return False
            elif cat == "code_gen":
                raw_text = local.answer(t["prompt"], cat)
                blocks = extract_code_blocks(raw_text)
                text = blocks[0].strip() if blocks else raw_text
            else:
                # FIX: Intercept standard categories and strip the think blocks!
                raw_text = local.answer(t["prompt"], cat)
                text = strip_think(raw_text)
                
            if text:
                sink.set(tid, text, f"local:{cat}")
                return True
        except Exception as e:
            log(f"[local] {tid}: {e}")
        return False

    remote_first = [t for t in tasks
                    if cats[t["task_id"]] not in LOCAL_CATEGORIES
                    or len(t.get("prompt", "")) > LOCAL_MAX_PROMPT_CHARS
                    or local is None]
    local_first = [t for t in tasks if t not in remote_first]

    for t in remote_first:
        if not do_remote(t): do_local(t)

    for t in local_first:
        projected = (local.avg_task_s if local else 10.0)
        if remaining() - RESERVE_S < projected:
            log("[governor] time low — routing remaining tasks remote")
            if not do_remote(t): do_local(t)
            continue
        if not do_local(t): do_remote(t)

    for t in tasks:
        if sink.results[t["task_id"]]: continue
        if remaining() > 25 and do_remote(t): continue
        if remaining() > 12: do_local(t)

    sink.meta["fireworks_tokens"] = remote.tokens if remote else 0
    sink.flush()
    log(f"[done] {len(tasks)} tasks, {sink.meta['fireworks_tokens']} fireworks tokens, {time.time()-START:.1f}s")
    return 0

def main() -> int:
    try:
        return run()
    except Exception as e:
        log(f"[fatal] {type(e).__name__}: {e} — attempting salvage output")
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
