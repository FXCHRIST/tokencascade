#!/usr/bin/env python3
"""TokenCascade v3 (hardened) — AMD ACT II Track 1.

Post-mortem fixes vs v2 (ACCURACY_GATE_FAILED at 26.3% = 5/19):
  1. Local generation now STREAMS with a cooperative deadline: at timeout it
     returns the partial answer instead of nothing, and local NEVER disables
     itself (v2's two-timeout kill-switch could silently zero the whole run
     on a slow grading CPU).
  2. Remote calls probe a fallback matrix of base-URL and model-ID variants
     (proxy may want '/v1' or not; model IDs may be bare or fully-prefixed).
     First working combo is cached; failed probes cost no tokens.
  3. Final sweep: any still-empty answer gets one more attempt with whatever
     path is alive. An answer, even partial, beats an empty string.
"""

import json
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor

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
LOCAL_TASK_TIMEOUT_S = float(os.environ.get("LOCAL_TASK_TIMEOUT_S", "26"))
LOCAL_MAX_PROMPT_CHARS = int(os.environ.get("LOCAL_MAX_PROMPT_CHARS", "1600"))
START = time.time()

SYS = "Answer directly and concisely."
LOCAL_SYS = "Answer directly and concisely. Answer every part of the question."

MAX_TOK = {
    "math": 300, "logic": 220, "code_debug": 400, "code_gen": 450,
    "factual": 150, "sentiment": 80, "ner": 180, "summarization": 120,
}


def remaining() -> float:
    return TIME_BUDGET_S - (time.time() - START)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------- classifier
CATEGORY_RULES = [
    ("sentiment", r"\bsentiment\b"),
    ("ner", r"named entit|entities and their types|extract .{0,40}entit"),
    ("summarization", r"\bsummar(y|ise|ize|iz)"),
    ("code_debug", r"(bug|fix|broken|incorrect).{0,120}(def |function|code|```)"
                   r"|(def |function|```).{0,160}(bug|fix|broken)"),
    ("code_gen", r"\bwrite (a |an )?\w{0,12}\s?(function|program|script|class|method)\b"),
    ("logic", r"each own|who owns|exactly one|three friends|puzzle|deduce"
              r"|all (the )?conditions|must be satisfied"
              r"|taller than|shorter than|older than|younger than"
              r"|who is the (shortest|tallest|oldest|youngest)"),
    ("math", r"\bhow (many|much)\b.*\d|\d+\s*%|\bpercent|\bcalculate\b"
             r"|\baverage\b.*\d|\bremain\b.*\d|\d.*\bremain\b"),
]


def classify(prompt: str) -> str:
    p = prompt.lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, p, re.DOTALL):
            return cat
    if re.search(r"\d", p) and re.search(r"total|left|per hour|speed|cost|price", p):
        return "math"
    return "factual"


# ----------------------------------------------------------- remote engine
class Remote:
    """Fireworks via the harness proxy, with a self-locating fallback matrix."""

    def __init__(self):
        from openai import OpenAI

        base = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
        key = os.environ["FIREWORKS_API_KEY"]
        self.allowed = [
            m.strip()
            for m in os.environ.get("ALLOWED_MODELS", "").split(",")
            if m.strip()
        ]
        if not self.allowed:
            raise RuntimeError("ALLOWED_MODELS is empty")

        # Base-URL variants: verbatim first, then with/without common suffixes.
        variants = [base]
        if base.endswith("/v1"):
            variants.append(base[: -len("/v1")])
        else:
            variants.append(base + "/v1")
        if not base.endswith("/inference/v1"):
            variants.append(base + "/inference/v1")
        self.base_variants = []
        for b in variants:  # dedupe, keep order
            if b not in self.base_variants:
                self.base_variants.append(b)

        self._mk = lambda b: OpenAI(base_url=b, api_key=key, timeout=25, max_retries=0)
        self.clients = {b: self._mk(b) for b in self.base_variants}
        self.tokens_used = 0
        self.locked = None  # (base, transform_fn_name) once a combo works
        self.alive = True

    def _model_variants(self, cat: str):
        prefs = ["kimi"] if cat in ("code_debug", "code_gen") else ["minimax"]
        ordered = []
        for kw in prefs:
            ordered += [m for m in self.allowed if kw in m.lower() and "gemma" not in m.lower()]
        ordered += [m for m in self.allowed if "gemma" not in m.lower() and m not in ordered]
        ordered += [m for m in self.allowed if m not in ordered]
        out = []
        for m in ordered:
            bare = m.split("/")[-1]
            prefixed = m if m.startswith("accounts/") else f"accounts/fireworks/models/{bare}"
            for v in (m, prefixed, bare):
                if v not in out:
                    out.append(v)
        return out

    def _chat(self, base: str, model: str, prompt: str, cat: str) -> str:
        resp = self.clients[base].chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=MAX_TOK.get(cat, 200),
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
            ],
        )
        u = resp.usage
        if u:
            self.tokens_used += (u.prompt_tokens or 0) + (u.completion_tokens or 0)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty completion")
        return text

    def answer(self, prompt: str, cat: str) -> str:
        # Fast path: a known-working combo.
        if self.locked:
            base, model_for = self.locked
            try:
                return self._chat(base, model_for(cat), prompt, cat)
            except Exception as e:
                log(f"[remote] locked combo failed ({e}); re-probing")
                self.locked = None
        # Probe matrix. Failed probes (404 etc.) bill zero tokens.
        last = None
        for base in self.base_variants:
            for model in self._model_variants(cat):
                if remaining() < 15:
                    raise RuntimeError("time exhausted during probing")
                try:
                    text = self._chat(base, model, prompt, cat)
                    picked = model
                    log(f"[remote] LOCKED base={base} model={picked}")
                    kimi_like = [m for m in self._model_variants("code_gen")
                                 if "kimi" in m.lower()]
                    mm_like = [m for m in self._model_variants("factual")
                               if "minimax" in m.lower()]

                    def model_for(c, _p=picked, _k=kimi_like, _m=mm_like):
                        cands = _k if c in ("code_debug", "code_gen") else _m
                        for cand in cands:
                            # keep same ID style as the working pick
                            if ("/" in cand) == ("/" in _p):
                                return cand
                        return _p

                    self.locked = (base, model_for)
                    return text
                except Exception as e:
                    last = e
                    continue
        self.alive = False
        raise RuntimeError(f"all remote combos failed: {last}")


# ------------------------------------------------------------ local engine
class Local:
    """Streaming generation with a cooperative deadline: timeout returns the
    PARTIAL answer, and local never disables itself."""

    def __init__(self):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=max(2, os.cpu_count() or 2),
            verbose=False,
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        t0 = time.time()
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say OK."}],
            max_tokens=8, temperature=0,
        )
        dt = max(time.time() - t0, 0.1)
        toks = out.get("usage", {}).get("completion_tokens", 4) or 4
        self.tps = max(toks / dt, 1.0)
        log(f"[local] ready, ~{self.tps:.1f} tok/s (probe)")

    def _stream(self, prompt: str, cap: int, deadline: float) -> str:
        parts = []
        stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": LOCAL_SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0, max_tokens=cap, stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                parts.append(piece)
            if time.monotonic() > deadline:
                log("[local] deadline hit — returning partial answer")
                break
        return "".join(parts).strip()

    def answer(self, prompt: str, cat: str) -> str:
        cap = min(MAX_TOK.get(cat, 150), max(48, int(self.tps * LOCAL_TASK_TIMEOUT_S * 0.8)))
        deadline = time.monotonic() + LOCAL_TASK_TIMEOUT_S
        fut = self.pool.submit(self._stream, prompt, cap, deadline)
        # Hard backstop 50% past the cooperative deadline (prompt processing
        # happens before the first streamed token and can't be interrupted).
        return fut.result(timeout=LOCAL_TASK_TIMEOUT_S * 1.5 + 5)


# -------------------------------------------------------------------- main
def main() -> int:
    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    results = {str(t["task_id"]): "" for t in tasks}
    routes = {}

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

    cats = {str(t["task_id"]): classify(t.get("prompt", "")) for t in tasks}

    def try_remote(t) -> bool:
        tid = str(t["task_id"])
        if not remote or not remote.alive:
            return False
        try:
            results[tid] = remote.answer(t["prompt"], cats[tid])
            routes[tid] = f"remote:{cats[tid]}"
            return True
        except Exception as e:
            log(f"[remote] {tid}: {e}")
            return False

    def try_local(t, timeout_scale: float = 1.0) -> bool:
        tid = str(t["task_id"])
        if not local:
            return False
        try:
            text = local.answer(t["prompt"], cats[tid])
            if text:
                results[tid] = text
                routes[tid] = f"local:{cats[tid]}"
                return True
        except Exception as e:
            log(f"[local] {tid}: {e}")
        return False

    is_local_first = lambda t: (
        cats[str(t["task_id"])] in LOCAL_CATEGORIES
        and len(t.get("prompt", "")) <= LOCAL_MAX_PROMPT_CHARS
    )
    local_q = [t for t in tasks if is_local_first(t)]
    remote_q = [t for t in tasks if not is_local_first(t)]

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(try_remote, t) for t in remote_q]

    for t in local_q:
        if remaining() < 45:
            log("[watchdog] flushing remaining local-first tasks to remote")
            futures.append(pool.submit(try_remote, t))
            continue
        if not try_local(t):
            futures.append(pool.submit(try_remote, t))

    for f in futures:
        try:
            f.result(timeout=max(remaining() - 10, 5))
        except Exception:
            pass

    # Final sweep: no task ships empty while any path is alive.
    for t in tasks:
        tid = str(t["task_id"])
        if results[tid]:
            continue
        if remaining() > 25 and try_remote(t):
            continue
        if remaining() > 15:
            try_local(t)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            [{"task_id": tid, "answer": ans} for tid, ans in results.items()],
            f, ensure_ascii=False,
        )
    try:
        with open(LOG_PATH, "w") as f:
            json.dump({
                "fireworks_tokens": remote.tokens_used if remote else 0,
                "routes": routes,
                "empty_answers": [k for k, v in results.items() if not v],
                "elapsed_s": round(time.time() - START, 1),
            }, f, indent=2)
    except Exception:
        pass
    log(f"[done] {len(tasks)} tasks, "
        f"{remote.tokens_used if remote else 0} fireworks tokens, "
        f"{time.time()-START:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
