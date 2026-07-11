#!/usr/bin/env python3
"""TokenCascade v4 — AMD ACT II Track 1. Final hardened build.

Design invariants:
  I1. results.json is ALWAYS written (valid schema, every task_id) and the
      process exits 0, even on internal crashes.
  I2. Local inference can be slow but can never zero the run: streaming with
      a cooperative deadline returns partial answers; a hard-timeout puts
      local into slow-mode (remote-first) instead of piling up.
  I3. Remote self-locates: base-URL x model-ID fallback matrix with SHORT
      probe timeouts; failed probes cost no tokens; first success locks.
  I4. Every remote token is deliberate: terse system prompt, per-category
      output caps, temperature 0.
"""

import json
import os
import re
import sys
import threading
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
PROBE_TIMEOUT_S = float(os.environ.get("PROBE_TIMEOUT_S", "8"))
START = time.time()

SYS = "Answer only what is asked. No preamble, no markdown."
LOCAL_SYS = (
    "Answer every part of the question. State the final answer in the "
    "first sentence, then briefly justify if useful. Follow any format "
    "or length constraints exactly."
)

MAX_TOK = {
    "math": 300, "logic": 170, "code_debug": 400, "code_gen": 380,
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
    """Fireworks via harness proxy; self-locating with short-timeout probes."""

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

        variants = [base]
        if base.endswith("/v1"):
            variants.append(base[: -len("/v1")])
        else:
            variants.append(base + "/v1")
        if not base.endswith("/inference/v1"):
            variants.append(base + "/inference/v1")
        self.base_variants = list(dict.fromkeys(variants))

        self._probe_clients = {
            b: OpenAI(base_url=b, api_key=key, timeout=PROBE_TIMEOUT_S, max_retries=0)
            for b in self.base_variants
        }
        self._locked_clients = {
            b: OpenAI(base_url=b, api_key=key, timeout=25, max_retries=1)
            for b in self.base_variants
        }
        self.tokens_used = 0
        self._tok_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self.locked = None  # (base, id_style) after first success
        self.alive = True

    # -------- model-ID candidates, ordered by category preference
    def _model_variants(self, cat: str):
        prefs = ["kimi"] if cat in ("code_debug", "code_gen") else ["minimax"]
        ordered = []
        for kw in prefs:
            ordered += [m for m in self.allowed
                        if kw in m.lower() and "gemma" not in m.lower()]
        ordered += [m for m in self.allowed
                    if "gemma" not in m.lower() and m not in ordered]
        ordered += [m for m in self.allowed if m not in ordered]
        out = []
        for m in ordered:
            bare = m.split("/")[-1]
            prefixed = m if m.startswith("accounts/") \
                else f"accounts/fireworks/models/{bare}"
            for v in (m, prefixed, bare):
                if v not in out:
                    out.append(v)
        return out

    def _styled(self, cat: str, style: str) -> str:
        """Best model for cat, rendered in the locked ID style."""
        for m in self._model_variants(cat):
            if ("/" in m) == (style == "prefixed"):
                return m
        return self._model_variants(cat)[0]

    def _chat(self, client, model: str, prompt: str, cat: str) -> str:
        resp = client.chat.completions.create(
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
            with self._tok_lock:
                self.tokens_used += (u.prompt_tokens or 0) + (u.completion_tokens or 0)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty completion")
        return text

    def answer(self, prompt: str, cat: str) -> str:
        if self.locked:
            base, style = self.locked
            try:
                return self._chat(self._locked_clients[base],
                                  self._styled(cat, style), prompt, cat)
            except Exception as e:
                log(f"[remote] locked combo failed ({e}); re-probing")
                self.locked = None
        # Serialize probing so 4 workers don't storm the matrix in parallel.
        with self._probe_lock:
            if self.locked:  # another thread locked while we waited
                return self.answer(prompt, cat)
            last = None
            for base in self.base_variants:
                for model in self._model_variants(cat):
                    if remaining() < 15:
                        raise RuntimeError("time exhausted during probing")
                    try:
                        text = self._chat(self._probe_clients[base],
                                          model, prompt, cat)
                        style = "prefixed" if "/" in model else "bare"
                        self.locked = (base, style)
                        log(f"[remote] LOCKED base={base} style={style} model={model}")
                        return text
                    except Exception as e:
                        last = e
            self.alive = False
            raise RuntimeError(f"all remote combos failed: {last}")


# ------------------------------------------------------------ local engine
class Local:
    """Streaming generation, cooperative deadline, slow-mode degradation."""

    def __init__(self):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=max(2, os.cpu_count() or 2),
            verbose=False,
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.slow_mode = False  # set on hard timeout: remote-first thereafter
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
            piece = chunk["choices"][0].get("delta", {}).get("content")
            if piece:
                parts.append(piece)
            if time.monotonic() > deadline:
                log("[local] deadline hit — returning partial answer")
                break
        return "".join(parts).strip()

    def answer(self, prompt: str, cat: str) -> str:
        cap = min(MAX_TOK.get(cat, 150),
                  max(48, int(self.tps * LOCAL_TASK_TIMEOUT_S * 0.8)))
        deadline = time.monotonic() + LOCAL_TASK_TIMEOUT_S
        fut = self.pool.submit(self._stream, prompt, cap, deadline)
        try:
            return fut.result(timeout=LOCAL_TASK_TIMEOUT_S * 1.5 + 5)
        except Exception:
            # Hard timeout: worker is stuck in prompt processing. Degrade.
            self.slow_mode = True
            log("[local] hard timeout — entering slow mode (remote-first)")
            raise


# -------------------------------------------------------------------- main
def run(tasks, results, routes) -> dict:
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

    def try_local(t) -> bool:
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

    def local_first(t) -> bool:
        return (
            local is not None
            and not local.slow_mode
            and cats[str(t["task_id"])] in LOCAL_CATEGORIES
            and len(t.get("prompt", "")) <= LOCAL_MAX_PROMPT_CHARS
        )

    local_q = [t for t in tasks if local_first(t)]
    remote_q = [t for t in tasks if not local_first(t)]

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(try_remote, t) for t in remote_q]

    for t in local_q:
        tid = str(t["task_id"])
        if remaining() < 45 or (local and local.slow_mode):
            futures.append(pool.submit(try_remote, t))
            continue
        if not try_local(t):
            futures.append(pool.submit(try_remote, t))

    for f in futures:
        try:
            f.result(timeout=max(remaining() - 10, 5))
        except Exception:
            pass

    # Final sweep: no empty answers while any path lives.
    for t in tasks:
        tid = str(t["task_id"])
        if results[tid]:
            continue
        if remaining() > 25 and try_remote(t):
            continue
        if remaining() > 15:
            try_local(t)

    return {
        "fireworks_tokens": remote.tokens_used if remote else 0,
        "routes": routes,
        "empty_answers": [k for k, v in results.items() if not v],
        "elapsed_s": round(time.time() - START, 1),
    }


def main() -> int:
    tasks = []
    results: dict = {}
    routes: dict = {}
    summary = {}
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
        results.update({str(t["task_id"]): "" for t in tasks})
        summary = run(tasks, results, routes)
    except Exception as e:  # invariant I1: salvage whatever exists
        log(f"[fatal] {type(e).__name__}: {e} — salvaging output")
        summary = {"fatal": str(e), "routes": routes}

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(
                [{"task_id": tid, "answer": ans} for tid, ans in results.items()],
                f, ensure_ascii=False,
            )
        with open(LOG_PATH, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        log(f"[fatal] could not write output: {e}")
        return 1
    log(f"[done] {len(tasks)} tasks, "
        f"{summary.get('fireworks_tokens', 0)} fireworks tokens, "
        f"{time.time()-START:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
