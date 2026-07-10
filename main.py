#!/usr/bin/env python3
"""TokenCascade v2 — AMD Developer Hackathon ACT II, Track 1.

Design constraints (from the official participant guide):
  - Grading env: 4 GB RAM, 2 vCPU, NO GPU  -> small quantized local model,
    no self-consistency voting, no local LLM-judge (both die on the clock).
  - 10 min total runtime, <30 s per response, 60 s startup, exit 0.
  - Read /input/tasks.json, write /output/results.json [{"task_id","answer"}].
  - FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS injected by the
    harness at runtime. ALL remote calls through the base URL; models ONLY
    from ALLOWED_MODELS (read at runtime — never hardcoded).
  - Local tokens are free; score = Fireworks tokens (ascending) after an
    80% accuracy gate over 19 fixed tasks (16/19 needed).

Strategy: category-routed hybrid.
  regex classifier (free, instant)
    -> categories in LOCAL_CATEGORIES answered by the bundled GGUF model
       (zero score tokens)
    -> everything else answered remotely with terse prompts + output caps
  Global time watchdog: if local inference threatens the runtime budget,
  remaining tasks flush to remote (fast) — a scored answer beats a timeout.

Tuning is ALL environment variables, so the same image flips from safe
hybrid to zero-token local-only without a rebuild:
  LOCAL_CATEGORIES=factual,sentiment,ner,summarization      (hybrid default)
  LOCAL_CATEGORIES=factual,sentiment,ner,summarization,math,logic,code_debug,code_gen
                                                            (local-only push)
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------- config
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
LOG_PATH = os.environ.get("LOG_PATH", "/output/inference_log.json")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")
LOCAL_CATEGORIES = {
    c.strip()
    for c in os.environ.get(
        "LOCAL_CATEGORIES", "factual,sentiment,ner,summarization"
    ).split(",")
    if c.strip()
}
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "520"))  # 600 - slack
LOCAL_TASK_TIMEOUT_S = float(os.environ.get("LOCAL_TASK_TIMEOUT_S", "24"))
LOCAL_MAX_PROMPT_CHARS = int(os.environ.get("LOCAL_MAX_PROMPT_CHARS", "1600"))
START = time.time()

SYS = "Answer directly and concisely."  # counts toward remote input tokens: keep tiny

# Per-category output caps. WHY: remote completion tokens are the score;
# local caps protect the 30s/request clock. Code needs room to be correct —
# a truncated function fails the accuracy gate, which costs more than tokens.
MAX_TOK = {
    "math": 300,
    "logic": 220,
    "code_debug": 400,
    "code_gen": 450,
    "factual": 150,
    "sentiment": 80,
    "ner": 180,
    "summarization": 120,
}


def remaining() -> float:
    return TIME_BUDGET_S - (time.time() - START)


# ------------------------------------------------------------- classifier
# Order matters: earlier rules win. Calibrated on the official practice set.
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
    def __init__(self):
        from openai import OpenAI

        base = os.environ["FIREWORKS_BASE_URL"]  # harness proxy — MANDATORY path
        key = os.environ["FIREWORKS_API_KEY"]    # harness key — never our own
        self.allowed = [
            m.strip()
            for m in os.environ.get("ALLOWED_MODELS", "").split(",")
            if m.strip()
        ]
        if not self.allowed:
            raise RuntimeError("ALLOWED_MODELS is empty")
        self.client = OpenAI(base_url=base, api_key=key, timeout=25, max_retries=1)
        self.tokens_used = 0

    def pick_model(self, cat: str) -> str:
        """Choose from ALLOWED_MODELS at runtime by keyword preference.
        WHY avoid gemma: Fireworks Gemma is on-demand deployment — it can 404
        on the grading account and bills hourly on ours. The MoE serverless
        models (kimi for code, minimax general) are the safe picks — but the
        env list is always the source of truth."""
        prefs = ["kimi"] if cat in ("code_debug", "code_gen") else ["minimax"]
        for kw in prefs:
            for m in self.allowed:
                if kw in m.lower() and "gemma" not in m.lower():
                    return m
        for m in self.allowed:
            if "gemma" not in m.lower():
                return m
        return self.allowed[0]

    def answer(self, prompt: str, cat: str) -> str:
        model = self.pick_model(cat)
        resp = self.client.chat.completions.create(
            model=model,
            temperature=0,  # we pay for these tokens — no dice rolls
            max_tokens=MAX_TOK.get(cat, 200),
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
            ],
        )
        u = resp.usage
        if u:
            self.tokens_used += (u.prompt_tokens or 0) + (u.completion_tokens or 0)
        return (resp.choices[0].message.content or "").strip()


# ------------------------------------------------------------ local engine
class Local:
    """Bundled GGUF via llama.cpp on CPU. One worker thread so generations
    are sequential; a timed-out generation keeps running in background (can't
    be killed), so after 2 consecutive timeouts local is disabled entirely —
    a stalled local model must not drag remote tasks past the 10-min wall."""

    def __init__(self):
        from llama_cpp import Llama

        t0 = time.time()
        self.llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=3072,
            n_threads=max(2, os.cpu_count() or 2),
            verbose=False,
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.consecutive_timeouts = 0
        self.dead = False
        # Warmup + speed probe: dynamic output caps sized to the 30s clock.
        gen_t0 = time.time()
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say OK."}],
            max_tokens=16,
            temperature=0,
        )
        gen_dt = max(time.time() - gen_t0, 0.1)
        toks = out.get("usage", {}).get("completion_tokens", 8) or 8
        self.tps = max(toks / gen_dt, 2.0)
        # Leave 30% of the per-task window for prompt processing.
        self.dyn_cap = max(48, int(self.tps * LOCAL_TASK_TIMEOUT_S * 0.7))
        print(
            f"[local] loaded in {t0 and time.time()-t0:.1f}s, "
            f"~{self.tps:.1f} tok/s, dynamic cap {self.dyn_cap}",
            file=sys.stderr,
        )

    def answer(self, prompt: str, cat: str) -> str:
        if self.dead:
            raise RuntimeError("local engine disabled")
        cap = min(MAX_TOK.get(cat, 150), self.dyn_cap)
        fut = self.pool.submit(self._gen, prompt, cap)
        try:
            text = fut.result(timeout=LOCAL_TASK_TIMEOUT_S)
            self.consecutive_timeouts = 0
            if not text:
                raise RuntimeError("empty local answer")
            return text
        except TimeoutError:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= 2:
                self.dead = True
                print("[local] disabled after repeated timeouts", file=sys.stderr)
            raise

    def _gen(self, prompt: str, cap: int) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=cap,
        )
        return out["choices"][0]["message"]["content"].strip()


# -------------------------------------------------------------------- main
def main() -> int:
    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    results: dict[str, str] = {str(t["task_id"]): "" for t in tasks}
    routes: dict[str, str] = {}

    remote = None
    try:
        remote = Remote()
    except Exception as e:  # no env vars in a weird run — degrade to local-only
        print(f"[remote] unavailable: {e}", file=sys.stderr)

    local = None
    try:
        local = Local()
    except Exception as e:  # model missing/corrupt — degrade to remote-only
        print(f"[local] unavailable: {e}", file=sys.stderr)

    cats = {str(t["task_id"]): classify(t.get("prompt", "")) for t in tasks}

    def go_local(t) -> bool:
        tid = str(t["task_id"])
        return (
            local is not None
            and not local.dead
            and cats[tid] in LOCAL_CATEGORIES
            and len(t.get("prompt", "")) <= LOCAL_MAX_PROMPT_CHARS
        )

    local_q = [t for t in tasks if go_local(t)]
    remote_q = [t for t in tasks if not go_local(t)]

    # Remote tasks run concurrently (I/O bound) while local grinds sequentially.
    def do_remote(t):
        tid = str(t["task_id"])
        try:
            results[tid] = remote.answer(t["prompt"], cats[tid])
            routes[tid] = f"remote:{cats[tid]}"
        except Exception as e:
            print(f"[remote] {tid} failed: {e}", file=sys.stderr)
            routes[tid] = "remote:failed"

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(do_remote, t) for t in remote_q if remote]

    for t in local_q:
        tid = str(t["task_id"])
        # Watchdog: reserve time for outstanding remote work + writeout.
        if remaining() < 45:
            print("[watchdog] flushing remaining local tasks to remote", file=sys.stderr)
            if remote:
                futures.append(pool.submit(do_remote, t))
            continue
        try:
            results[tid] = local.answer(t["prompt"], cats[tid])
            routes[tid] = f"local:{cats[tid]}"
        except Exception:
            if remote:
                futures.append(pool.submit(do_remote, t))
            else:
                routes[tid] = "local:failed"

    for f in futures:
        try:
            f.result(timeout=max(remaining() - 10, 5))
        except Exception:
            pass

    # Never-empty fallback: a wrong answer can score; a blank never does.
    if remote:
        for t in tasks:
            tid = str(t["task_id"])
            if not results[tid] and remaining() > 20:
                do_remote(t)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            [{"task_id": tid, "answer": ans} for tid, ans in results.items()],
            f,
            ensure_ascii=False,
        )
    try:
        with open(LOG_PATH, "w") as f:
            json.dump(
                {
                    "fireworks_tokens": remote.tokens_used if remote else 0,
                    "routes": routes,
                    "elapsed_s": round(time.time() - START, 1),
                },
                f,
                indent=2,
            )
    except Exception:
        pass  # log is best-effort; results.json is what scores

    print(
        f"[done] {len(tasks)} tasks, "
        f"{(remote.tokens_used if remote else 0)} fireworks tokens, "
        f"{time.time()-START:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
