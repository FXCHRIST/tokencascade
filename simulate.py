#!/usr/bin/env python3
"""End-to-end simulation of main_v4 against a mock harness.

Mocks:
  - A real HTTP server speaking the OpenAI chat-completions protocol, with
    configurable accepted URL path and model-ID style (tests the fallback
    matrix through the REAL openai SDK over REAL sockets).
  - A stub llama_cpp module with configurable speed and canned answers.

Scenarios:
  S1 happy       : correct /v1 base, prefixed IDs      -> hybrid, no probes wasted
  S2 harness-bug : base missing /v1, bare IDs only     -> matrix must lock
  S3 remote-dead : server rejects everything            -> local answers all it can
  S4 local-dead  : llama import fails                   -> all-remote
  S5 local-slow  : local generation crawls              -> partials + slow-mode, no empties
  S6 crash       : malformed input file                 -> salvage: results.json still written, exit 0
"""

import json
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AUDIT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------- mock server
class MockFireworks(BaseHTTPRequestHandler):
    accepted_path = "/v1/chat/completions"
    accepted_style = "prefixed"  # or "bare"
    dead = False
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body.get("model", "")
        MockFireworks.calls.append((self.path, model))
        if MockFireworks.dead:
            self.send_response(500); self.end_headers()
            self.wfile.write(b'{"error":"dead"}'); return
        style = "prefixed" if "/" in model else "bare"
        if self.path != MockFireworks.accepted_path or style != MockFireworks.accepted_style:
            self.send_response(404)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"code":"model_not_found"}}'); return
        user = body["messages"][-1]["content"]
        ans = "MOCK-REMOTE: 4" if "2+2" in user else f"MOCK-REMOTE answer to: {user[:40]}"
        resp = {
            "choices": [{"message": {"role": "assistant", "content": ans}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockFireworks)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ------------------------------------------------------- stub llama_cpp
def make_llama_stub(delay_per_token=0.0, fail_import=False):
    mod = types.ModuleType("llama_cpp")
    if fail_import:
        class Llama:  # noqa
            def __init__(self, *a, **k):
                raise RuntimeError("model file missing (simulated)")
        mod.Llama = Llama
        return mod

    class Llama:  # noqa
        def __init__(self, *a, **k):
            pass

        def create_chat_completion(self, messages, stream=False,
                                   max_tokens=64, temperature=0, **k):
            user = messages[-1]["content"]
            ans = "MOCK-LOCAL: 4" if "2+2" in user else f"MOCK-LOCAL answer to: {user[:40]}"
            if not stream:
                return {"choices": [{"message": {"content": ans}}],
                        "usage": {"completion_tokens": 4}}
            def gen():
                for word in ans.split(" "):
                    if delay_per_token:
                        time.sleep(delay_per_token)
                    yield {"choices": [{"delta": {"content": word + " "}}]}
            return gen()
    mod.Llama = Llama
    return mod


# ------------------------------------------------------------- scenarios
def load_agent(llama_stub):
    sys.modules["llama_cpp"] = llama_stub
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent", os.path.join(AUDIT, "main_v4.py"))
    agent = importlib.util.module_from_spec(spec)
    src = open(os.path.join(AUDIT, "main_v4.py")).read()
    src = src.replace('if __name__ == "__main__":\n    sys.exit(main())', "")
    exec(compile(src, "main_v4.py", "exec"), agent.__dict__)
    return agent


def make_tasks(path, n=19):
    cats = [
        "What is the capital of Japan and what island is it on?",
        "A jacket costs 1000 pesos with a 30% discount. How many pesos is the sale price?",
        "Classify the sentiment of this review: The camera is superb but the battery dies fast.",
        "Summarize in one sentence: The team shipped the release two weeks early after automating their test suite, and management approved a bonus.",
        "Extract all named entities and their types from: Liza Reyes flew to Osaka for Globe Telecom in June.",
        "This function should double a number but has a bug: def double(x): return x + 1. Find and fix it.",
        "Three friends, A, B, and C, each own a different fruit: apple, banana, cherry. A does not own the cherry. B owns the banana. Who owns the cherry?",
        "Write a Python function that counts vowels in a string.",
    ]
    tasks = [{"task_id": f"t{i+1}", "prompt": cats[i % len(cats)]} for i in range(n)]
    json.dump(tasks, open(path, "w"))
    return tasks


def run_scenario(name, base_env, llama_stub, accepted_path, accepted_style,
                 dead=False, corrupt_input=False, budget="120"):
    srv, url = start_server()
    MockFireworks.accepted_path = accepted_path
    MockFireworks.accepted_style = accepted_style
    MockFireworks.dead = dead
    MockFireworks.calls = []

    tmp = os.path.join(AUDIT, f"sim_{name}")
    os.makedirs(tmp, exist_ok=True)
    inp = os.path.join(tmp, "tasks.json")
    if corrupt_input:
        open(inp, "w").write("{not json")
    else:
        make_tasks(inp)

    env = {
        "INPUT_PATH": inp,
        "OUTPUT_PATH": os.path.join(tmp, "results.json"),
        "LOG_PATH": os.path.join(tmp, "log.json"),
        "FIREWORKS_API_KEY": "mock",
        "FIREWORKS_BASE_URL": url + base_env,
        "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code,gemma-4-31b-it",
        "TIME_BUDGET_S": budget,
        "LOCAL_TASK_TIMEOUT_S": "2",
        "PROBE_TIMEOUT_S": "3",
    }
    old = dict(os.environ)
    os.environ.update(env)
    try:
        agent = load_agent(llama_stub)
        agent.START = time.time()  # reset budget clock per scenario
        agent.TIME_BUDGET_S = float(budget)
        agent.LOCAL_TASK_TIMEOUT_S = 2.0
        agent.PROBE_TIMEOUT_S = 3.0
        rc = agent.main()
    finally:
        os.environ.clear(); os.environ.update(old)
        srv.shutdown()

    out_path = os.path.join(tmp, "results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else None
    logj = json.load(open(os.path.join(tmp, "log.json"))) if os.path.exists(os.path.join(tmp, "log.json")) else {}
    return rc, results, logj


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    return cond


def main():
    ok = True
    print("S1 happy path (/v1 base, prefixed IDs)")
    rc, res, lg = run_scenario("s1", "/v1", make_llama_stub(), "/v1/chat/completions", "prefixed")
    ok &= check("exit 0", rc == 0)
    ok &= check("19 answers, none empty", res is not None and len(res) == 19 and all(r["answer"] for r in res))
    routes = lg.get("routes", {})
    ok &= check("hybrid routing", any(v.startswith("local:") for v in routes.values())
                and any(v.startswith("remote:") for v in routes.values()), str(sorted(set(routes.values()))))
    ok &= check("tokens counted", lg.get("fireworks_tokens", 0) > 0, f"tokens={lg.get('fireworks_tokens')}")

    print("S2 harness-bug path (base missing /v1, bare IDs only)")
    rc, res, lg = run_scenario("s2", "", make_llama_stub(), "/v1/chat/completions", "bare")
    ok &= check("exit 0", rc == 0)
    ok &= check("no empty answers", res and all(r["answer"] for r in res))
    ok &= check("matrix locked (remote answers exist)",
                any(v.startswith("remote:") for v in lg.get("routes", {}).values()))

    print("S3 remote dead")
    rc, res, lg = run_scenario("s3", "/v1", make_llama_stub(), "/v1/chat/completions", "prefixed", dead=True)
    ok &= check("exit 0", rc == 0)
    local_ct = sum(1 for v in lg.get("routes", {}).values() if v.startswith("local:"))
    ok &= check("local answered everything it could (final sweep incl. logic/codegen)",
                local_ct == 19, f"local={local_ct}")
    ok &= check("zero tokens", lg.get("fireworks_tokens") == 0)

    print("S4 local dead")
    rc, res, lg = run_scenario("s4", "/v1", make_llama_stub(fail_import=True), "/v1/chat/completions", "prefixed")
    ok &= check("exit 0", rc == 0)
    ok &= check("all 19 remote, none empty",
                res and all(r["answer"] for r in res)
                and all(v.startswith("remote:") for v in lg.get("routes", {}).values()))

    print("S5 local slow (0.6s/token vs 2s deadline)")
    rc, res, lg = run_scenario("s5", "/v1", make_llama_stub(delay_per_token=0.6), "/v1/chat/completions", "prefixed")
    ok &= check("exit 0", rc == 0)
    ok &= check("no empty answers (partials or remote fallback)", res and all(r["answer"] for r in res))

    print("S6 corrupt input (salvage invariant)")
    rc, res, lg = run_scenario("s6", "/v1", make_llama_stub(), "/v1/chat/completions", "prefixed", corrupt_input=True)
    ok &= check("exit 0 with salvage results.json", rc == 0 and res == [])

    print("\nOVERALL:", "ALL SCENARIOS PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
