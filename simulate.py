#!/usr/bin/env python3
"""v5 end-to-end simulation. Mock Fireworks over real sockets + stub llama.

Scenarios:
  S1 happy hybrid        S2 wrong-base/bare-IDs (matrix lock)
  S3 remote dead         S4 local dead
  S5 math tool corrects a wrong local answer
  S6 compile gate: broken local code -> remote fallback
  S7 corrupt input -> salvage, exit 0
  S8 SIGKILL mid-run -> incremental results already on disk (subprocess)
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "main.py")


# ----------------------------------------------------------- mock server
class MockFW(BaseHTTPRequestHandler):
    accepted_path = "/v1/chat/completions"
    accepted_style = "prefixed"
    dead = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body.get("model", "")
        if MockFW.dead:
            self.send_response(500); self.end_headers(); return
        style = "prefixed" if "/" in model else "bare"
        if self.path != MockFW.accepted_path or style != MockFW.accepted_style:
            self.send_response(404)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"code":"model_not_found"}}'); return
        user = body["messages"][-1]["content"]
        resp = {"choices": [{"message": {"content": f"MOCK-REMOTE: {user[:40]}"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30}}
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers(); self.wfile.write(payload)


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockFW)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ------------------------------------------------------- stub llama_cpp
def llama_stub(mode="good", per_call_delay=0.0, fail=False):
    mod = types.ModuleType("llama_cpp")

    class Llama:  # noqa
        def __init__(self, *a, **k):
            if fail:
                raise RuntimeError("model missing (simulated)")

        def create_chat_completion(self, messages, **k):
            if per_call_delay:
                time.sleep(per_call_delay)
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if "arithmetic expression" in system:
                content = "1000 - 1000*0.30"        # correct expression
            elif mode == "badmath" and "1000 pesos" in user:
                if "correct computed value" in user:
                    content = "The sale price is 700 pesos."
                else:
                    content = "The sale price is 650 pesos."   # wrong number
            elif mode == "badcode" and "bug" in user.lower():
                content = "```python\ndef double(x) return x * 2\n```"  # SyntaxError
            elif "bug" in user.lower():
                content = "```python\ndef double(x):\n    return x * 2\n```"
            else:
                content = f"MOCK-LOCAL: {user[:40]}"
            return {"choices": [{"message": {"content": content}}],
                    "usage": {"completion_tokens": 4}}
    mod.Llama = Llama
    return mod


def load_agent(stub):
    sys.modules["llama_cpp"] = stub
    src = open(AGENT).read().replace(
        'if __name__ == "__main__":\n    sys.exit(main())', "")
    ns = {}
    exec(compile(src, "main.py", "exec"), ns)
    return ns


TASKS = [
    {"task_id": "t-fact", "prompt": "What is the capital of Japan and what island is it on?"},
    {"task_id": "t-math", "prompt": "A jacket costs 1000 pesos with a 30% discount. How many pesos is the sale price?"},
    {"task_id": "t-sent", "prompt": "Classify the sentiment of this review: Great camera, weak battery."},
    {"task_id": "t-summ", "prompt": "Summarize in one sentence: The team shipped early and got a bonus after automating tests."},
    {"task_id": "t-ner",  "prompt": "Extract all named entities and their types from: Liza Reyes flew to Osaka for Globe Telecom in June."},
    {"task_id": "t-dbg",  "prompt": "This function should double a number but has a bug: def double(x): return x + 1. Find and fix it."},
    {"task_id": "t-logic", "prompt": "Three friends, A, B, and C, each own a different fruit. A does not own the cherry. B owns the banana. Who owns the cherry?"},
    {"task_id": "t-code", "prompt": "Write a Python function that counts vowels in a string."},
]


def scenario(name, stub, base_suffix, path, style, dead=False,
             corrupt=False, env_extra=None):
    srv, url = serve()
    MockFW.accepted_path, MockFW.accepted_style, MockFW.dead = path, style, dead
    tmp = os.path.join(HERE, f"sim_{name}"); os.makedirs(tmp, exist_ok=True)
    inp = os.path.join(tmp, "tasks.json")
    open(inp, "w").write("{broken" if corrupt else json.dumps(TASKS))
    env = {"INPUT_PATH": inp,
           "OUTPUT_PATH": os.path.join(tmp, "results.json"),
           "LOG_PATH": os.path.join(tmp, "log.json"),
           "FIREWORKS_API_KEY": "mock",
           "FIREWORKS_BASE_URL": url + base_suffix,
           "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code,gemma-4-31b-it",
           "TIME_BUDGET_S": "120", "PROBE_TIMEOUT_S": "3", "RESERVE_S": "5"}
    env.update(env_extra or {})
    old = dict(os.environ); os.environ.update(env)
    try:
        agent = load_agent(stub)
        agent["START"] = time.time()
        agent["TIME_BUDGET_S"] = 120.0
        agent["PROBE_TIMEOUT_S"] = 3.0
        agent["RESERVE_S"] = 5.0
        rc = agent["main"]()
    finally:
        os.environ.clear(); os.environ.update(old); srv.shutdown()
    res = None
    rp = os.path.join(tmp, "results.json")
    if os.path.exists(rp):
        res = json.load(open(rp))
    lg = {}
    lp = os.path.join(tmp, "log.json")
    if os.path.exists(lp):
        lg = json.load(open(lp))
    return rc, res, lg


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    return bool(cond)


def main():
    ok = True

    print("S1 happy hybrid")
    rc, res, lg = scenario("s1", llama_stub(), "/v1", "/v1/chat/completions", "prefixed")
    routes = lg.get("routes", {})
    ok &= check("exit 0, 8 answers, none empty", rc == 0 and res and len(res) == 8
                and all(r["answer"] for r in res))
    ok &= check("logic+code_gen remote, rest local",
                routes.get("t-logic", "").startswith("remote:")
                and routes.get("t-code", "").startswith("remote:")
                and sum(1 for v in routes.values() if v.startswith("local:")) == 6,
                str(sorted(routes.values())))
    ok &= check("tokens counted", lg.get("fireworks_tokens", 0) > 0,
                f"tokens={lg.get('fireworks_tokens')}")

    print("S2 wrong base + bare IDs only (matrix must lock)")
    rc, res, lg = scenario("s2", llama_stub(), "", "/v1/chat/completions", "bare")
    ok &= check("exit 0, none empty", rc == 0 and res and all(r["answer"] for r in res))
    ok &= check("remote locked", any(v.startswith("remote:")
                for v in lg.get("routes", {}).values()))

    print("S3 remote dead")
    rc, res, lg = scenario("s3", llama_stub(), "/v1", "/v1/chat/completions",
                           "prefixed", dead=True)
    ok &= check("exit 0, all 8 local, zero tokens",
                rc == 0 and res and all(r["answer"] for r in res)
                and lg.get("fireworks_tokens") == 0
                and sum(1 for v in lg.get("routes", {}).values()
                        if v.startswith("local:")) == 8)

    print("S4 local dead")
    rc, res, lg = scenario("s4", llama_stub(fail=True), "/v1",
                           "/v1/chat/completions", "prefixed")
    ok &= check("exit 0, all 8 remote, none empty",
                rc == 0 and res and all(r["answer"] for r in res)
                and all(v.startswith("remote:")
                        for v in lg.get("routes", {}).values()))

    print("S5 math tool corrects a wrong local answer (650 -> 700)")
    rc, res, lg = scenario("s5", llama_stub(mode="badmath"), "/v1",
                           "/v1/chat/completions", "prefixed")
    math_ans = next(r["answer"] for r in res if r["task_id"] == "t-math")
    ok &= check("corrected value present, wrong value gone",
                "700" in math_ans and "650" not in math_ans, repr(math_ans))

    print("S6 compile gate: broken local code -> remote fallback")
    rc, res, lg = scenario("s6", llama_stub(mode="badcode"), "/v1",
                           "/v1/chat/completions", "prefixed")
    ok &= check("t-dbg escalated to remote",
                lg.get("routes", {}).get("t-dbg", "").startswith("remote:"),
                lg.get("routes", {}).get("t-dbg"))

    print("S7 corrupt input -> salvage")
    rc, res, lg = scenario("s7", llama_stub(), "/v1", "/v1/chat/completions",
                           "prefixed", corrupt=True)
    ok &= check("exit 0 with salvage results.json", rc == 0 and res == [])

    print("S8 SIGKILL mid-run -> incremental results on disk")
    tmp = os.path.join(HERE, "sim_s8"); os.makedirs(tmp, exist_ok=True)
    inp = os.path.join(tmp, "tasks.json"); json.dump(TASKS, open(inp, "w"))
    stubdir = os.path.join(tmp, "stub"); os.makedirs(stubdir, exist_ok=True)
    open(os.path.join(stubdir, "llama_cpp.py"), "w").write(
        "import time\n"
        "class Llama:\n"
        "    def __init__(self,*a,**k): pass\n"
        "    def create_chat_completion(self, messages, **k):\n"
        "        time.sleep(1.0)\n"
        "        return {'choices':[{'message':{'content':'slow local answer'}}],\n"
        "                'usage':{'completion_tokens':4}}\n")
    srv, url = serve()
    MockFW.accepted_path, MockFW.accepted_style, MockFW.dead = "/v1/chat/completions", "prefixed", False
    env = dict(os.environ,
               INPUT_PATH=inp, OUTPUT_PATH=os.path.join(tmp, "results.json"),
               LOG_PATH=os.path.join(tmp, "log.json"),
               FIREWORKS_API_KEY="mock", FIREWORKS_BASE_URL=url + "/v1",
               ALLOWED_MODELS="minimax-m3,kimi-k2p7-code",
               TIME_BUDGET_S="120", PROBE_TIMEOUT_S="3", RESERVE_S="5",
               PYTHONPATH=stubdir)
    proc = subprocess.Popen([sys.executable, AGENT], env=env,
                            stderr=subprocess.DEVNULL)
    time.sleep(6)                      # let a few tasks bank
    proc.send_signal(signal.SIGKILL)   # uncatchable, like a segfault/OOM
    proc.wait(); srv.shutdown()
    rp = os.path.join(tmp, "results.json")
    banked = 0
    if os.path.exists(rp):
        banked = sum(1 for r in json.load(open(rp)) if r["answer"])
    ok &= check("results.json survived the kill with banked answers",
                banked >= 1, f"banked={banked}")

    print("\nOVERALL:", "ALL SCENARIOS PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
