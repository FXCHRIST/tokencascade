# TokenCascade — Zero-Token Local Agent (AMD Developer Hackathon ACT II, Track 1)

A fully local, containerized AI agent that answers all eight Track 1 task
categories with a bundled **Qwen3-4B-Instruct-2507** (Q4_K_M GGUF) running
on CPU via llama.cpp — **zero Fireworks tokens by construction**. Per the
Track 1 rules, only inference routed through `FIREWORKS_BASE_URL` counts
toward the token score, and answers produced by local models count fully
toward accuracy; this agent therefore submits the best possible token score
(0) and competes purely on correctness, format compliance, and reliability.

## How it works

```
/input/tasks.json
      │
      ▼
 classify ──► factual · sentiment · ner · summarization   (prose pipelines)
      │       math · logic · code_debug · code_gen        (verified pipelines)
      ▼
 local Qwen3-4B-Instruct-2507 (llama.cpp, CPU, temperature 0)
      │
      ▼
 verification layer (free compute, zero tokens):
   • math    — model proposes the arithmetic expression, Python computes it;
               mismatch triggers a hint-guided regeneration
   • code    — candidate must compile AND execute; failures regenerate with
               the actual error message
   • format  — "exactly N sentences" / "N bullets, ≤K words" constraints are
               parsed from the prompt, validated, regenerated on violation,
               and deterministically repaired as a last resort
   • sentiment — mixed reviews are guarded against a bare "Negative" label
      │
      ▼
 atomic per-task flush ──► /output/results.json  (+ /output/inference_log.json)
```

Reliability guarantees:

- **Sequential, no threads, no interrupted native calls** — the failure
  classes observed in earlier revisions are structurally impossible.
- **Crash-safe output** — results are flushed atomically after every task;
  SIGTERM flushes and exits 0. Every input `task_id` is always present in
  the output.
- **No runtime downloads, no secrets** — model weights are baked into the
  image at build time with a pinned SHA256; `FIREWORKS_API_KEY` is neither
  required nor read.
- **Time governor** — an EMA of per-task latency switches the agent into a
  fast mode (shorter caps, no retries) before the runtime limit is at risk.

## Build & push (GitHub Actions, recommended)

Actions → **build-push** → Run workflow → choose an image tag.
The workflow first runs the full devset on 2 threads (harness-like CPU
budget) and only builds/pushes `linux/amd64` to GHCR if the gate passes
(accuracy ≥ 20/22 and runtime ≤ 420 s). After the first push, make the
GHCR package **public** or the judge will `PULL_ERROR`.

## Build locally

```bash
docker buildx build --platform linux/amd64 \
  -t ghcr.io/YOUR_USER/tokencascade:TAG --push .
```

## Test like the harness

```bash
IMAGE=ghcr.io/YOUR_USER/tokencascade:TAG ./test_container.sh
```

No environment variables are needed. On Apple Silicon this runs under
emulation and is slow — treat it as a smoke test only; timing truth comes
from the CI gate.

## Devset measurement

```bash
curl -fL -o model.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
LOCAL_MODEL_PATH=./model.gguf python devset/check.py        # full speed
THREADS=2 LOCAL_MODEL_PATH=./model.gguf python devset/check.py  # harness-like
```

The devset includes all ten official public validation examples from the
judging Self-Check guide plus harder synthetic logic/code/math tasks, graded
on gold keywords, forbidden labels, exact format constraints, and executed
code tests.

## Tuning (env, no rebuild)

| Variable | Default | Purpose |
|---|---|---|
| `TIME_BUDGET_S` | 520 | wall-clock budget before the governor stops |
| `THREADS` | all cores | llama.cpp threads |
| `N_CTX` | 3072 | context window |

## AMD / Fireworks usage

Built for the AMD Developer Hackathon ACT II Track 1 standardized scoring
environment (`linux/amd64`). Development and evaluation used the Fireworks
AI ecosystem for baseline comparisons; the submitted agent intentionally
routes zero tokens through Fireworks, which the track rules state is the
best possible outcome for ranking.

MIT License.
