# TokenCascade v6 — AMD Developer Hackathon ACT II, Track 1

A fully local, zero-Fireworks-token agent. Every task in `/input/tasks.json`
is answered by a bundled **Qwen3-4B-Instruct-2507** (Q4_K_M GGUF) running on
CPU via llama.cpp. No remote inference exists in this codebase, so the scored
token count is **0 by construction** — the agent competes purely on
correctness, format compliance, reliability, and runtime.

## Architecture

```
/input/tasks.json
      │
      ▼
 regex classifier ──► factual · sentiment · summarization · ner
      │               math · logic · code_debug · code_gen
      ▼
 category-specific system prompt + local Qwen3-4B (llama.cpp, CPU)
      │
      ▼
 verification layer (free compute, zero tokens):
   • math      — solved TWICE independently: a natural-language derivation
                 AND a model-written Python script executed in a subprocess.
                 Agreement required; disagreement triggers a tie-breaking
                 resample; execution-verified code arbitrates.
   • code      — must parse (ast) AND execute; failures regenerate with the
                 ACTUAL error message fed back to the model.
   • ner       — when JSON is requested, sampling is constrained by a GBNF
                 grammar (malformed JSON is structurally impossible), with a
                 3-stage repair fallback behind it.
   • format    — "exactly N sentences / N bullets / ≤K words / exactly K
                 words" constraints are parsed from the prompt, validated,
                 regenerated on violation, and deterministically repaired at
                 sentence/bullet boundaries as a last resort.
   • sentiment — rubric-guarded: mixed reviews must acknowledge both sides,
                 are never labeled bare Negative, and the label always leads.
      │
      ▼
 crash-safe sink ──► /output/results.json  (+ /output/inference_log.json)
```

## Reliability guarantees

- **Sequential, no threads, no interrupted native calls** — the failure
  classes observed in earlier revisions are structurally impossible.
- **Crash-safe from second zero** — `results.json` contains every `task_id`
  from the moment the run starts and is rewritten atomically after every
  completed task. SIGTERM flushes and exits 0. A crash at task 17 leaves 16
  real answers on disk, never a missing file or missing task.
- **Portable binary** — llama-cpp-python is compiled in the image with
  `GGML_NATIVE=OFF`, so the build never targets the build machine's CPU
  instruction set. It runs on any x86-64 host, including the judging CPU.
- **No runtime downloads, no secrets** — weights are baked in at build time
  with a pinned SHA256; `FIREWORKS_API_KEY` is neither required nor read.
- **Time governor** — an EMA of per-task latency switches to a fast mode
  (shorter caps, no retries) before the runtime limit is at risk; cheap
  categories run first so late timeouts cost the fewest answers.

## Build & push (GitHub Actions, recommended)

Actions → **build-push** → Run workflow → choose an image tag. The workflow
runs the full devset gate on 2 threads (harness-like CPU budget) and only
builds/pushes `linux/amd64` to GHCR if the gate passes (≥20/22 correct,
≤420 s). After the first push, make the GHCR package **public** or the judge
will `PULL_ERROR`.

## Test like the harness

```bash
IMAGE=ghcr.io/YOUR_USER/tokencascade:TAG ./test_container.sh
```

No environment variables are needed. On Apple Silicon this runs under
emulation and is slow — treat it as a smoke test only; timing truth comes
from the CI gate.

## Devset measurement (no Docker)

```bash
curl -fL -o model.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
THREADS=2 LOCAL_MODEL_PATH=./model.gguf python devset/check.py
```

The devset includes all ten official public validation examples from the
judging Self-Check guide plus harder synthetic logic/code/math tasks, graded
on gold keywords, forbidden labels, exact format constraints, numeric
answers, and executed code tests.

## Pipeline simulation (no model, no Docker, <30 s)

```bash
python3 sim/simulate.py
```

Nine scenarios exercise every deterministic layer against a mock llama.cpp:
routing, all verification/repair paths, SIGTERM mid-run, and model-load
failure.

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
best possible outcome for ranking after the accuracy gate.

MIT License.
