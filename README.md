# TokenCascade v2 — AMD Developer Hackathon ACT II, Track 1

A category-routed hybrid agent: a small local GGUF model (zero score tokens)
handles the categories it is measured to be reliable on; frontier Fireworks
models handle the rest through the harness proxy with terse prompts and hard
output caps. A global time watchdog keeps the run inside the 10-minute limit.

- Reads `/input/tasks.json`, writes `/output/results.json` (+ best-effort
  `/output/inference_log.json`), exits 0.
- `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` are read from
  the environment at runtime — nothing is hardcoded.
- Model weights are bundled in the image (no runtime downloads).

## Build & push (GitHub Actions, recommended)
Push this repo to GitHub → Actions → "build-push" → Run workflow → then make
the GHCR package **public**.

## Build locally
    docker build -t ghcr.io/YOUR_USER/tokencascade:latest .
    # Apple Silicon: docker buildx build --platform linux/amd64 -t ... --push .

## Test like the harness
    export FIREWORKS_API_KEY=your_dev_key
    export ALLOWED_MODELS="<paste launch-day list>"
    IMAGE=ghcr.io/YOUR_USER/tokencascade:latest ./test_container.sh

## Tune (no rebuild needed)
    LOCAL_CATEGORIES=factual,sentiment,ner,summarization              # safe hybrid
    LOCAL_CATEGORIES=factual,sentiment,ner,summarization,math,logic,code_debug,code_gen  # zero-token push

## Dev-set measurement
    python devset/check.py   # see header of that file for env setup

MIT License.
