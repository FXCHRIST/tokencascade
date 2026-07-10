#!/usr/bin/env bash
# Full container dry-run against the OFFICIAL practice tasks — mimics the
# harness exactly (mounted /input + /output, env vars injected).
# Uses YOUR Fireworks key (dev only; the harness injects its own at grading).
set -euo pipefail

IMAGE="${IMAGE:-tokencascade:local}"
: "${FIREWORKS_API_KEY:?export FIREWORKS_API_KEY first (your dev key)}"
: "${ALLOWED_MODELS:?export ALLOWED_MODELS first (paste the launch-day list verbatim)}"

mkdir -p /tmp/tc_out
docker run --rm \
  --memory=4g --cpus=2 \
  -v "$(pwd)/practice:/input:ro" \
  -v /tmp/tc_out:/output \
  -e FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}" \
  -e ALLOWED_MODELS \
  -e LOCAL_CATEGORIES="${LOCAL_CATEGORIES:-factual,sentiment,ner,summarization}" \
  "$IMAGE"

echo "--- results.json ---"
cat /tmp/tc_out/results.json | python3 -m json.tool
echo "--- inference_log.json ---"
cat /tmp/tc_out/inference_log.json
