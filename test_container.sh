#!/usr/bin/env bash
# Harness-faithful container test. Passes LOCAL_CATEGORIES ONLY if you
# exported it — otherwise the image's baked ENV governs (the harness never
# injects it, so neither do we by default).
set -euo pipefail
IMAGE="${IMAGE:-ghcr.io/fxchrist/tokencascade:latest}"
: "${FIREWORKS_API_KEY:?export FIREWORKS_API_KEY first}"
: "${ALLOWED_MODELS:?export ALLOWED_MODELS first}"
mkdir -p /tmp/tc_out
docker run --rm --platform linux/amd64 --memory=4g --cpus=2 \
  -v "$(pwd)/practice:/input:ro" -v /tmp/tc_out:/output \
  -e FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}" \
  -e ALLOWED_MODELS \
  ${LOCAL_CATEGORIES:+-e LOCAL_CATEGORIES} \
  ${LOCAL_TASK_TIMEOUT_S:+-e LOCAL_TASK_TIMEOUT_S} \
  "$IMAGE"
echo "--- results.json ---"; python3 -m json.tool /tmp/tc_out/results.json
echo "--- inference_log.json ---"; cat /tmp/tc_out/inference_log.json
