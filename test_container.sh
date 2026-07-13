#!/usr/bin/env bash
# Full container dry-run against the official practice tasks — mimics the
# harness (mounted /input + /output, 2 CPUs, 4 GB RAM). Zero env needed:
# the agent reads no secrets and calls no APIs.
#
#   IMAGE=ghcr.io/YOUR_USER/tokencascade:v6-rc1 ./test_container.sh
#
# On Apple Silicon this runs under emulation and is SLOW — smoke test only;
# timing truth comes from the CI devset gate.
set -euo pipefail
IMAGE="${IMAGE:-tokencascade:local}"
mkdir -p /tmp/tc_out && rm -f /tmp/tc_out/*.json
docker run --rm \
  --memory=4g --cpus=2 \
  -v "$(pwd)/practice:/input:ro" \
  -v /tmp/tc_out:/output \
  ${TIME_BUDGET_S:+-e TIME_BUDGET_S=$TIME_BUDGET_S} \
  "$IMAGE"
echo "--- results.json ---"
python3 -m json.tool < /tmp/tc_out/results.json
echo "--- inference_log.json ---"
python3 -m json.tool < /tmp/tc_out/inference_log.json | head -40
