#!/usr/bin/env bash
# Test the built image exactly the way the judging harness runs it:
# a clean pull, /input mounted read-only, /output collected, no manual setup.
# No Fireworks credentials are needed — the agent is fully local.
#
#   IMAGE=ghcr.io/fxchrist/tokencascade:rc1 ./test_container.sh
#
# On Apple Silicon this runs under emulation and will be slow — use it only
# as a smoke test. Timing truth comes from the CI gate job (linux/amd64).
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/fxchrist/tokencascade:latest}"
TASKS="${TASKS:-practice/tasks.json}"
WORK="$(mktemp -d)"

mkdir -p "$WORK/input" "$WORK/output"
cp "$TASKS" "$WORK/input/tasks.json"

echo "[test] pulling $IMAGE (linux/amd64)"
docker pull --platform linux/amd64 "$IMAGE"

echo "[test] running container"
docker run --rm --platform linux/amd64 \
  -v "$WORK/input:/input:ro" \
  -v "$WORK/output:/output" \
  "$IMAGE"

echo "[test] validating output"
python3 - "$WORK/output/results.json" "$WORK/input/tasks.json" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))
tasks = json.load(open(sys.argv[2]))
assert isinstance(results, list), "results.json must be a JSON array"
ids = {r["task_id"] for r in results}
want = {t["task_id"] for t in tasks}
missing = want - ids
assert not missing, f"missing task ids: {missing}"
empty = [r["task_id"] for r in results if not str(r.get("answer", "")).strip()]
print(f"OK: {len(results)} results, all task ids present, "
      f"{len(empty)} empty answers{': ' + str(empty) if empty else ''}")
PY

echo "[test] output directory: $WORK/output"
