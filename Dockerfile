# ============================================================================
# TokenCascade v6 — fully local Track-1 agent. Zero Fireworks tokens.
# Target: linux/amd64 (judging harness requirement).
#
# CRITICAL BUILD PROPERTY: llama-cpp-python is compiled from source with
# GGML_NATIVE=OFF / LLAMA_NATIVE=OFF. Without these flags the compiler
# targets the BUILD machine's CPU (-march=native); if the judging CPU lacks
# any of those instructions (e.g. AVX-512), the container dies with an
# illegal-instruction crash that no local test would ever reproduce.
# Portable baseline instructions only.
# ============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Toolchain -> compile pinned llama-cpp-python portably -> purge toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake gcc g++ libgomp1 \
    && CMAKE_ARGS="-DGGML_NATIVE=OFF -DLLAMA_NATIVE=OFF -DGGML_BLAS=OFF -DLLAMA_BLAS=OFF -DGGML_CUDA=OFF -DLLAMA_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_METAL=OFF" \
       pip install --no-cache-dir --no-binary llama-cpp-python \
       "llama-cpp-python==0.3.33" \
    && apt-get purge -y build-essential cmake gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Model weights are baked in at build time (no runtime downloads — TIMEOUT
# insurance). The CI workflow downloads and SHA256-verifies model.gguf into
# the build context before this step.
COPY models/model.gguf /models/model.gguf

# Make an image without weights impossible to push: fail the build if the
# model is missing or suspiciously small.
RUN test -f /models/model.gguf \
    && [ "$(stat -c%s /models/model.gguf)" -gt 1000000000 ] \
    || { echo "ERROR: /models/model.gguf missing or too small"; exit 1; }

COPY main.py /app/main.py

# Entrypoint runs automatically; reads /input/tasks.json, writes
# /output/results.json, exits 0.
ENTRYPOINT ["python", "/app/main.py"]
