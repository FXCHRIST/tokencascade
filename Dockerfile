# TokenCascade — AMD Developer Hackathon ACT II, Track 1 (linux/amd64).
# Pure local zero-token agent: Qwen3-4B-Instruct-2507 Q4_K_M bundled at
# build time, run on CPU via llama.cpp. Exact dependency pins (design law
# L4) = the combination proven to run.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Model weights are baked into the image — the harness must never download
# anything at runtime (TIMEOUT guard). SHA256 is pinned so a truncated or
# corrupted download can never produce a "working" image that fails later.
ARG MODEL_URL="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
ARG MODEL_SHA256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"
RUN mkdir -p /models \
    && curl -fL --retry 3 --retry-delay 5 -o /models/model.gguf "$MODEL_URL" \
    && echo "$MODEL_SHA256  /models/model.gguf" | sha256sum -c -

WORKDIR /app
COPY main.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]
