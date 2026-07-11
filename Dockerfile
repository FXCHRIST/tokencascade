# TokenCascade v2 — Track 1 submission image.
# Rules honored: weights BUNDLED in the image (no runtime downloads — network
# use outside FIREWORKS_BASE_URL risks disqualification and busts the 60s
# startup), linux/amd64 manifest required, <10GB compressed.
#
# Build (native amd64 or CI):
#   docker build -t ghcr.io/YOUR_USER/tokencascade:latest .
# Build on Apple Silicon:
#   docker buildx build --platform linux/amd64 -t ghcr.io/YOUR_USER/tokencascade:latest --push .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Local model — swap via --build-arg MODEL_URL=... (one flag = new model).
# Default: Qwen2.5-3B-Instruct Q4_K_M (~1.9GB, fits 4GB RAM with room for
# the agent). [verify: exact filename on the HF repo before building;
# Apache-2.0 fallback if license matters: Qwen2.5-1.5B-Instruct-GGUF]
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && curl -fL --retry 3 -o /models/model.gguf "$MODEL_URL"

WORKDIR /app
COPY main.py .

ENV PYTHONUNBUFFERED=1
ENV LOCAL_CATEGORIES="factual,sentiment,ner,summarization,math,logic,code_debug"
CMD ["python", "main.py"]
