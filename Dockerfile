# TokenCascade v5 — Track 1 submission image (linux/amd64, weights bundled).
# Exact dependency pins = the combination proven to run (design law L4).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && curl -fL --retry 3 -o /models/model.gguf "$MODEL_URL"

WORKDIR /app
COPY main.py .

ENV PYTHONUNBUFFERED=1
ENV LOCAL_CATEGORIES="factual,sentiment,ner,summarization,math,code_debug"
CMD ["python", "main.py"]
