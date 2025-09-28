FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

ARG QDRANT_VERSION=v1.10.3
ARG OLLAMA_VERSION=v0.12.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential nginx supervisor curl ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/qdrant.tar.gz \
    && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin \
    && rm /tmp/qdrant.tar.gz \
    && curl -L "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz" -o /tmp/ollama.tgz \
    && tar -xzf /tmp/ollama.tgz -C /usr/local \
    && rm /tmp/ollama.tgz

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data /data
COPY .env ./

ENV DATA_DIR=/data/storage

RUN useradd -ms /bin/bash appuser \
    && mkdir -p /data/qdrant /data/ollama /var/log/qdrant /var/log/ollama \
    && chown -R appuser:appuser /app /data /var/log/qdrant /var/log/ollama

USER appuser

CMD ["/usr/bin/supervisord", "-c", "/data/supervisor/supervisord.conf"]
