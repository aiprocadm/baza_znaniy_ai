FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/srv/projects/kb/app \
    DATA_ROOT=/srv/projects/kb/data \
    FILES_ROOT=/srv/projects/kb/data/files \
    OLLAMA_MODELS=/srv/projects/kb/data/storage/ollama \
    OLLAMA_HOST=0.0.0.0 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    QDRANT_URL=http://127.0.0.1:6333

ARG QDRANT_VERSION=1.9.1
ARG OLLAMA_VERSION=0.3.12

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    nginx \
    supervisor \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash knowlab

RUN mkdir -p "${APP_HOME}" \
 && mkdir -p "${DATA_ROOT}/files" \
 && mkdir -p "${DATA_ROOT}/storage/ollama" \
 && mkdir -p "${DATA_ROOT}/storage/qdrant" \
 && mkdir -p "${DATA_ROOT}/logs/api" \
 && mkdir -p "${DATA_ROOT}/logs/nginx" \
 && mkdir -p "${DATA_ROOT}/logs/ollama" \
 && mkdir -p "${DATA_ROOT}/logs/qdrant" \
 && mkdir -p "${DATA_ROOT}/logs/supervisor" \
 && mkdir -p "${DATA_ROOT}/www" \
 && mkdir -p "${DATA_ROOT}/ssl" \
 && mkdir -p /var/log/supervisor \
 && chown -R knowlab:knowlab /srv/projects/kb

WORKDIR ${APP_HOME}

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN set -eux; \
    curl -L "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/qdrant.tar.gz; \
    tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin; \
    rm /tmp/qdrant.tar.gz; \
    chmod +x /usr/local/bin/qdrant

RUN set -eux; \
    curl -L "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tgz" -o /tmp/ollama.tgz; \
    tar -xzf /tmp/ollama.tgz -C /usr/local; \
    rm /tmp/ollama.tgz; \
    chmod +x /usr/local/bin/ollama

COPY data/nginx.conf /etc/nginx/nginx.conf
COPY app/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN chown -R knowlab:knowlab /srv/projects/kb

EXPOSE 80 443 8080 11434 6333

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
