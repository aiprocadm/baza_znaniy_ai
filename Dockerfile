FROM python:3.12.4-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG HUGGINGFACE_HUB_TOKEN=""
ARG DOWNLOAD_MODEL=0
ARG LLM_MODEL_TARGET=default
ARG LLM_MODEL_OUTPUT=models/model.gguf
ARG INSTALL_DEV=0

ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

COPY requirements-runtime.txt requirements-llm.txt requirements-train.txt requirements-dev.txt pyproject.toml ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install -r requirements-runtime.txt -r requirements-llm.txt \
    && if [ "${INSTALL_DEV}" = "1" ]; then pip install -r requirements-dev.txt; fi

COPY app ./app
COPY data ./data
COPY docx ./docx
COPY scripts ./scripts
COPY models ./models
COPY backend ./backend
COPY README.md ./

RUN mkdir -p /app/models /app/var/data /app/var/data/logs /app/var/data/files/db \
    && touch /app/var/data/logs/app.log

RUN if [ "${DOWNLOAD_MODEL}" = "1" ]; then \
        python -m scripts.download_model --manifest ./models/model_manifest.json \
            --target "${LLM_MODEL_TARGET}" --output "${LLM_MODEL_OUTPUT}" \
            --allow-missing-hash --max-retries 5; \
    else \
        echo "Skipping GGUF download during build"; \
    fi

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
