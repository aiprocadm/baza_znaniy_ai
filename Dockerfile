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
# By default, the image bundles the LLM and e5-small embedder at build time for a self-contained,
# air-gap-ready image (~4 GB larger). Override with --build-arg BUNDLE_MODEL=false for lightweight/CI builds.
ARG BUNDLE_MODEL=true

ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

COPY requirements-runtime.txt requirements-llm.txt requirements-train.txt requirements-dev.txt pyproject.toml ./

# apt versions in -slim images change too often to pin reliably; security updates would break the build.
# build-essential is needed because llama-cpp-python (requirements-llm.txt) ships no
# cpython-3.12 manylinux wheel and has to be compiled from source. cmake itself
# arrives via the pip build env, but gcc/g++ do not. Trade-off: image grows by
# ~250 MB. A multi-stage build that drops these from the runtime layer is a
# follow-up if the image size becomes a concern.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip+setuptools+wheel are build tooling; runtime deps are pinned via requirements*.txt.
# hadolint ignore=DL3013
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

# Re-declare BUNDLE_MODEL so it's in scope for the following RUN block.
ARG BUNDLE_MODEL
RUN if [ "${BUNDLE_MODEL}" = "true" ]; then \
        python -m scripts.download_model \
            --manifest ./models/model_manifest.json \
            --target qwen2.5-3b-instruct \
            --output ./models/qwen2.5-3b-instruct-q4_k_m.gguf \
            --max-retries 5 && \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"; \
    else \
        echo "Skipping model bundle (BUNDLE_MODEL != true)"; \
    fi

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
