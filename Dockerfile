FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG HUGGINGFACE_HUB_TOKEN=""
ARG DOWNLOAD_MODEL=1
ARG LLM_MODEL_TARGET=default
ARG LLM_MODEL_OUTPUT=models/model.gguf

ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

COPY requirements-runtime.txt requirements-llm.txt requirements-train.txt requirements-dev.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install --no-cache-dir -r requirements-runtime.txt -r requirements-llm.txt

COPY app ./app
COPY data ./data
COPY docx ./docx
COPY scripts ./scripts
COPY models ./models
COPY pyproject.toml README.md ./

RUN mkdir -p models var/data

RUN python - <<'PY'
import os
from pathlib import Path

def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()

def main() -> None:
    data_dir = _resolve(os.environ.get("DATA_DIR", "/app/var/data"))
    log_dir = _resolve(os.environ.get("LOG_DIR", str(data_dir / "logs")))
    log_file = _resolve(os.environ.get("APP_LOG_FILE", str(log_dir / "app.log")))

    for directory in (data_dir, data_dir / "files", data_dir / "files" / "db", log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

if __name__ == "__main__":
    main()
PY

RUN if [ "${DOWNLOAD_MODEL}" = "1" ]; then \
        python -m scripts.download_model --manifest ./models/model_manifest.json \
            --target "${LLM_MODEL_TARGET}" --output "${LLM_MODEL_OUTPUT}" \
            --allow-missing-hash --max-retries 5; \
    else \
        echo "Skipping GGUF download during build"; \
    fi

EXPOSE 8000

ENV BACKEND_ENTRYPOINT=app.api.main:app

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
