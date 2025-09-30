FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY docx ./docx
COPY pyproject.toml README.md ./

RUN mkdir -p var/data

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

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
