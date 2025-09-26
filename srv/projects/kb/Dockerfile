FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential nginx supervisor curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data /data
COPY .env ./

ENV DATA_DIR=/data/storage

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app /data

USER appuser

CMD ["/usr/bin/supervisord", "-c", "/data/supervisor/supervisord.conf"]
