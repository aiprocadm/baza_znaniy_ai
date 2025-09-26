FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash knowlab

WORKDIR /opt/knowlab/app

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /opt/knowlab/data/files \
 && chown -R knowlab:knowlab /opt/knowlab

USER knowlab

ENV OLLAMA_BASE_URL="http://ollama:11434"

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
