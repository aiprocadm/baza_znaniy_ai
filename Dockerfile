FROM python:3.11-slim

        codex/split-existing-service-into-containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/projects/kb

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ /srv/projects/kb/app/

RUN useradd -m -s /bin/bash kb \
 && chown -R kb:kb /srv/projects/kb

USER kb

ENV OLLAMA_BASE_URL="http://ollama:11434"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash knowlab
WORKDIR /opt/knowlab/app

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /opt/knowlab/data/files \
 && chown -R knowlab:knowlab /opt/knowlab

USER knowlab

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
        main
