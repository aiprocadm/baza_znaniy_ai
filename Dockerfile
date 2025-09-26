FROM debian:12-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    nginx openssl supervisor curl ca-certificates apache2-utils procps \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash kb
WORKDIR /srv/projects/kb
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY app/ /srv/projects/kb/app/
COPY data/nginx.conf /etc/nginx/nginx.conf
COPY app/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN mkdir -p /srv/projects/kb/data/{www,ssl,db,storage/docs,storage/qdrant,storage/ollama} \
 && mkdir -p /run/nginx /var/log/nginx && chown -R kb:kb /srv/projects/kb
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
 && chmod +x /usr/local/bin/ollama
USER kb
EXPOSE 80 443
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
