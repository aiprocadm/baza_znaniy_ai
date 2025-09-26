#!/usr/bin/env bash
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
DST="/srv/backups/${TS}"
mkdir -p "$DST"
tar -czf "${DST}/kb.tar.gz" \
  -C /srv/projects/kb \
  .env data/db data/storage data/www data/nginx.conf
echo "TS=${TS} EXIT_CODE=$?" >> /var/log/kb_backup.log
