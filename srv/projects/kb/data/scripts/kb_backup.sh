#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/srv/projects/kb"
BACKUP_ROOT="/srv/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$BACKUP_ROOT/$TIMESTAMP"
TARGET="$DEST_DIR/kb.tar.gz"
LOG_FILE="/var/log/kb_backup.log"

log_exit_code() {
  local exit_code=$?
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "$(date --iso-8601=seconds) EXIT_CODE=$exit_code" >> "$LOG_FILE"
}
trap log_exit_code EXIT

mkdir -p "$DEST_DIR"

_tar() {
  tar -czf "$TARGET" -C "$PROJECT_ROOT" \
    .env \
    data/db \
    data/storage \
    data/www \
    data/nginx.conf
}

_tar

echo "Backup created at $TARGET"
