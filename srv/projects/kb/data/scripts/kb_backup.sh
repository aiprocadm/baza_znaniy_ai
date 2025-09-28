#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/srv/projects/kb"
BACKUP_ROOT="/srv/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$BACKUP_ROOT/$TIMESTAMP"
TARGET="$DEST_DIR/kb.tar.gz"
LOG_FILE="/var/log/kb_backup.log"
APP_PORT="${APP_PORT:-8000}"
BASIC_USER="${BASIC_USER:-admin}"

log_exit_code() {
  local exit_code=$?
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "$(date --iso-8601=seconds) EXIT_CODE=$exit_code" >> "$LOG_FILE"
}
trap log_exit_code EXIT

mkdir -p "$DEST_DIR"

INCLUDE_PATHS=(
  .env
  data/db
  data/storage
  data/www
  data/nginx.conf
)

HTPASSWD_PATH="data/ssl/${BASIC_USER}"
if [ -f "$PROJECT_ROOT/$HTPASSWD_PATH" ]; then
  INCLUDE_PATHS+=("$HTPASSWD_PATH")
fi

tar -czf "$TARGET" -C "$PROJECT_ROOT" "${INCLUDE_PATHS[@]}"

echo "Backup created at $TARGET (service port ${APP_PORT})"
