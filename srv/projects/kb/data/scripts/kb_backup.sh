#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/srv/projects/kb"}
BACKUP_ROOT=${BACKUP_ROOT:-"/srv/backups"}
LOG_FILE=${LOG_FILE:-"/var/log/kb_backup.log"}
BASIC_USER=${BASIC_USER:-"admin"}
TIMESTAMP=$(date +%Y%m%d%H%M%S)
DEST_DIR="$BACKUP_ROOT/$TIMESTAMP"
ARCHIVE="$DEST_DIR/kb.tar.gz"
mkdir -p "$DEST_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

{
    echo "Starting backup from $PROJECT_ROOT"
    tar -czf "$ARCHIVE" -C "$PROJECT_ROOT" \
        .env \
        data/db \
        data/storage \
        data/www \
        data/nginx.conf \
        "data/ssl/$BASIC_USER"
    echo "Backup created at $ARCHIVE"
    echo "EXIT_CODE=0"
} >>"$LOG_FILE" 2>&1

echo "Backup created at $ARCHIVE"
