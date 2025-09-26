#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="/data/storage"
DESTINATION="/data/storage/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TARGET="$DESTINATION/kb_backup_$TIMESTAMP.tar.gz"

mkdir -p "$DESTINATION"

tar -czf "$TARGET" -C "$BACKUP_ROOT" .

echo "Backup created at $TARGET"
