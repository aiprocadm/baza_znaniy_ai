#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="http://127.0.0.1:8000/health"
MAX_RETRIES=5
INTERVAL=5

for attempt in $(seq 1 "$MAX_RETRIES"); do
  if curl -fsS "$SERVICE_URL" >/dev/null; then
    echo "Service is healthy"
    exit 0
  fi
  echo "Health check failed (attempt $attempt/$MAX_RETRIES). Retrying in $INTERVAL seconds..."
  sleep "$INTERVAL"
done

echo "Service did not respond after $MAX_RETRIES attempts"
exit 1
