#!/usr/bin/env bash
set -euo pipefail
U=admin
P=admin
H=kb.local
if ! curl -ks -u "${U}:${P}" https://${H}/health >/dev/null ; then
  docker restart kb_web || true
  echo "$(date -Iseconds) RESTART kb_web" >> /var/log/kb_watchdog.log
fi
