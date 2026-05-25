#!/usr/bin/env bash
# install.sh — one-click installer for KB.AI on Linux/macOS.
#
# Usage:
#   bash install.sh                    # full install
#   bash install.sh --dry-run          # print what would happen, no changes
#
# Requirements:
#   - Python 3.12+
#   - pip
#   - Internet access for pip install
#
# After install, start the server with:
#   python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# Or, for the MVP-only stack:
#   python -m uvicorn scripts.dev_server_mvp:app --port 8001

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

say() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] would: $*"
    else
        echo "[install] $*"
    fi
}

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

# 1. Python version check
say "Checking Python 3.12+"
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not on PATH. Install Python 3.12+ first." >&2
    exit 1
fi

PY_OK=$(python3 -c 'import sys; print("1" if sys.version_info >= (3, 12) else "0")')
if [[ "$PY_OK" != "1" ]]; then
    echo "ERROR: Python 3.12+ required (got $(python3 --version))." >&2
    exit 1
fi

# 2. Install runtime + MVP deps
say "Installing dependencies (this may take a few minutes)"
run python3 -m pip install --upgrade pip
run python3 -m pip install -e .

# 3. Copy .env.example → .env if not present
if [[ ! -f .env ]]; then
    say "Copying .env.example → .env"
    run cp .env.example .env
    echo "  Edit .env to add your LLM API keys (DEEPSEEK_API_KEY etc.)"
else
    say ".env already exists — leaving untouched"
fi

# 4. Create var/data directory
say "Ensuring var/data/ exists"
run mkdir -p var/data

# 5. Final message
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Install complete."
echo "  Start the server:"
echo "    python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000"
echo "  Or MVP-only:"
echo "    python3 -m uvicorn scripts.dev_server_mvp:app --port 8001"
echo "  Then open http://localhost:8000/"
echo "════════════════════════════════════════════════════════════════"
