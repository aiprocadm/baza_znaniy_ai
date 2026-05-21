"""Opt-in API key authentication for ``/api/kb/*``.

Activated by setting ``KB_API_KEY`` in the environment. When empty,
auth is disabled and the MVP behaves as before (anyone on the network
can call any endpoint). When set, mutating endpoints require a header:

    X-API-Key: <secret>

Constant-time comparison via :func:`secrets.compare_digest` protects
against timing-side-channel attacks.

The dependency is intentionally a small FastAPI ``Depends``-callable
rather than a global middleware — this keeps two diagnostic routes
(``/health``, ``/providers``) reachable without a key, so healthchecks
(docker, nginx, k8s) and the frontend probe still work.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

LOGGER = logging.getLogger(__name__)


def _resolve_expected_key() -> Optional[str]:
    """Return the configured expected key or ``None`` when auth is disabled."""

    raw = os.environ.get("KB_API_KEY")
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Dependency raising 401 when ``KB_API_KEY`` is set and the header is wrong.

    Header is read case-insensitively by FastAPI/Starlette, so clients
    can send ``X-Api-Key``, ``x-api-key`` etc.
    """

    expected = _resolve_expected_key()
    if expected is None:
        return  # auth disabled

    provided = (x_api_key or "").strip()
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API_KEY_REQUIRED",
            headers={"WWW-Authenticate": 'ApiKey realm="kb-mvp"'},
        )
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="INVALID_API_KEY",
            headers={"WWW-Authenticate": 'ApiKey realm="kb-mvp"'},
        )


def auth_status() -> dict[str, object]:
    """Snapshot for ``GET /api/kb/health`` — never leaks the key itself."""

    expected = _resolve_expected_key()
    return {
        "enabled": expected is not None,
        "header": "X-API-Key",
        "key_env": "KB_API_KEY",
    }


__all__ = ["auth_status", "require_api_key"]
