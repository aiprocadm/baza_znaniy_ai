"""FastAPI middleware that records every /api/kb/* request to audit_log."""
from __future__ import annotations

import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.audit_db import persist_audit_event

LOGGER = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Log path + method + status + IP for paths matching path_prefix.

    Uses a session_factory to support tests with in-memory DB. In
    production, factory returns a session bound to the main DB engine.
    Errors during audit writes are logged but never block the request.
    """

    def __init__(self, app, *, session_factory: Callable, path_prefix: str = "/api/kb"):
        super().__init__(app)
        self._session_factory = session_factory
        self._path_prefix = path_prefix

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        if not request.url.path.startswith(self._path_prefix):
            return response

        try:
            ip = request.client.host if request.client else None
            session = self._session_factory()
            try:
                persist_audit_event(
                    session,
                    event="api_request",
                    ip=ip,
                    request_path=str(request.url.path),
                    request_method=request.method,
                    status_code=response.status_code,
                )
            finally:
                session.close()
        except Exception:
            LOGGER.exception("audit middleware failed to persist event")

        return response
