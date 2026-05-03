from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.app.api.constants import TRACE_HEADER
from backend.app.db.session import get_session_factory
from backend.app.models import BillingEvent, Tenant
from backend.app.services.policy_engine import PolicyEngine


class TraceIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = TRACE_HEADER) -> None:  # type: ignore[override]
        super().__init__(app)
        self._header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # type: ignore[override]
        trace_id = request.headers.get(self._header_name, uuid.uuid4().hex)
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers.setdefault(self._header_name, trace_id)
        return response


class JSONBodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int) -> None:  # type: ignore[override]
        super().__init__(app)
        self._max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # type: ignore[override]
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type == "application/json":
            body = await request.body()
            if len(body) > self._max_body_size:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="JSON payload exceeds allowed size")
        return await call_next(request)


class UsageAndQuotaMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:  # type: ignore[override]
        super().__init__(app)
        self._session_factory = get_session_factory()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # type: ignore[override]
        started = time.perf_counter()
        db = self._session_factory()
        tenant = db.execute(select(Tenant).where(Tenant.status == "active").order_by(Tenant.id.asc()).limit(1)).scalars().first()
        tenant_id = tenant.id if tenant else None

        if tenant_id and request.url.path in {"/api/v1/search", "/api/v1/upload", "/api/v1/documents/generate", "/api/v1/packs/run"}:
            engine = PolicyEngine(db)
            operation = "search" if "search" in request.url.path else "upload"
            if "generate" in request.url.path or "packs" in request.url.path:
                operation = "llm"
            try:
                engine.enforce(tenant_id, operation)
            except ValueError as exc:
                db.close()
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

        response = await call_next(request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        if tenant_id:
            event = BillingEvent(
                tenant_id=tenant_id,
                subscription_id=None,
                event_type="usage_request",
                amount_cents=0,
                currency="USD",
                payload={
                    "endpoint": request.url.path,
                    "method": request.method,
                    "latency_ms": latency_ms,
                    "tokens": int(request.headers.get("x-tokens", "0") or 0),
                    "chunks": int(request.headers.get("x-chunks", "0") or 0),
                },
                created_at=datetime.now(timezone.utc),
            )
            db.add(event)
            if request.url.path == "/api/v1/search":
                PolicyEngine(db).increment_counter(tenant_id, "search")
            if request.url.path in {"/api/v1/documents/generate", "/api/v1/packs/run"}:
                PolicyEngine(db).increment_counter(tenant_id, "llm")
            db.commit()
        db.close()
        return response


__all__ = ["TraceIdMiddleware", "JSONBodyLimitMiddleware", "UsageAndQuotaMiddleware"]
