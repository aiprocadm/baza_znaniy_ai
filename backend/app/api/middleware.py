from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.app.api.constants import TRACE_HEADER


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Attach a trace identifier to every incoming request."""

    def __init__(self, app, header_name: str = TRACE_HEADER) -> None:  # type: ignore[override]
        super().__init__(app)
        self._header_name = header_name

    async def dispatch(  # type: ignore[override]
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get(self._header_name, uuid.uuid4().hex)
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers.setdefault(self._header_name, trace_id)
        return response


class JSONBodyLimitMiddleware(BaseHTTPMiddleware):
    """Limit the maximum size of JSON request payloads."""

    def __init__(self, app, max_body_size: int) -> None:  # type: ignore[override]
        super().__init__(app)
        self._max_body_size = max_body_size

    async def dispatch(  # type: ignore[override]
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type == "application/json":
            content_length_header = request.headers.get("content-length")
            if content_length_header is not None:
                try:
                    content_length = int(content_length_header)
                except ValueError:
                    content_length = None
                else:
                    if content_length > self._max_body_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="JSON payload exceeds allowed size",
                        )

            body = await request.body()
            if len(body) > self._max_body_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="JSON payload exceeds allowed size",
                )

        return await call_next(request)


__all__ = ["TraceIdMiddleware", "JSONBodyLimitMiddleware"]
