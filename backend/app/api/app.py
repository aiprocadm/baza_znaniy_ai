from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.error_handlers import install_error_handlers
from backend.app.api.middleware import JSONBodyLimitMiddleware, TraceIdMiddleware
from backend.app.api.v1.router import api_router
from backend.app.core.config import get_settings


def create_api_app() -> FastAPI:
    """Construct the FastAPI application with common middleware and routes."""

    settings = get_settings()
    app = FastAPI(title="Document Generator", max_request_body_size=settings.upload_max_bytes)

    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(JSONBodyLimitMiddleware, max_body_size=settings.json_max_bytes)

    install_error_handlers(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    app.include_router(api_router)

    return app


__all__ = ["create_api_app"]
