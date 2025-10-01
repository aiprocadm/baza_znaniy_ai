"""Uvicorn entrypoint that exposes the FastAPI application instance."""

from __future__ import annotations

from app.core.app import create_app

app = create_app()

__all__ = ["app"]
