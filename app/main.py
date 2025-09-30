"""Entrypoint for the FastAPI application."""

from __future__ import annotations

from app.core.app import create_app
from app.llm import get_cached_provider

_provider = get_cached_provider()
_provider.ensure_model()

app = create_app(provider=_provider)

__all__ = ["app"]
