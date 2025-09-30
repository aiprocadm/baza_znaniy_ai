"""CORS middleware placeholder for the compatibility layer."""

from __future__ import annotations

from typing import Any


class CORSMiddleware:
    """Store configuration for CORS without performing any processing."""

    def __init__(self, app: Any | None = None, **options: Any) -> None:
        self.app = app
        self.options = options
