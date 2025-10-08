from __future__ import annotations

from app._module_reset import ensure_core_modules

ensure_core_modules()

from app.core.app import create_app

app = create_app()

__all__ = ["app"]
