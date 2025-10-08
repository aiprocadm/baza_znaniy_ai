"""Aggregate and expose FastAPI routers used by the service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter

from app.api import routes as root_routes


def _ensure_v1_package() -> None:
    module = sys.modules.get("app.api.v1")
    if module is None:
        return
    if getattr(module, "__file__", None):
        return

    package_init = Path(__file__).with_name("v1") / "__init__.py"
    spec = importlib.util.spec_from_file_location("app.api.v1", package_init)
    if spec is None or spec.loader is None:
        return

    real_module = importlib.util.module_from_spec(spec)
    sys.modules["app.api.v1"] = real_module
    spec.loader.exec_module(real_module)


_ensure_v1_package()

from app.api.v1 import router as v1_router

api_router = APIRouter()

api_router.include_router(root_routes.router)

try:  # pragma: no cover - exercised in real FastAPI runtime
    api_router.include_router(v1_router, prefix="/api/v1")
except TypeError:  # pragma: no cover - fallback for test stubs without prefix support
    routes = list(getattr(v1_router, "_routes", []))
    route_type = type(routes[0]) if routes else None
    for route in routes:
        prefixed = f"/api/v1{route.path}".replace("//", "/")
        if route_type is not None:
            api_router._routes.append(
                route_type(route.method, prefixed, route.handler, route.status_code)
            )
        else:  # pragma: no cover - defensive
            api_router._routes.append(route)


__all__ = ["api_router"]
