"""Aggregate and expose FastAPI routers used by the service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter

from app.api import routes as root_routes
from app.api.kb_mvp import router as kb_mvp_router
from app.api.routes_lora import router as lora_admin_router


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
api_router.include_router(lora_admin_router)


def _include_with_prefix(target: APIRouter, source: APIRouter, prefix: str) -> None:
    """Include *source* under *prefix*, tolerating stubbed APIRouter classes."""

    try:  # pragma: no cover - exercised in real FastAPI runtime
        target.include_router(source, prefix=prefix)
        return
    except TypeError:  # pragma: no cover - fallback for test stubs without prefix support
        pass

    routes = list(getattr(source, "_routes", []))
    route_type = type(routes[0]) if routes else None
    for route in routes:
        prefixed = f"{prefix}{route.path}".replace("//", "/")
        if route_type is not None:
            target._routes.append(  # type: ignore[attr-defined]  # _routes exists only on the test-stub APIRouter (tests/stubs/fastapi); real FastAPI uses include_router above
                route_type(route.method, prefixed, route.handler, route.status_code)
            )
        else:  # pragma: no cover - defensive
            target._routes.append(route)  # type: ignore[attr-defined]  # _routes exists only on the test-stub APIRouter (tests/stubs/fastapi); real FastAPI uses include_router above


_include_with_prefix(api_router, kb_mvp_router, "/api/kb")
_include_with_prefix(api_router, v1_router, "/api/v1")


__all__ = ["api_router"]
