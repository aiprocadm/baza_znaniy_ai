"""Aggregate and expose FastAPI routers used by the service."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import routes as root_routes
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
