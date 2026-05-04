"""Versioned API routers."""

import importlib
import logging

from fastapi import APIRouter

LOGGER = logging.getLogger(__name__)

router = APIRouter()
for _module_name in (
    "admin",
    "auth",
    "users",
    "tenants",
    "upload",
    "ingest",
    "lora",
    "ops",
    "search",
    "chat",
    "files",
    "delete",
    "documents",
):
    try:
        module = importlib.import_module(f"{__name__}.{_module_name}")
    except Exception as exc:  # pragma: no cover - defensive for constrained test stubs
        LOGGER.warning("Skipping router import for %s: %s", _module_name, exc)
        continue
    router.include_router(module.router)

__all__ = ["router"]
