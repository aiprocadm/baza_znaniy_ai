"""Multi-tenant mature API surface, mounted under ``/api/v1/*``.

TWIN-SURFACE NOTE — do not unify. This is the MULTI-TENANT surface (JWT/RBAC,
Postgres + Qdrant). Its deliberate twin is the SINGLE-TENANT MVP surface in
``app/api/kb_mvp/`` (/api/kb/*, one KB_API_KEY, SQLite). Merging them is a known
anti-pattern — see ``docs/architecture.md``.
"""

import importlib
import logging

from fastapi import APIRouter

LOGGER = logging.getLogger(__name__)

router = APIRouter()
for _module_name in (
    "admin",
    "admin_audit",
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
