from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes import documents, packs

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents.router)
api_router.include_router(packs.router)

__all__ = ["api_router"]
