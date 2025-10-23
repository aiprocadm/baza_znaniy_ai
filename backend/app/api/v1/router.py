from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes import documents

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents.router)

__all__ = ["api_router"]
