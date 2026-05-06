from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes import documents, knowledge_base, packs
from backend.app.api.v1 import rag

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents.router)
api_router.include_router(packs.router)
api_router.include_router(knowledge_base.router)
api_router.include_router(rag.router)

__all__ = ["api_router"]
