"""Versioned API routers."""

from fastapi import APIRouter

from . import chat, delete, files, ingest, search, upload

router = APIRouter()
router.include_router(upload.router)
router.include_router(ingest.router)
router.include_router(search.router)
router.include_router(chat.router)
router.include_router(files.router)
router.include_router(delete.router)

__all__ = ["router"]
