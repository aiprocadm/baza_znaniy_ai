"""Versioned API routers."""

from fastapi import APIRouter

from . import admin, chat, delete, files, ingest, search, upload, lora

router = APIRouter()
router.include_router(admin.router)
router.include_router(upload.router)
router.include_router(ingest.router)
router.include_router(search.router)
router.include_router(chat.router)
router.include_router(files.router)
router.include_router(delete.router)
router.include_router(lora.router)

__all__ = ["router"]
