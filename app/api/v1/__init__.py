"""Versioned API routers."""

from fastapi import APIRouter

from . import admin, auth, chat, delete, files, ingest, lora, search, tenants, upload, users

router = APIRouter()
router.include_router(admin.router)
router.include_router(auth.router)
router.include_router(users.router)
router.include_router(tenants.router)
router.include_router(upload.router)
router.include_router(ingest.router)
router.include_router(lora.router)
router.include_router(search.router)
router.include_router(chat.router)
router.include_router(files.router)
router.include_router(delete.router)

__all__ = ["router"]
