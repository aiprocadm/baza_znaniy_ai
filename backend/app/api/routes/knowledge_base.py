from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.api.constants import ALLOWED_UPLOAD_MIME_TYPES, MAX_UPLOAD_SIZE_BYTES
from backend.app.schemas.knowledge_base import (
    ActivityItem,
    ApiKey,
    FileMeta,
    LoginPayload,
    RefreshResponse,
    SearchRequest,
    SearchResponse,
    SessionResponse,
    SystemSettings,
    SystemStatusResponse,
    UserPayload,
    UserResponse,
    UserUpdatePayload,
)
from backend.app.services.kb_runtime import runtime_store

router = APIRouter(tags=["knowledge-base"])


@router.get("/status", response_model=SystemStatusResponse)
def get_status() -> SystemStatusResponse:
    return runtime_store.get_status()


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest) -> SearchResponse:
    return runtime_store.search(payload)


@router.get("/activities", response_model=list[ActivityItem])
def list_activities() -> list[ActivityItem]:
    return runtime_store.list_activities()


@router.get("/files", response_model=list[FileMeta])
def list_files() -> list[FileMeta]:
    return runtime_store.list_files()


@router.post("/upload", response_model=FileMeta)
async def upload_file(file: UploadFile = File(...)) -> FileMeta:
    mime_type = file.content_type or "application/octet-stream"
    filename = Path(file.filename or "untitled").name
    chunk_size = 1024 * 1024
    total_size = 0

    if mime_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {mime_type}",
        )

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Max size is {MAX_UPLOAD_SIZE_BYTES} bytes",
            )

    if total_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    return runtime_store.add_file(name=filename, size=total_size, mime_type=mime_type)


@router.get("/admin/users", response_model=list[UserResponse])
def get_users() -> list[UserResponse]:
    return runtime_store.list_users()


@router.post("/admin/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserPayload) -> UserResponse:
    return runtime_store.create_user(payload)


@router.patch("/admin/users/{user_id}", response_model=UserResponse)
def patch_user(user_id: str, payload: UserUpdatePayload) -> UserResponse:
    user = runtime_store.update_user(user_id, payload)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_user(user_id: str) -> None:
    if not runtime_store.delete_user(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.get("/admin/api-keys", response_model=list[ApiKey])
def get_api_keys() -> list[ApiKey]:
    return runtime_store.list_api_keys()


@router.post("/admin/api-keys/{key_id}/rotate")
def rotate_api_key(key_id: str) -> dict[str, str]:
    return {"secret": f"{key_id}_{uuid4().hex}"}


@router.get("/admin/settings", response_model=SystemSettings)
def get_settings() -> SystemSettings:
    return runtime_store.settings


@router.put("/admin/settings", response_model=SystemSettings)
def put_settings(payload: SystemSettings) -> SystemSettings:
    runtime_store.settings = payload
    return runtime_store.settings


@router.get("/auth/session", response_model=SessionResponse)
def get_session() -> SessionResponse:
    return runtime_store.get_session()


@router.post("/auth/login", response_model=SessionResponse)
def login(payload: LoginPayload) -> SessionResponse:
    session = runtime_store.get_session()
    if payload.email.strip().lower() != session.email.lower():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return session


@router.post("/auth/refresh", response_model=RefreshResponse)
def refresh_token() -> RefreshResponse:
    return RefreshResponse(token=f"kb_refresh_{int(datetime.now(timezone.utc).timestamp())}")


__all__ = ["router"]
