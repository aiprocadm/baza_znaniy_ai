from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.constants import ALLOWED_UPLOAD_MIME_TYPES, MAX_UPLOAD_SIZE_BYTES
from backend.app.api.deps import authenticate_credentials, get_db, get_tenant_context, require_platform_admin, require_tenant_admin
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
from backend.app.models import BillingEvent, Plan, Subscription, UsageCounter
from backend.app.services.kb_runtime import runtime_store

router = APIRouter(tags=["knowledge-base"])


@router.get("/status", response_model=SystemStatusResponse)
def get_status() -> SystemStatusResponse:
    return runtime_store.get_status()


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> SearchResponse:
    _, tenant_slug = tenant_ctx
    return runtime_store.search(tenant_slug, payload)


@router.get("/activities", response_model=list[ActivityItem])
def list_activities() -> list[ActivityItem]:
    return runtime_store.list_activities()


@router.get("/files", response_model=list[FileMeta])
def list_files() -> list[FileMeta]:
    return runtime_store.list_files()


@router.post("/upload", response_model=FileMeta)
async def upload_file(file: UploadFile = File(...), tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> FileMeta:
    mime_type = file.content_type or "application/octet-stream"
    filename = Path(file.filename or "untitled").name
    chunk_size = 1024 * 1024
    total_size = 0

    if mime_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {mime_type}",
        )

    chunks: list[bytes] = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Max size is {MAX_UPLOAD_SIZE_BYTES} bytes",
            )

    if total_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    _, tenant_slug = tenant_ctx
    return runtime_store.add_file(tenant_slug=tenant_slug, name=filename, size=total_size, mime_type=mime_type, content=b"".join(chunks))


@router.get("/admin/users", response_model=list[UserResponse], dependencies=[Depends(require_tenant_admin)])
def get_users() -> list[UserResponse]:
    return runtime_store.list_users()


@router.post("/admin/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_tenant_admin)])
def create_user(payload: UserPayload) -> UserResponse:
    return runtime_store.create_user(payload)


@router.patch("/admin/users/{user_id}", response_model=UserResponse, dependencies=[Depends(require_tenant_admin)])
def patch_user(user_id: str, payload: UserUpdatePayload) -> UserResponse:
    user = runtime_store.update_user(user_id, payload)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_tenant_admin)])
def remove_user(user_id: str) -> None:
    if not runtime_store.delete_user(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.get("/admin/api-keys", response_model=list[ApiKey], dependencies=[Depends(require_tenant_admin)])
def get_api_keys() -> list[ApiKey]:
    return runtime_store.list_api_keys()


@router.post("/admin/api-keys/{key_id}/rotate", dependencies=[Depends(require_tenant_admin)])
def rotate_api_key(key_id: str) -> dict[str, str]:
    return {"secret": f"{key_id}_{uuid4().hex}"}


@router.get("/admin/settings", response_model=SystemSettings, dependencies=[Depends(require_platform_admin)])
def get_settings() -> SystemSettings:
    return runtime_store.settings


@router.put("/admin/settings", response_model=SystemSettings, dependencies=[Depends(require_platform_admin)])
def put_settings(payload: SystemSettings) -> SystemSettings:
    runtime_store.settings = payload
    return runtime_store.settings


@router.get("/auth/session", response_model=SessionResponse)
def get_session(identity=Depends(get_tenant_context)) -> SessionResponse:
    _tenant_id, _tenant_slug = identity
    return runtime_store.get_session()


@router.post("/auth/login", response_model=SessionResponse)
def login(payload: LoginPayload) -> SessionResponse:
    identity = authenticate_credentials(payload.email, payload.password)
    if identity is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    session = runtime_store.get_session()
    if session.user_id != identity.user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return session


@router.post("/auth/refresh", response_model=RefreshResponse)
def refresh_token() -> RefreshResponse:
    return RefreshResponse(token=f"kb_refresh_{int(datetime.now(timezone.utc).timestamp())}")


@router.get("/admin/usage", dependencies=[Depends(require_tenant_admin)])
def get_usage(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(select(UsageCounter).order_by(UsageCounter.updated_at.desc()).limit(200)).scalars().all()
    return [
        {
            "tenant_id": r.tenant_id,
            "period_start": r.period_start.isoformat(),
            "period_end": r.period_end.isoformat(),
            "storage_bytes": r.storage_bytes,
            "documents_count": r.documents_count,
            "search_requests": r.search_requests,
            "llm_requests": r.llm_requests,
        }
        for r in rows
    ]


@router.get("/admin/plan", dependencies=[Depends(require_tenant_admin)])
def get_current_plan(db: Session = Depends(get_db)) -> dict:
    subscription = db.execute(select(Subscription).where(Subscription.status == "active").order_by(Subscription.id.desc()).limit(1)).scalars().first()
    if subscription is None:
        return {"subscription": None}
    plan = db.execute(select(Plan).where(Plan.code == subscription.plan_code)).scalars().first()
    usage_events = db.execute(select(BillingEvent).where(BillingEvent.tenant_id == subscription.tenant_id).order_by(BillingEvent.created_at.desc()).limit(10)).scalars().all()
    return {
        "subscription": {"tenant_id": subscription.tenant_id, "plan_code": subscription.plan_code, "status": subscription.status},
        "plan": None if plan is None else {
            "name": plan.name,
            "max_storage_bytes": plan.max_storage_bytes,
            "max_documents": plan.max_documents,
            "max_search_requests": plan.max_search_requests,
            "max_llm_requests": plan.max_llm_requests,
        },
        "recent_events": [
            {"type": e.event_type, "payload": e.payload, "created_at": e.created_at.isoformat()} for e in usage_events
        ],
    }


__all__ = ["router"]
