from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from time import perf_counter
import os
from uuid import uuid4

from sqlalchemy import select

from backend.app.db.session import session_scope
from backend.app.models.ingestion import IngestionEvent, IngestionJob
from backend.app.schemas.knowledge_base import (
    ActivityItem,
    ApiKey,
    FileMeta,
    SearchRequest,
    SearchResponse,
    ServiceHealth,
    SessionResponse,
    SystemSettings,
    SystemStats,
    SystemStatusResponse,
    UserPayload,
    UserResponse,
    UserUpdatePayload,
)
from backend.app.services.search_service import search_service


@dataclass
class KBRuntimeStore:
    lock: Lock = field(default_factory=Lock)
    users: dict[str, UserResponse] = field(default_factory=dict)
    settings: SystemSettings = field(default_factory=lambda: SystemSettings(qdrant_url="http://localhost:6333", llm_model="meta-llama/Meta-Llama-3-8B-Instruct", ingestion_parallelism=4, allow_guest_access=False))

    def __post_init__(self) -> None:
        admin = UserResponse(id="u_admin", name="KB Administrator", email="admin@kb.ai", roles=["platform-admin"], status="active")
        self.users[admin.id] = admin

    def get_status(self) -> SystemStatusResponse:
        db_health = self._db_health()
        vector_health = search_service.health()
        llm_health = self._llm_health()
        files = self.list_files()
        errors = sum(1 for file in files if file.status == "error")
        ingestions = sum(1 for file in files if file.status == "processing")
        return SystemStatusResponse(services=[db_health, vector_health, llm_health], stats=SystemStats(documents=len(files), ingestions=ingestions, errors=errors))

    def _db_health(self) -> ServiceHealth:
        started = perf_counter()
        try:
            with session_scope() as session:
                session.execute(select(1)).scalar_one()
            return ServiceHealth(name="db", status="healthy", latency_ms=int((perf_counter() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            return ServiceHealth(name="db", status="offline", latency_ms=int((perf_counter() - started) * 1000), last_error=str(exc))

    def _llm_health(self) -> ServiceHealth:
        provider = os.getenv("LLM_PROVIDER", "stub")
        return ServiceHealth(name="llm", status="healthy" if provider else "degraded", latency_ms=0)

    def add_file(self, *, tenant_slug: str, name: str, size: int, mime_type: str, content: bytes | None = None) -> FileMeta:
        now = datetime.now(timezone.utc)
        file_id = f"f_{uuid4().hex[:10]}"
        with session_scope() as session:
            session.add(IngestionJob(id=file_id, name=name, size=size, mime_type=mime_type, status="indexed", created_at=now))
            session.add(IngestionEvent(id=f"a_{uuid4().hex[:10]}", type="upload", title=f"Uploaded {name}", description="Document uploaded and queued for indexing.", created_at=now))
        if content is not None:
            search_service.index_document(file_id=file_id, file_name=name, text=content.decode("utf-8", errors="ignore"), owner=tenant_slug)
        return FileMeta(id=file_id, name=name, size=size, mime_type=mime_type, status="indexed", created_at=now)

    def list_files(self) -> list[FileMeta]:
        with session_scope() as session:
            rows = session.execute(select(IngestionJob).order_by(IngestionJob.created_at.desc())).scalars().all()
        return [FileMeta(id=row.id, name=row.name, size=row.size, mime_type=row.mime_type, status=row.status, created_at=row.created_at) for row in rows]

    def list_activities(self) -> list[ActivityItem]:
        with session_scope() as session:
            rows = session.execute(select(IngestionEvent).order_by(IngestionEvent.created_at.desc()).limit(30)).scalars().all()
        if not rows:
            now = datetime.now(timezone.utc)
            return [ActivityItem(id="a_bootstrap", type="ingest", title="System initialized", description="Knowledge base runtime is ready.", created_at=now)]
        return [ActivityItem(id=row.id, type=row.type, title=row.title, description=row.description, created_at=row.created_at) for row in rows]

    def search(self, tenant_slug: str, query: SearchRequest) -> SearchResponse:
        response = search_service.search(query, tenant_slug=tenant_slug)
        with session_scope() as session:
            session.add(IngestionEvent(id=f"a_{uuid4().hex[:10]}", type="search", title="Search executed", description=f"Query '{query.query}' returned {len(response.results)} results.", created_at=datetime.now(timezone.utc)))
        return response

    # user and auth methods unchanged
    def list_users(self) -> list[UserResponse]: return list(self.users.values())
    def create_user(self, payload: UserPayload) -> UserResponse:
        user = UserResponse(id=f"u_{uuid4().hex[:10]}", name=payload.name, email=payload.email, roles=payload.roles, status="invited")
        self.users[user.id] = user
        return user
    def update_user(self, user_id: str, payload: UserUpdatePayload) -> UserResponse | None:
        current = self.users.get(user_id)
        if current is None: return None
        patched = current.model_copy(update=payload.model_dump(exclude_none=True)); self.users[user_id] = patched; return patched
    def delete_user(self, user_id: str) -> bool: return self.users.pop(user_id, None) is not None
    def list_api_keys(self) -> list[ApiKey]:
        now = datetime.now(timezone.utc); return [ApiKey(id="k_default", name="Default Integration", prefix="kb_live", created_at=now - timedelta(days=3))]
    def get_session(self) -> SessionResponse:
        return SessionResponse(user_id="u_admin", email="admin@kb.ai", name="KB Administrator", tenant_id="t_platform", tenant_slug="platform", roles=["platform-admin"], token_expires_at=datetime.now(timezone.utc) + timedelta(hours=8))


runtime_store = KBRuntimeStore()
