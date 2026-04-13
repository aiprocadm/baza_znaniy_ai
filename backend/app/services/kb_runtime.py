from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4

from backend.app.schemas.knowledge_base import (
    ActivityItem,
    ApiKey,
    FileMeta,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ServiceHealth,
    SessionResponse,
    SystemSettings,
    SystemStats,
    SystemStatusResponse,
    UserPayload,
    UserResponse,
    UserUpdatePayload,
)


@dataclass
class KBRuntimeStore:
    lock: Lock = field(default_factory=Lock)
    files: list[FileMeta] = field(default_factory=list)
    activities: list[ActivityItem] = field(default_factory=list)
    users: dict[str, UserResponse] = field(default_factory=dict)
    settings: SystemSettings = field(
        default_factory=lambda: SystemSettings(
            qdrant_url="http://localhost:6333",
            llm_model="meta-llama/Meta-Llama-3-8B-Instruct",
            ingestion_parallelism=4,
            allow_guest_access=False,
        )
    )

    def __post_init__(self) -> None:
        admin = UserResponse(
            id="u_admin",
            name="KB Administrator",
            email="admin@kb.ai",
            roles=["admin"],
            status="active",
        )
        self.users[admin.id] = admin

    def get_status(self) -> SystemStatusResponse:
        with self.lock:
            errors = sum(1 for file in self.files if file.status == "error")
            ingestions = sum(1 for file in self.files if file.status == "processing")
            return SystemStatusResponse(
                services=[
                    ServiceHealth(name="api", status="healthy", latency_ms=24),
                    ServiceHealth(name="vector_store", status="healthy", latency_ms=38),
                    ServiceHealth(name="llm", status="healthy", latency_ms=97),
                ],
                stats=SystemStats(documents=len(self.files), ingestions=ingestions, errors=errors),
            )

    def add_file(self, *, name: str, size: int, mime_type: str) -> FileMeta:
        now = datetime.now(timezone.utc)
        file = FileMeta(
            id=f"f_{uuid4().hex[:10]}",
            name=name,
            size=size,
            mime_type=mime_type,
            status="indexed",
            created_at=now,
        )
        with self.lock:
            self.files.insert(0, file)
            self.activities.insert(
                0,
                ActivityItem(
                    id=f"a_{uuid4().hex[:10]}",
                    type="upload",
                    title=f"Uploaded {name}",
                    description="Document uploaded and queued for indexing.",
                    created_at=now,
                ),
            )
        return file

    def list_files(self) -> list[FileMeta]:
        with self.lock:
            return list(self.files)

    def list_activities(self) -> list[ActivityItem]:
        with self.lock:
            if not self.activities:
                now = datetime.now(timezone.utc)
                return [
                    ActivityItem(
                        id="a_bootstrap",
                        type="ingest",
                        title="System initialized",
                        description="Knowledge base runtime is ready.",
                        created_at=now,
                    )
                ]
            return list(self.activities[:30])

    def search(self, query: SearchRequest) -> SearchResponse:
        with self.lock:
            source = self.files or [
                FileMeta(
                    id="f_demo",
                    name="Knowledge Base Quickstart",
                    size=2048,
                    mime_type="text/markdown",
                    status="indexed",
                    created_at=datetime.now(timezone.utc),
                )
            ]

        results = [
            SearchResultItem(
                id=f"r_{item.id}_{index}",
                title=item.name,
                snippet=f"{query.query}: relevant section #{index + 1} from {item.name}",
                score=max(0.5, 0.96 - (index * 0.08)),
                source=item.id,
                updated_at=item.created_at,
            )
            for index, item in enumerate(source[: query.top_k])
        ]

        with self.lock:
            self.activities.insert(
                0,
                ActivityItem(
                    id=f"a_{uuid4().hex[:10]}",
                    type="search",
                    title="Search executed",
                    description=f"Query '{query.query}' returned {len(results)} results.",
                    created_at=datetime.now(timezone.utc),
                ),
            )
        return SearchResponse(results=results, total=len(results))

    def list_users(self) -> list[UserResponse]:
        with self.lock:
            return list(self.users.values())

    def create_user(self, payload: UserPayload) -> UserResponse:
        user = UserResponse(
            id=f"u_{uuid4().hex[:10]}",
            name=payload.name,
            email=payload.email,
            roles=payload.roles,
            status="invited",
        )
        with self.lock:
            self.users[user.id] = user
        return user

    def update_user(self, user_id: str, payload: UserUpdatePayload) -> UserResponse | None:
        with self.lock:
            current = self.users.get(user_id)
            if current is None:
                return None
            patched = current.model_copy(update=payload.model_dump(exclude_none=True))
            self.users[user_id] = patched
            return patched

    def delete_user(self, user_id: str) -> bool:
        with self.lock:
            return self.users.pop(user_id, None) is not None

    def list_api_keys(self) -> list[ApiKey]:
        now = datetime.now(timezone.utc)
        return [
            ApiKey(id="k_default", name="Default Integration", prefix="kb_live", created_at=now - timedelta(days=3))
        ]

    def get_session(self) -> SessionResponse:
        return SessionResponse(
            user_id="u_admin",
            email="admin@kb.ai",
            name="KB Administrator",
            roles=["admin"],
            token_expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
        )


runtime_store = KBRuntimeStore()

__all__ = ["runtime_store", "KBRuntimeStore"]
