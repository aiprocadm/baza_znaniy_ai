"""Common vector store interfaces and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from app.core.config import Settings, get_settings


@runtime_checkable
class VectorStore(Protocol):
    """Protocol describing the interface exposed by vector stores."""

    def ensure_ready(self) -> None:  # pragma: no cover - protocol definition
        """Ensure that the backing index exists and is configured."""

    def upsert(
        self, chunks: Iterable[dict[str, object]]
    ) -> None:  # pragma: no cover - protocol definition
        """Persist or update a batch of document chunks."""

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters,
    ) -> list[dict[str, object]]:  # pragma: no cover - protocol definition
        """Return the top matching chunks for the supplied query."""


def _load_backends():
    from .faiss import FaissVectorStore
    from .qdrant import QdrantVectorStore

    return {
        "faiss": FaissVectorStore,
        "qdrant": QdrantVectorStore,
    }


def _build_backend(settings: Settings) -> VectorStore:
    """Instantiate the backend requested by :class:`Settings`."""

    backend = (settings.vector_backend or "").lower()
    backends = _load_backends()
    factory = backends.get(backend)
    if factory is None:
        raise ValueError(f"Unsupported vector backend: {settings.vector_backend!r}")
    return factory(settings=settings)


_DEFAULT_STORE: VectorStore | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the cached vector store instance for the active settings."""

    global _DEFAULT_STORE
    if settings is None:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = _build_backend(get_settings())
        return _DEFAULT_STORE
    return _build_backend(settings)


def _clear_cache() -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = None


setattr(get_vector_store, "cache_clear", _clear_cache)


__all__ = ["SearchFilters", "VectorStore", "get_vector_store", "_build_backend"]


@dataclass(frozen=True)
class SearchFilters:
    """Canonical tenant-scoped filter contract used by API and vector stores."""

    tenant_id: str
    owner: str | None = None
    tags: tuple[str, ...] = ()
    act_type: str | None = None
    issuer: str | None = None
    reg_number: str | None = None
    is_active: bool | None = None
    revision_mode: str = "current"

    @classmethod
    def from_input(
        cls,
        *,
        tenant_id: str,
        owner: str | None = None,
        tags: list[str] | None = None,
        act_type: str | None = None,
        issuer: str | None = None,
        reg_number: str | None = None,
        is_active: bool | None = None,
        revision_mode: str = "current",
    ) -> "SearchFilters":
        tenant = tenant_id.strip()
        if not tenant:
            raise ValueError("tenant_id is required for search")
        normalized_owner = owner.strip() if owner and owner.strip() else None
        normalized_tags = tuple(tag.strip() for tag in (tags or []) if tag and tag.strip())
        return cls(
            tenant_id=tenant,
            owner=normalized_owner,
            tags=normalized_tags,
            act_type=act_type.strip() if act_type and act_type.strip() else None,
            issuer=issuer.strip() if issuer and issuer.strip() else None,
            reg_number=reg_number.strip() if reg_number and reg_number.strip() else None,
            is_active=is_active,
            revision_mode=revision_mode,
        )
