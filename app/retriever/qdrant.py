"""Qdrant-backed vector store implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, cast

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# qdrant-client renamed ``PayloadSchemaType.BOOL`` to ``BOOLEAN`` around
# v1.13. Resolve once at import time so the call sites stay readable
# and we keep working against both wire shapes (the in-tree test stub
# still exposes ``BOOL``).
_PAYLOAD_BOOL = getattr(
    qmodels.PayloadSchemaType,
    "BOOL",
    getattr(qmodels.PayloadSchemaType, "BOOLEAN", None),
)

try:  # pragma: no cover - optional dependency for real deployments
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - lightweight fallback used in tests
    import hashlib

    class SentenceTransformer:  # type: ignore[override]
        """Deterministic embedding stub mirroring the FAISS fallback."""

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self._dimension = 384

        def get_sentence_embedding_dimension(self) -> int:
            return self._dimension

        def encode(self, texts, *, convert_to_numpy: bool = True):
            vectors = []
            for text in texts:
                digest = hashlib.sha256((self.model_name + str(text)).encode("utf-8")).digest()
                raw = np.frombuffer(digest * 8, dtype=np.uint8)[: self._dimension]
                vector = raw.astype(np.float32)
                norm = np.linalg.norm(vector) or 1.0
                vectors.append(vector / norm)
            array = (
                np.vstack(vectors) if vectors else np.zeros((0, self._dimension), dtype=np.float32)
            )
            if convert_to_numpy:
                return array
            return array.tolist()


from app.core.config import Settings, get_settings
from app.retriever.e5 import e5_prefix
from app.retriever.embedding_protocol import EmbedderProtocol
from app.retriever.vector_store import SearchFilters

__all__ = ["QdrantVectorStore"]

try:  # pragma: no cover - optional LangChain integration
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - lightweight fallback for tests

    class Document:  # type: ignore[override]
        def __init__(self, page_content: str, metadata: dict[str, object] | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}


class QdrantVectorStore:
    """Wrapper around :mod:`qdrant_client` configured via :class:`Settings`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder_factory: Callable[[str], EmbedderProtocol] | None = None,
        client_factory: Callable[..., QdrantClient] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_factory: Callable[[str], EmbedderProtocol] = embedder_factory or cast(
            Callable[[str], EmbedderProtocol], SentenceTransformer
        )
        self._client_factory = client_factory or QdrantClient
        self._model: EmbedderProtocol | None = None
        self._client: QdrantClient | None = None

        self._storage_dir = Path(self.settings.qdrant_path_resolved)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    def _embedder(self) -> EmbedderProtocol:
        if self._model is None:
            model = self._embedder_factory(self.settings.vector_embed_model)
            dimension = self._resolve_dimension(model)
            if dimension != self.settings.vector_embed_dimension:
                raise RuntimeError(
                    "Embedding model dimension mismatch: expected %s, got %s"
                    % (self.settings.vector_embed_dimension, dimension)
                )
            self._model = model
        return self._model

    @staticmethod
    def _resolve_dimension(model: EmbedderProtocol) -> int:
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = model.encode([""], convert_to_numpy=True)
        return int(sample.shape[1])

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            url = (self.settings.qdrant_url or "").strip()
            if url:
                kwargs: dict[str, object] = {"url": url}
            else:
                kwargs = {"path": str(self._storage_dir)}
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            self._client = self._client_factory(**kwargs)
        return self._client

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        if not len(vectors):
            return vectors
        norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return (vectors / norm).astype(np.float32)

    def _batched_encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.settings.vector_embed_dimension), dtype=np.float32)
        embedder = self._embedder()
        batch_size = max(1, int(self.settings.embed_batch_size))
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            encoded = embedder.encode(texts[start:end], convert_to_numpy=True)
            batches.append(np.asarray(encoded, dtype=np.float32))
        embeddings = np.vstack(batches)
        return self._normalise(embeddings)

    def _ensure_schema(self, client: QdrantClient) -> None:
        collection = self.settings.qdrant_collection
        dimension = self.settings.vector_embed_dimension

        try:
            info = client.get_collection(collection)
        except UnexpectedResponse:
            info = None

        needs_recreate = True
        if info and info.config and info.config.params:
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                sizes = {cfg.size for cfg in vectors.values() if cfg}
                needs_recreate = sizes != {dimension}
            else:
                needs_recreate = vectors.size != dimension

        if needs_recreate:
            client.recreate_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=48, ef_construct=256),
            )
            for field, schema in (
                ("file", qmodels.PayloadSchemaType.KEYWORD),
                ("page", qmodels.PayloadSchemaType.INTEGER),
                ("sha256", qmodels.PayloadSchemaType.KEYWORD),
                ("owner", qmodels.PayloadSchemaType.KEYWORD),
                ("tags", qmodels.PayloadSchemaType.KEYWORD),
                ("tenant_id", qmodels.PayloadSchemaType.KEYWORD),
                ("act_type", qmodels.PayloadSchemaType.KEYWORD),
                ("issuer", qmodels.PayloadSchemaType.KEYWORD),
                ("reg_number", qmodels.PayloadSchemaType.KEYWORD),
                ("is_active", _PAYLOAD_BOOL),
                ("revision", qmodels.PayloadSchemaType.KEYWORD),
                ("meta.act_type", qmodels.PayloadSchemaType.KEYWORD),
                ("meta.issuer", qmodels.PayloadSchemaType.KEYWORD),
                ("meta.reg_number", qmodels.PayloadSchemaType.KEYWORD),
                ("meta.is_active", _PAYLOAD_BOOL),
                ("meta.revision", qmodels.PayloadSchemaType.KEYWORD),
            ):
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )

    # ------------------------------------------------------------------
    # Vector store API
    def ensure_ready(self) -> None:
        client = self._client_instance()
        self._ensure_schema(client)

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        self.ensure_ready()

        client = self._client_instance()
        collection = self.settings.qdrant_collection
        max_batch = max(1, int(getattr(self.settings, "qdrant_upsert_batch", 512)))

        pending: dict[str, dict[str, object]] = {}

        def _flush() -> None:
            if not pending:
                return

            texts = [
                e5_prefix(
                    str(item.get("text") or item.get("content") or ""),
                    role="passage",
                    model=self.settings.vector_embed_model,
                    enabled=self.settings.vector_e5_prefix,
                )
                for item in pending.values()
            ]
            embeddings = self._batched_encode(texts)
            if not len(embeddings):
                pending.clear()
                return

            points: list[qmodels.PointStruct] = []
            for embedding, (identifier, chunk) in zip(embeddings, pending.items()):
                vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
                payload = {
                    "file": chunk.get("file"),
                    "page": int(chunk.get("page") or 0),
                    "sha256": chunk.get("sha256"),
                    "owner": chunk.get("owner"),
                    "tenant_id": chunk.get("tenant_id") or chunk.get("owner"),
                    "tags": chunk.get("tags") if isinstance(chunk.get("tags"), list) else [],
                    "text": chunk.get("text") or chunk.get("content"),
                    "meta": meta,
                    "act_type": meta.get("act_type"),
                    "issuer": meta.get("issuer"),
                    "reg_number": meta.get("reg_number"),
                    "is_active": meta.get("is_active"),
                    "revision": meta.get("revision"),
                }
                points.append(
                    qmodels.PointStruct(
                        id=identifier,
                        vector=vector,
                        payload=payload,
                    )
                )

            client.upsert(collection_name=collection, points=points)
            pending.clear()

        for chunk in chunks:
            identifier = str(chunk.get("sha256") or chunk.get("id") or "")
            if not identifier:
                raise ValueError("Chunk is missing sha256 identifier")
            pending[identifier] = chunk
            if len(pending) >= max_batch:
                _flush()

        _flush()

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters,
    ) -> list[dict[str, object]]:
        if top_k <= 0:
            return []

        self.ensure_ready()
        query_vector = self._batched_encode(
            [
                e5_prefix(
                    query,
                    role="query",
                    model=self.settings.vector_embed_model,
                    enabled=self.settings.vector_e5_prefix,
                )
            ]
        )
        if not len(query_vector):
            return []

        client = self._client_instance()
        query_filter = self._to_qdrant_filter(filters)

        results = client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector[0].tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        hits: list[dict[str, object]] = []
        for record in results:
            payload = getattr(record, "payload", {}) or {}
            payload = dict(payload)
            payload.setdefault("id", getattr(record, "id", None))
            payload.setdefault("file", payload.get("source"))
            payload.setdefault("source", payload.get("file"))
            payload.setdefault("article", None)
            payload.setdefault("clause", None)
            if isinstance(payload.get("meta"), dict):
                payload.setdefault("revision", payload.get("meta", {}).get("revision"))
            else:
                payload.setdefault("revision", None)
            payload["score"] = float(getattr(record, "score", 0.0))
            hits.append(payload)
        return hits

    @staticmethod
    def _to_qdrant_filter(filters: SearchFilters) -> qmodels.Filter:
        conditions: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="tenant_id",
                match=qmodels.MatchValue(value=filters.tenant_id),
            )
        ]
        if filters.owner:
            conditions.append(
                qmodels.FieldCondition(key="owner", match=qmodels.MatchValue(value=filters.owner))
            )
        for tag in filters.tags:
            conditions.append(
                qmodels.FieldCondition(key="tags", match=qmodels.MatchValue(value=tag))
            )
        if filters.act_type:
            conditions.append(
                qmodels.FieldCondition(
                    key="act_type", match=qmodels.MatchValue(value=filters.act_type)
                )
            )
        if filters.issuer:
            conditions.append(
                qmodels.FieldCondition(key="issuer", match=qmodels.MatchText(text=filters.issuer))
            )
        if filters.reg_number:
            conditions.append(
                qmodels.FieldCondition(
                    key="reg_number", match=qmodels.MatchValue(value=filters.reg_number)
                )
            )
        if filters.is_active is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="is_active", match=qmodels.MatchValue(value=filters.is_active)
                )
            )
        if filters.revision_mode == "current":
            conditions.append(
                qmodels.FieldCondition(key="is_active", match=qmodels.MatchValue(value=True))
            )
        elif filters.revision_mode == "historical":
            conditions.append(
                qmodels.FieldCondition(key="is_active", match=qmodels.MatchValue(value=False))
            )
        return qmodels.Filter(must=conditions)

    def as_retriever(self, *, query: str, top_k: int, filters: SearchFilters) -> list[Document]:
        hits = self.search(query=query, top_k=top_k, filters=filters)
        return self.hits_to_documents(hits)

    @staticmethod
    def hits_to_documents(hits: Sequence[dict[str, object]]) -> list[Document]:
        documents: list[Document] = []
        for hit in hits:
            payload = dict(hit)
            page_content = str(payload.pop("text", payload.pop("content", "")))
            documents.append(Document(page_content=page_content, metadata=payload))
        return documents

    def as_retriever_adapter(self, *, tenant_id: str, k: int = 10):
        from app.langchain.retrievers import TenantFilteredQdrantRetriever

        return TenantFilteredQdrantRetriever(store=self, tenant_id=tenant_id, k=k)

    def resolve_collection_name(self, alias_name: str) -> str:
        client = self._client_instance()
        try:
            aliases = client.get_aliases()
            for item in getattr(aliases, "aliases", []) or []:
                if getattr(item, "alias_name", None) == alias_name:
                    return str(getattr(item, "collection_name"))
        except Exception:
            pass
        return alias_name

    def create_alias(self, alias_name: str, collection_name: str) -> None:
        client = self._client_instance()
        client.update_collection_aliases(
            change_aliases_operations=[
                qmodels.CreateAliasOperation(
                    create_alias=qmodels.CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            ]
        )

    def switch_alias(self, alias_name: str, collection_name: str) -> None:
        client = self._client_instance()
        client.update_collection_aliases(
            change_aliases_operations=[
                qmodels.CreateAliasOperation(
                    create_alias=qmodels.CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            ]
        )

    def delete_alias(self, alias_name: str) -> None:
        client = self._client_instance()
        client.update_collection_aliases(
            change_aliases_operations=[
                qmodels.DeleteAliasOperation(
                    delete_alias=qmodels.DeleteAlias(alias_name=alias_name)
                )
            ]
        )

    def create_collection_like(self, target_collection: str, source_collection: str) -> None:
        client = self._client_instance()
        info = client.get_collection(source_collection)
        vectors = info.config.params.vectors
        client.recreate_collection(collection_name=target_collection, vectors_config=vectors)

    def export_payloads_from_collection(self, collection_name: str, batch_size: int = 256):
        client = self._client_instance()
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            for record in records:
                payload = getattr(record, "payload", {}) or {}
                if getattr(record, "id", None) is not None:
                    payload.setdefault("id", record.id)
                if getattr(record, "vector", None) is not None:
                    payload.setdefault("vector", record.vector)
                yield payload
            if not offset:
                break

    def import_payloads_to_collection(
        self, collection_name: str, payloads: Iterable[dict[str, object]]
    ) -> None:
        client = self._client_instance()
        batch: list[qmodels.PointStruct] = []
        for payload in payloads:
            vector = payload.get("vector")
            text = payload.get("text")
            if vector is None or text is None:
                continue
            vector_list = list(vector.tolist()) if hasattr(vector, "tolist") else list(vector)
            if not vector_list:
                continue
            batch.append(
                qmodels.PointStruct(
                    id=str(payload.get("id") or payload.get("sha256") or ""),
                    vector=vector_list,
                    payload=dict(payload),
                )
            )
            if len(batch) >= 512:
                client.upsert(collection_name=collection_name, points=list(batch))
                batch.clear()
        if batch:
            client.upsert(collection_name=collection_name, points=list(batch))

    def validate_collection_not_empty(self, collection_name: str) -> None:
        client = self._client_instance()
        records, _ = client.scroll(
            collection_name=collection_name, limit=1, with_payload=False, with_vectors=False
        )
        if not records:
            raise ValueError("Temporary collection is empty after reindex")

    def delete_collection_safe(self, collection_name: str) -> None:
        client = self._client_instance()
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    def ensure_collection(self) -> None:  # pragma: no cover - compatibility shim
        self.ensure_ready()

    def upsert_chunks(self, chunks: Iterable[dict[str, object]]) -> None:  # pragma: no cover
        self.upsert(chunks)

    def reset_collection(self) -> None:
        client = self._client_instance()
        try:
            client.delete_collection(collection_name=self.settings.qdrant_collection)
        except Exception:  # pragma: no cover - collection may not exist
            pass
        self.ensure_ready()

    def export_payloads(self, batch_size: int = 256) -> Iterator[dict[str, object]]:
        client = self._client_instance()
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            for record in records:
                payload = getattr(record, "payload", {}) or {}
                if getattr(record, "id", None) is not None:
                    payload.setdefault("id", record.id)
                if getattr(record, "vector", None) is not None:
                    payload.setdefault("vector", record.vector)
                yield payload
            if not offset:
                break

    def import_payloads(self, payloads: Iterable[dict[str, object]]) -> None:
        self.ensure_ready()
        client = self._client_instance()
        collection = self.settings.qdrant_collection
        batch: list[qmodels.PointStruct] = []
        for payload in payloads:
            vector = payload.get("vector")
            text = payload.get("text")
            if vector is None or text is None:
                continue
            vector_list = list(vector.tolist()) if hasattr(vector, "tolist") else list(vector)
            if not vector_list:
                continue
            batch.append(
                qmodels.PointStruct(
                    id=str(payload.get("id") or payload.get("sha256") or ""),
                    vector=vector_list,
                    payload={
                        "file": payload.get("file"),
                        "page": payload.get("page"),
                        "sha256": payload.get("sha256"),
                        "text": text,
                    },
                )
            )
            if len(batch) >= 512:
                client.upsert(collection_name=collection, points=list(batch))
                batch.clear()
        if batch:
            client.upsert(collection_name=collection, points=list(batch))
