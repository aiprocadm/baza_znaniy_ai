        # codex/implement-vector-store-interface-and-refactor-qdrant-logic
"""Vector store abstractions and concrete implementations."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, Iterator, List, MutableMapping, Protocol, Sequence, runtime_checkable

import faiss  # type: ignore[import-untyped]
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

"""Common vector store interfaces and factory helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Protocol, runtime_checkable
        # main

from app.core.config import Settings, get_settings


        # codex/implement-vector-store-interface-and-refactor-qdrant-logic
logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol describing the behaviour of vector store backends."""

    def ensure_ready(self) -> None:
        """Ensure the underlying resources exist and are initialised."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        """Insert or update the provided chunks in the index."""

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Run a similarity search returning up to *top_k* results."""


class _EmbeddingMixin:
    """Utility mixin that provides SentenceTransformer management."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: SentenceTransformer | None = None

    def _embedder(self) -> SentenceTransformer:
        if self._model is None:
            cache_dir = self.settings.data_dir / "models" / "sentence-transformers"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.settings.vector_embed_model,
                cache_folder=str(cache_dir),
            )
            dimension = self._resolve_dimension(self._model)
            if dimension != self.settings.vector_embed_dimension:
                raise RuntimeError(
                    "Embedding model dimension mismatch: expected %s, got %s",
                    self.settings.vector_embed_dimension,
                    dimension,
                )
        return self._model

    @staticmethod
    def _resolve_dimension(model: SentenceTransformer) -> int:
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = model.encode([""], convert_to_numpy=True)
        if sample.ndim == 1:
            return int(sample.shape[0])
        return int(sample.shape[1])

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.settings.vector_embed_dimension), dtype=np.float32)

        embedder = self._embedder()
        batch_size = max(1, int(self.settings.embed_batch_size or 1))
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            embeddings = embedder.encode(chunk, convert_to_numpy=True)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            batches.append(np.asarray(embeddings, dtype=np.float32))

        if not batches:
            return np.zeros((0, self.settings.vector_embed_dimension), dtype=np.float32)

        matrix = np.vstack(batches)
        return self._normalise(matrix).astype(np.float32)

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norm

    @staticmethod
    def _deduplicate_chunks(
        chunks: Iterable[dict[str, object]],
    ) -> list[dict[str, object]]:
        unique: dict[str, dict[str, object]] = {}
        for chunk in chunks:
            sha = str(chunk.get("sha256") or chunk.get("id") or "").strip()
            if not sha:
                raise ValueError("Chunk is missing sha256 identifier")
            unique[sha] = chunk
        return list(unique.values())


class QdrantVectorStore(_EmbeddingMixin):
    """Wrapper around ``qdrant-client`` configured via :class:`Settings`."""

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        super().__init__(resolved)
        self._client: QdrantClient | None = None

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            kwargs = {"url": self.settings.qdrant_url}
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # VectorStoreProtocol implementation
    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        client = self._client_instance()
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
                    size=dimension, distance=qmodels.Distance.COSINE
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=48, ef_construct=256),
            )
            for field, schema in (
                ("file", qmodels.PayloadSchemaType.KEYWORD),
                ("page", qmodels.PayloadSchemaType.INTEGER),
                ("sha256", qmodels.PayloadSchemaType.KEYWORD),
            ):
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        self.ensure_ready()

        items = self._deduplicate_chunks(chunks)
        if not items:
            return

        texts = [str(chunk["text"]) for chunk in items]
        embeddings = self._encode_texts(texts)
        if not len(embeddings):
            return

        client = self._client_instance()
        points: List[qmodels.PointStruct] = []
        for embedding, chunk in zip(embeddings, items):
            payload = {
                "file": chunk.get("file"),
                "page": int(chunk.get("page") or 0),
                "sha256": chunk.get("sha256"),
                "text": chunk.get("text"),
            }
            points.append(
                qmodels.PointStruct(
                    id=str(chunk.get("sha256") or chunk.get("id")),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        self.ensure_ready()

        query_vector = self._encode_texts([query])
        if not len(query_vector):
            return []

        client = self._client_instance()
        results = client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector[0].tolist(),
            limit=top_k,
            with_payload=True,
            score_threshold=None,
            search_params=qmodels.SearchParams(hnsw_ef=128),
        )

        hits: List[dict[str, object]] = []
        for item in results:
            payload = item.payload or {}
            hits.append(
                {
                    "file": payload.get("file"),
                    "page": payload.get("page"),
                    "sha256": payload.get("sha256"),
                    "text": payload.get("text", ""),
                    "score": float(item.score),
                }
            )
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Additional helpers retained for backwards compatibility
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        self.ensure_ready()

    def upsert_chunks(self, chunks: Iterable[dict[str, object]]) -> None:
        self.upsert(chunks)

    def reset_collection(self) -> None:
        client = self._client_instance()
        try:
            client.delete_collection(collection_name=self.settings.qdrant_collection)
        except Exception:  # pragma: no cover - collection may not exist
            logger.debug(
                "Collection %s did not exist during reset",
                self.settings.qdrant_collection,
            )
        self.ensure_ready()

    def export_payloads(self, batch_size: int = 256) -> Iterator[dict[str, object]]:
        client = self._client_instance()
        next_page = None

        while True:
            records, next_page = client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=batch_size,
                offset=next_page,
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
            if not next_page:
                break

    def import_payloads(self, payloads: Iterable[dict[str, object]]) -> None:
        self.ensure_ready()

        client = self._client_instance()
        points: List[qmodels.PointStruct] = []
        for payload in payloads:
            text = payload.get("text")
            vector = payload.get("vector")
            if vector is None or text is None:
                continue
            vector_list = list(vector.tolist()) if hasattr(vector, "tolist") else list(vector)
            if not vector_list:
                continue
            points.append(
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
        if points:
            client.upsert(collection_name=self.settings.qdrant_collection, points=points)


class FaissVectorStore(_EmbeddingMixin):
    """FAISS-based vector store stored on the local filesystem."""

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        super().__init__(resolved)
        self._storage_dir = self.settings.data_dir / "faiss"
        self._index_path = self._storage_dir / "index.faiss"
        self._metadata_path = self._storage_dir / "metadata.json"
        self._index: faiss.Index | None = None
        self._payload_by_id: Dict[int, dict[str, object]] = {}
        self._sha_to_id: Dict[str, int] = {}
        self._next_id: int = 1
        self._metadata_loaded = False

    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
            else:
                self._index = self._build_index()

        if not hasattr(self._index, "add_with_ids"):
            self._index = faiss.IndexIDMap2(self._index)

        if not self._metadata_loaded:
            self._load_metadata()
            self._metadata_loaded = True

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        self.ensure_ready()

        items = self._deduplicate_chunks(chunks)
        if not items:
            return

        texts = [str(chunk["text"]) for chunk in items]
        embeddings = self._encode_texts(texts)
        if not len(embeddings):
            return

        index = self._require_index()

        vectors_to_add: list[np.ndarray] = []
        ids_to_add: list[int] = []

        for embedding, chunk in zip(embeddings, items):
            sha = str(chunk.get("sha256") or chunk.get("id") or "").strip()
            if not sha:
                raise ValueError("Chunk is missing sha256 identifier")

            payload = {
                "file": chunk.get("file"),
                "page": chunk.get("page"),
                "sha256": chunk.get("sha256"),
                "text": chunk.get("text"),
            }

            existing_id = self._sha_to_id.get(sha)
            if existing_id is not None:
                self._remove_id(existing_id)
            else:
                existing_id = self._next_id
                self._next_id += 1

            self._sha_to_id[sha] = existing_id
            self._payload_by_id[existing_id] = payload
            vectors_to_add.append(np.asarray(embedding, dtype=np.float32))
            ids_to_add.append(existing_id)

        if ids_to_add:
            vectors = np.vstack(vectors_to_add).astype(np.float32)
            id_array = np.asarray(ids_to_add, dtype=np.int64)
            index.add_with_ids(vectors, id_array)
            self._persist_state()

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        self.ensure_ready()

        index = self._require_index()
        if index.ntotal == 0:
            return []

        query_vector = self._encode_texts([query])
        if not len(query_vector):
            return []

        scores, ids = index.search(query_vector.astype(np.float32), top_k)
        if scores.size == 0:
            return []

        hits: list[dict[str, object]] = []
        for score, vector_id in zip(scores[0], ids[0]):
            if int(vector_id) == -1:
                continue
            payload = self._payload_by_id.get(int(vector_id))
            if not payload:
                continue
            hit = dict(payload)
            hit["score"] = float(score)
            hits.append(hit)
        return hits[:top_k]

    # ------------------------------------------------------------------
    def _build_index(self) -> faiss.Index:
        base = faiss.IndexFlatIP(self.settings.vector_embed_dimension)
        return faiss.IndexIDMap2(base)

    def _require_index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index has not been initialised")
        return self._index

    def _remove_id(self, vector_id: int) -> None:
        index = self._require_index()
        id_array = np.asarray([vector_id], dtype=np.int64)
        index.remove_ids(id_array)

    def _load_metadata(self) -> None:
        if not self._metadata_path.exists():
            return

        data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        self._next_id = int(data.get("next_id", 1))
        self._sha_to_id.clear()
        self._payload_by_id.clear()

        for sha, record in entries.items():
            try:
                vector_id = int(record["id"])
            except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
                continue
            payload = record.get("payload") or {}
            self._sha_to_id[sha] = vector_id
            self._payload_by_id[vector_id] = dict(payload)

    def _persist_state(self) -> None:
        index = self._require_index()
        faiss.write_index(index, str(self._index_path))

        entries: Dict[str, dict[str, object]] = {}
        for sha, vector_id in self._sha_to_id.items():
            payload = self._payload_by_id.get(vector_id, {})
            entries[sha] = {"id": vector_id, "payload": payload}

        data = {"next_id": self._next_id, "entries": entries}
        self._metadata_path.write_text(json.dumps(data), encoding="utf-8")


_VECTOR_STORE_CACHE: MutableMapping[str, VectorStore] = {}


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the configured vector store implementation."""

    resolved = settings or get_settings()
    backend = (resolved.vector_backend or "qdrant").strip().lower()

    store = _VECTOR_STORE_CACHE.get(backend)
    if store is not None:
        return store

    if backend == "qdrant":
        store = QdrantVectorStore(resolved)
    elif backend == "faiss":
        store = FaissVectorStore(resolved)
    else:
        raise ValueError(f"Unknown vector backend: {resolved.vector_backend}")

    _VECTOR_STORE_CACHE[backend] = store
    return store


__all__ = [
    "VectorStore",
    "QdrantVectorStore",
    "FaissVectorStore",
    "get_vector_store",
]


@runtime_checkable
class VectorStore(Protocol):
    """Minimal protocol implemented by vector store backends."""

    def ensure_ready(self) -> None:
        """Ensure the underlying store is ready for use."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        """Insert or update the provided chunks in the store."""

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Return the ``top_k`` most relevant chunks for ``query``."""


def _build_backend(settings: Settings) -> VectorStore:
    """Instantiate the configured vector store implementation."""

    from .faiss import FaissVectorStore
    from .qdrant import QdrantVectorStore

    backend = settings.vector_backend
    if backend == "qdrant":
        return QdrantVectorStore(settings=settings)
    if backend == "faiss":
        return FaissVectorStore(settings=settings)
    raise ValueError(f"Unsupported vector backend: {backend}")


@lru_cache(maxsize=1)
def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the cached vector store instance for the given settings."""

    resolved_settings = settings or get_settings()
    return _build_backend(resolved_settings)


__all__ = ["VectorStore", "get_vector_store"]
        # main
