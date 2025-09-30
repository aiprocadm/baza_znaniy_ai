"""Qdrant-backed vector store implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings


class QdrantVectorStore:
    """Wrapper around :mod:`qdrant_client` configured via :class:`Settings`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder_factory: Callable[[str], SentenceTransformer] | None = None,
        client_factory: Callable[..., QdrantClient] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_factory = embedder_factory or SentenceTransformer
        self._client_factory = client_factory or QdrantClient
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None

        data_root = Path(os.getenv("DATA_DIR", str(self.settings.data_dir)))
        self._storage_dir = data_root / "qdrant"
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    def _embedder(self) -> SentenceTransformer:
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
    def _resolve_dimension(model: SentenceTransformer) -> int:
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = model.encode([""], convert_to_numpy=True)
        return int(sample.shape[1])

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            self._client = self._client_factory(path=str(self._storage_dir))
        return self._client

    def _normalise(self, vectors: np.ndarray) -> np.ndarray:
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

    def _collection_exists(self, client: QdrantClient) -> bool:
        try:
            client.get_collection(self.settings.qdrant_collection)
        except UnexpectedResponse:
            return False
        return True

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
        items = list(chunks)
        if not items:
            return

        self.ensure_ready()

        unique: dict[str, dict[str, object]] = {}
        for chunk in items:
            identifier = str(chunk.get("sha256") or chunk.get("id") or "")
            if not identifier:
                raise ValueError("Chunk is missing sha256 identifier")
            unique[identifier] = chunk

        texts = [str(chunk.get("text") or chunk.get("content") or "") for chunk in unique.values()]
        embeddings = self._batched_encode(texts)
        if not len(embeddings):
            return

        points: List[qmodels.PointStruct] = []
        for embedding, (identifier, chunk) in zip(embeddings, unique.items()):
            payload = {
                "file": chunk.get("file"),
                "page": int(chunk.get("page") or 0),
                "sha256": chunk.get("sha256"),
                "text": chunk.get("text") or chunk.get("content"),
            }
            points.append(
                qmodels.PointStruct(
                    id=identifier,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        client = self._client_instance()
        client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        if top_k <= 0:
            return []

        self.ensure_ready()
        query_vector = self._batched_encode([query])
        if not len(query_vector):
            return []

        client = self._client_instance()
        results = client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector[0].tolist(),
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, object]] = []
        for record in results:
            payload = getattr(record, "payload", {}) or {}
            payload = dict(payload)
            payload.setdefault("id", getattr(record, "id", None))
            payload["score"] = float(getattr(record, "score", 0.0))
            hits.append(payload)
        return hits

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
        except Exception:
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
        points: List[qmodels.PointStruct] = []
        for payload in payloads:
            vector = payload.get("vector")
            text = payload.get("text")
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
            client = self._client_instance()
            client.upsert(collection_name=self.settings.qdrant_collection, points=points)


__all__ = ["QdrantVectorStore"]
