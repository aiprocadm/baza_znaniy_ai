"""Qdrant-backed vector store implementation."""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Wrapper around ``qdrant-client`` configured via :class:`Settings`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None

    def _embedder(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.vector_embed_model)
            dimension = self._resolve_dimension(self._model)
            if dimension != self.settings.vector_embed_dimension:
                raise RuntimeError(
                    "Embedding model dimension mismatch: expected %s, got %s"
                    % (self.settings.vector_embed_dimension, dimension)
                )
        return self._model

    @staticmethod
    def _resolve_dimension(model: SentenceTransformer) -> int:
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = model.encode([""], convert_to_numpy=True)
        return int(sample.shape[1])

    def _client_instance(self) -> QdrantClient:
        if self._client is None:
            kwargs = {"url": self.settings.qdrant_url}
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norm

    def ensure_collection(self) -> None:
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

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.settings.vector_embed_dimension), dtype=np.float32)
        embeddings = self._embedder().encode(texts, convert_to_numpy=True)
        return self._normalise(embeddings).astype(np.float32)

    def upsert_chunks(self, chunks: Iterable[dict[str, object]]) -> None:
        self.ensure_collection()

        chunk_list = list(chunks)
        if not chunk_list:
            return

        unique_chunks: dict[str, dict[str, object]] = {}
        for chunk in chunk_list:
            sha = str(chunk.get("sha256") or "")
            if not sha:
                raise ValueError("Chunk is missing sha256")
            unique_chunks[sha] = chunk

        texts = [str(chunk["text"]) for chunk in unique_chunks.values()]
        embeddings = self._encode_texts(texts)
        if not len(embeddings):
            return

        client = self._client_instance()
        points: List[qmodels.PointStruct] = []
        for embedding, chunk in zip(embeddings, unique_chunks.values()):
            payload = {
                "file": chunk.get("file"),
                "page": int(chunk.get("page") or 0),
                "sha256": chunk.get("sha256"),
                "text": chunk.get("text"),
            }
            points.append(
                qmodels.PointStruct(
                    id=str(chunk["sha256"]),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def reset_collection(self) -> None:
        client = self._client_instance()
        try:
            client.delete_collection(collection_name=self.settings.qdrant_collection)
        except Exception:  # pragma: no cover - collection may not exist
            logger.debug("Collection %s did not exist during reset", self.settings.qdrant_collection)
        self.ensure_collection()

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
        self.ensure_collection()

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

    def search(self, query: str, top_k: int) -> List[dict[str, object]]:
        self.ensure_collection()

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


def get_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    """Return a configured ``QdrantVectorStore`` instance."""

    return QdrantVectorStore(settings=settings)


__all__ = ["QdrantVectorStore", "get_vector_store"]
