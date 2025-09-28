"""Utility helpers for interacting with Qdrant."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIMENSION = 384

_model: SentenceTransformer | None = None
_qdrant: QdrantClient | None = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        if hasattr(_model, "get_sentence_embedding_dimension"):
            dim = int(_model.get_sentence_embedding_dimension())
        else:  # pragma: no cover - compatibility fallback
            sample = _model.encode([""], convert_to_numpy=True)
            dim = int(sample.shape[1])
        if dim != EMBED_DIMENSION:
            raise RuntimeError(
                f"Embedding model dimension mismatch: expected {EMBED_DIMENSION}, got {dim}"
            )
    return _model


def _qdrant_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
    return _qdrant


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norm


def ensure_collection() -> None:
    client = _qdrant_client()

    try:
        info = client.get_collection(QDRANT_COLLECTION)
    except UnexpectedResponse:
        info = None

    needs_recreate = True
    if info and info.config and info.config.params:
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            sizes = {cfg.size for cfg in vectors.values() if cfg}
            needs_recreate = sizes != {EMBED_DIMENSION}
        else:
            needs_recreate = vectors.size != EMBED_DIMENSION

    if needs_recreate:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIMENSION, distance=qmodels.Distance.COSINE),
            hnsw_config=qmodels.HnswConfigDiff(m=48, ef_construct=256),
        )
        for field, schema in (
            ("file", qmodels.PayloadSchemaType.KEYWORD),
            ("page", qmodels.PayloadSchemaType.INTEGER),
            ("sha256", qmodels.PayloadSchemaType.KEYWORD),
        ):
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )


def _encode_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIMENSION), dtype=np.float32)
    embeddings = _embedder().encode(texts, convert_to_numpy=True)
    return _normalise(embeddings).astype(np.float32)


def upsert_chunks(chunks: Iterable[dict[str, object]]) -> None:
    ensure_collection()

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
    embeddings = _encode_texts(texts)
    if not len(embeddings):
        return

    client = _qdrant_client()
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

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search_chunks(query: str, top_k: int = 10) -> List[dict[str, object]]:
    ensure_collection()

    query_vector = _encode_texts([query])
    if not len(query_vector):
        return []

    client = _qdrant_client()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
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


__all__ = ["ensure_collection", "search_chunks", "upsert_chunks"]
