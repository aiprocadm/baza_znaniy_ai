import os
from typing import Dict, List
from uuid import uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks")

_model: SentenceTransformer | None = None
_expected_dim: int | None = None
_qdrant: QdrantClient | None = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    return _model


def _embedding_dim() -> int:
    global _expected_dim
    if _expected_dim is None:
        model = _embedder()
        if hasattr(model, "get_sentence_embedding_dimension"):
            _expected_dim = int(model.get_sentence_embedding_dimension())
        else:
            sample = model.encode([""], convert_to_numpy=True)
            _expected_dim = int(sample.shape[1])
    return _expected_dim


def _qdrant_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
    return _qdrant


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def ensure_collection() -> None:
    client = _qdrant_client()
    dim = _embedding_dim()
    info = None
    try:
        info = client.get_collection(QDRANT_COLLECTION)
    except UnexpectedResponse:
        info = None

    needs_recreate = True
    if info is not None and info.config and info.config.params:
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            sizes = {cfg.size for cfg in vectors.values() if cfg}
            needs_recreate = sizes != {dim}
        else:
            needs_recreate = vectors.size != dim

    if needs_recreate:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="file",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="page",
            field_schema=qmodels.PayloadSchemaType.INTEGER,
        )


def upsert_chunks(chunks: List[Dict]) -> None:
    ensure_collection()
    texts = [c["text"] for c in chunks]
    if not texts:
        return
    embs = _norm(_embedder().encode(texts, convert_to_numpy=True)).astype(np.float32)
    client = _qdrant_client()
    points = []
    for vec, ch in zip(embs, chunks):
        payload = {
            "file": ch.get("file"),
            "page": int(ch.get("page") or 0),
            "text": ch["text"],
        }
        points.append(
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=vec.tolist(),
                payload=payload,
            )
        )
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search_chunks(query: str, top_k: int = 10) -> List[Dict]:
    ensure_collection()
    q = _norm(_embedder().encode([query], convert_to_numpy=True)).astype(np.float32)
    client = _qdrant_client()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q[0].tolist(),
        limit=top_k,
        with_payload=True,
        score_threshold=None,
    )
    hits: List[Dict] = []
    for item in results:
        payload = item.payload or {}
        hits.append(
            {
                "file": payload.get("file"),
                "page": payload.get("page"),
                "text": payload.get("text", ""),
                "score": float(item.score),
            }
        )
    return hits
