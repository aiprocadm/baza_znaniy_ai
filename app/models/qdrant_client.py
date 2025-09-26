import hashlib
import os
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


_client: QdrantClient | None = None
_model: SentenceTransformer | None = None
_expected_dim: int | None = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
        _model = SentenceTransformer(model_name)
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


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "kb_chunks")


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        api_key = os.getenv("QDRANT_API_KEY") or None
        url = os.getenv("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=api_key)
        else:
            host = os.getenv("QDRANT_HOST", "qdrant")
            port_str = os.getenv("QDRANT_PORT")
            port = int(port_str) if port_str else 6333
            _client = QdrantClient(host=host, port=port, api_key=api_key)
    return _client


def _current_collection_dim(client: QdrantClient, collection: str) -> int | None:
    info = client.get_collection(collection)
    vectors_config = info.config.params.vectors
    if vectors_config is None:
        return None
    if isinstance(vectors_config, dict):
        first = next(iter(vectors_config.values()), None)
        return getattr(first, "size", None)
    return getattr(vectors_config, "size", None)


def ensure_collection() -> None:
    client = _get_client()
    dim = _embedding_dim()
    collection = _collection_name()

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return

    current_dim = _current_collection_dim(client, collection)
    if current_dim is not None and current_dim != dim:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _chunk_id(chunk: Dict) -> str:
    raw = f"{chunk.get('file','')}|{chunk.get('page','')}|{chunk.get('text','')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest


def upsert_chunks(chunks: List[Dict]) -> None:
    if not chunks:
        return

    ensure_collection()
    collection = _collection_name()
    client = _get_client()

    texts = [chunk["text"] for chunk in chunks]
    vectors = _embedder().encode(texts, convert_to_numpy=True)

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "file": chunk.get("file"),
            "page": chunk.get("page"),
            "text": chunk.get("text"),
        }
        points.append(
            PointStruct(
                id=_chunk_id(chunk),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)


def search_chunks(query: str, top_k: int = 10) -> List[Dict]:
    ensure_collection()
    collection = _collection_name()
    client = _get_client()

    query_vector = _embedder().encode([query], convert_to_numpy=True)[0].tolist()
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    hits: List[Dict] = []
    for point in results:
        payload = point.payload or {}
        hits.append(
            {
                "file": payload.get("file"),
                "page": payload.get("page"),
                "text": payload.get("text", ""),
                "score": float(point.score),
            }
        )
    return hits
