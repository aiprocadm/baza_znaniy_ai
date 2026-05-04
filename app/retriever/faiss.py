"""FAISS-backed vector store implementation."""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, Sequence, cast

import faiss  # type: ignore[import]
import numpy as np

try:  # pragma: no cover - optional dependency for real deployments
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - lightweight fallback used in tests
    import hashlib

    class SentenceTransformer:  # type: ignore[override]
        """Minimal sentence transformer stub producing deterministic vectors."""

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
            array = np.vstack(vectors) if vectors else np.zeros((0, self._dimension), dtype=np.float32)
            if convert_to_numpy:
                return array
            return array.tolist()

from app.core.config import Settings, get_settings
from app.retriever.embedding_protocol import EmbedderProtocol
from app.retriever.vector_store import SearchFilters

EmbedderFactory = Callable[[str], EmbedderProtocol]
_default_embedder_factory: EmbedderFactory = cast(EmbedderFactory, SentenceTransformer)
LOGGER = logging.getLogger(__name__)


class FaissVectorStore:
    """Vector store persisting embeddings locally using FAISS."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_factory: EmbedderFactory = embedder_factory or _default_embedder_factory
        self._model: EmbedderProtocol | None = None
        self._index: faiss.Index | None = None
        self._payloads: "OrderedDict[str, dict[str, object]]" = OrderedDict()
        self._ordered_ids: list[str] = []

        data_root = Path(os.getenv("DATA_DIR", str(self.settings.data_dir)))
        self._storage_dir = data_root / "faiss"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._storage_dir / "index.faiss"
        self._payloads_path = self._storage_dir / "payloads.json"
        self._mapping_path = self._storage_dir / "mapping.json"

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
            batch = embedder.encode(texts[start:end], convert_to_numpy=True)
            batches.append(np.asarray(batch, dtype=np.float32))
        embeddings = np.vstack(batches)
        return self._normalise(embeddings)

    def _load_payloads(self) -> None:
        if self._payloads_path.exists():
            raw = json.loads(self._payloads_path.read_text("utf-8"))
            self._payloads = OrderedDict((item["id"], item["payload"]) for item in raw)
        else:
            self._payloads = OrderedDict()

        if self._mapping_path.exists():
            data = json.loads(self._mapping_path.read_text("utf-8"))
            if isinstance(data, list):
                self._ordered_ids = [str(item) for item in data if str(item) in self._payloads]
            else:
                self._ordered_ids = []
        else:
            self._ordered_ids = list(self._payloads.keys())

    def _persist_payloads(self) -> None:
        payload_items = [
            {"id": identifier, "payload": payload}
            for identifier, payload in self._payloads.items()
        ]
        self._payloads_path.write_text(json.dumps(payload_items), encoding="utf-8")
        self._mapping_path.write_text(json.dumps(self._ordered_ids), encoding="utf-8")

    def _rebuild_index(self) -> None:
        dimension = self.settings.vector_embed_dimension
        self._index = faiss.IndexFlatIP(dimension)
        if not self._ordered_ids:
            try:
                self._index_path.unlink()
            except FileNotFoundError:
                pass
            return

        texts = [str(self._payloads[item]["text"]) for item in self._ordered_ids]
        embeddings = self._batched_encode(texts)
        if not len(embeddings):
            self._ordered_ids = []
            try:
                self._index_path.unlink()
            except FileNotFoundError:
                pass
            self._index.reset()
            return

        self._index.add(embeddings)
        faiss.write_index(self._index, str(self._index_path))

    def _ensure_index_loaded(self) -> None:
        if self._index is not None:
            return
        if not self._payloads:
            self._load_payloads()
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Vector store API
    def ensure_ready(self) -> None:
        self._load_payloads()
        self._ensure_index_loaded()

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        items = list(chunks)
        if not items:
            return

        if not self._payloads:
            self._load_payloads()

        updated_ids: list[str] = []
        for chunk in items:
            identifier = str(chunk.get("sha256") or chunk.get("id") or "")
            if not identifier:
                raise ValueError("Chunk is missing sha256 identifier")
            payload = {
                "file": chunk.get("file"),
                "page": chunk.get("page"),
                "sha256": chunk.get("sha256"),
                "owner": chunk.get("owner"),
                "tags": chunk.get("tags") if isinstance(chunk.get("tags"), list) else [],
                "text": chunk.get("text") or chunk.get("content"),
                "tenant_id": chunk.get("tenant_id") or chunk.get("owner"),
                "meta": chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {},
            }
            self._payloads[identifier] = payload
            updated_ids.append(identifier)
            if identifier not in self._ordered_ids:
                self._ordered_ids.append(identifier)

        self._persist_payloads()
        self._rebuild_index()

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters,
    ) -> list[dict[str, object]]:
        if top_k <= 0:
            return []

        if not self._payloads:
            self._load_payloads()
        self._ensure_index_loaded()

        if self._index is None or not self._ordered_ids:
            return []

        query_vector = self._batched_encode([query])
        if not len(query_vector):
            return []

        LOGGER.warning("FAISS backend uses post-filtering; filters are applied after ANN candidate retrieval")
        candidate_limit = max(top_k, top_k * 20)
        scores, indices = self._index.search(query_vector, candidate_limit)
        normalized_owner = (filters.owner or "").strip().lower()
        normalized_tags = {tag.lower() for tag in filters.tags}
        normalized_tenant = filters.tenant_id.lower()
        hits: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ordered_ids):
                continue
            identifier = self._ordered_ids[idx]
            payload = dict(self._payloads.get(identifier, {}))
            payload_tenant = str(payload.get("tenant_id") or payload.get("owner") or "").strip().lower()
            if payload_tenant != normalized_tenant:
                continue
            if normalized_owner:
                payload_owner = str(payload.get("owner", "")).strip().lower()
                if payload_owner != normalized_owner:
                    continue
            if normalized_tags:
                payload_tags = payload.get("tags")
                if not isinstance(payload_tags, list):
                    continue
                payload_tag_set = {
                    str(tag).strip().lower() for tag in payload_tags if str(tag).strip()
                }
                if not normalized_tags.issubset(payload_tag_set):
                    continue
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            if filters.act_type and str(meta.get("act_type", "")).strip().lower() != filters.act_type.lower():
                continue
            if filters.issuer and filters.issuer.lower() not in str(meta.get("issuer", "")).strip().lower():
                continue
            if filters.reg_number and filters.reg_number.lower() != str(meta.get("reg_number", "")).strip().lower():
                continue
            if filters.is_active is not None and bool(meta.get("is_active", True)) is not filters.is_active:
                continue
            if filters.revision_mode == "current" and meta.get("is_active") is False:
                continue
            if filters.revision_mode == "historical" and meta.get("is_active") is True:
                continue
            payload.setdefault("id", identifier)
            payload["score"] = float(score)
            hits.append(payload)
            if len(hits) >= top_k:
                break
        return hits


__all__ = ["FaissVectorStore"]
