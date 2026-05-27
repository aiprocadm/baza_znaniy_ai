"""In-memory qdrant client stub used for unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from .http import models as qmodels
from .http.exceptions import UnexpectedResponse


@dataclass
class _Record:
    id: str
    score: float
    payload: Dict[str, object]
    vector: List[float]


def _match_condition(payload: Dict[str, object], condition: qmodels.FieldCondition) -> bool:
    raw = payload.get(condition.key)
    match = condition.match
    if isinstance(match, qmodels.MatchValue):
        if isinstance(raw, list):
            return match.value in raw
        return raw == match.value
    if isinstance(match, qmodels.MatchText):
        return match.text.lower() in str(raw or "").lower()
    return True


def _match_filter(payload: Dict[str, object], flt: qmodels.Filter) -> bool:
    for clause in flt.must or []:
        if not _match_condition(payload, clause):
            return False
    for clause in flt.must_not or []:
        if _match_condition(payload, clause):
            return False
    if flt.should:
        if not any(_match_condition(payload, clause) for clause in flt.should):
            return False
    return True


class QdrantClient:
    def __init__(self, **_: object) -> None:
        self._collections: Dict[str, Dict[str, _Record]] = {}
        self._schemas: Dict[str, qmodels.VectorParams] = {}

    # Collection management -------------------------------------------------
    def get_collection(self, collection_name: str):
        if collection_name not in self._collections:
            raise UnexpectedResponse(f"Collection {collection_name!r} missing")
        params = SimpleNamespace(vectors=self._schemas.get(collection_name))
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)

    def recreate_collection(
        self, collection_name: str, vectors_config: qmodels.VectorParams, **_: object
    ) -> None:
        self._schemas[collection_name] = vectors_config
        self._collections[collection_name] = {}

    def create_payload_index(self, *_, **__):  # pragma: no cover - no-op stub
        return None

    def delete_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)
        self._schemas.pop(collection_name, None)

    # Data access -----------------------------------------------------------
    def upsert(self, *, collection_name: str, points: Iterable[qmodels.PointStruct]) -> None:
        store = self._collections.setdefault(collection_name, {})
        for point in points:
            vector = list(point.vector)
            payload = dict(point.payload)
            identifier = str(point.id)
            store[identifier] = _Record(id=identifier, score=1.0, payload=payload, vector=vector)

    def search(
        self,
        *,
        collection_name: str,
        query_vector: Iterable[float],
        limit: int,
        with_payload: bool = True,
        query_filter: Optional[qmodels.Filter] = None,
    ) -> List[qmodels.ScoredPoint]:
        store = self._collections.get(collection_name, {})
        results: List[qmodels.ScoredPoint] = []
        for record in store.values():
            if query_filter is not None and not _match_filter(record.payload, query_filter):
                continue
            results.append(
                qmodels.ScoredPoint(
                    id=record.id,
                    score=record.score,
                    payload=dict(record.payload) if with_payload else {},
                    vector=list(record.vector),
                )
            )
        return results[:limit]

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[qmodels.ScoredPoint], Optional[str]]:
        store = self._collections.get(collection_name, {})
        keys = sorted(store.keys())
        start = 0
        if offset:
            try:
                start = keys.index(offset) + 1
            except ValueError:
                start = len(keys)
        batch = keys[start : start + limit]
        records: List[qmodels.ScoredPoint] = []
        for key in batch:
            record = store[key]
            payload = dict(record.payload) if with_payload else {}
            vector = list(record.vector) if with_vectors else []
            records.append(
                qmodels.ScoredPoint(
                    id=record.id, score=record.score, payload=payload, vector=vector
                )
            )
        next_offset = batch[-1] if len(batch) == limit else None
        return records, next_offset


__all__ = ["QdrantClient", "UnexpectedResponse", "qmodels"]
