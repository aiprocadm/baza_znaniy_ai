"""Minimal data models for the qdrant client stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


class Distance:
    COSINE = "cosine"


@dataclass
class VectorParams:
    size: int
    distance: str = Distance.COSINE


@dataclass
class HnswConfigDiff:
    m: int
    ef_construct: int


class PayloadSchemaType:
    KEYWORD = "keyword"
    INTEGER = "integer"
    BOOL = "bool"
    FLOAT = "float"
    GEO = "geo"
    TEXT = "text"


@dataclass
class PointStruct:
    id: str
    vector: Iterable[float]
    payload: Dict[str, Any]


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Iterable[float] | None = None


@dataclass
class MatchValue:
    """Mirror of qdrant ``models.MatchValue`` — exact-value matching."""

    value: Any


@dataclass
class MatchText:
    """Mirror of qdrant ``models.MatchText`` — full-text matching."""

    text: str


@dataclass
class FieldCondition:
    """Mirror of qdrant ``models.FieldCondition`` — single payload predicate."""

    key: str
    match: Any


@dataclass
class Filter:
    """Mirror of qdrant ``models.Filter`` — boolean composition of conditions."""

    must: Any = None
    should: Any = None
    must_not: Any = None


__all__ = [
    "Distance",
    "VectorParams",
    "HnswConfigDiff",
    "PayloadSchemaType",
    "PointStruct",
    "ScoredPoint",
    "MatchValue",
    "MatchText",
    "FieldCondition",
    "Filter",
]
