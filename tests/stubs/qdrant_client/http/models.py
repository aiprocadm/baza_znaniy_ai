"""Minimal data models for the qdrant client stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


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


__all__ = [
    "Distance",
    "VectorParams",
    "HnswConfigDiff",
    "PayloadSchemaType",
    "PointStruct",
    "ScoredPoint",
]
