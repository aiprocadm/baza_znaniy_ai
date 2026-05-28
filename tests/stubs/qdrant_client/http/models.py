"""Minimal data models for the qdrant client stub.

Shape notes (kept in sync with the real ``qdrant-client~=1.11`` package
production code targets):

* ``MatchValue(value=...)`` — exact-value match. ``value`` can be a
  string, bool, or int. For multi-valued payload fields (e.g. ``tags``
  stored as a list), the qdrant server interprets ``MatchValue`` as
  *"the list contains this value"*; the stub's ``_match_condition``
  helper mirrors that (see ``tests/stubs/qdrant_client/__init__.py``).
* ``MatchText(text=...)`` — substring/full-text match (case-insensitive
  in the stub).
* ``PayloadSchemaType.BOOL`` — kept as the canonical name even though
  real ``qdrant-client>=1.13`` renamed it to ``BOOLEAN``. Production
  code (``app/retriever/qdrant.py:17-21``) resolves the symbol via
  ``getattr`` so both names work; the stub stays on ``BOOL`` for
  parity with the historical wire format.

The alias-management dataclasses (``CreateAlias`` / ``CreateAliasOperation``
/ ``DeleteAlias`` / ``DeleteAliasOperation``) are no-op wrappers — they
only have to satisfy attribute access since the stub's ``QdrantClient``
does not implement ``update_collection_aliases``. They are kept for
production-code-import parity (see ``app/retriever/qdrant.py:386-414``).
"""

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


@dataclass
class CreateAlias:
    """Mirror of qdrant ``models.CreateAlias`` — alias→collection binding."""

    collection_name: str
    alias_name: str


@dataclass
class CreateAliasOperation:
    """Mirror of qdrant ``models.CreateAliasOperation`` — alias-create op envelope."""

    create_alias: CreateAlias


@dataclass
class DeleteAlias:
    """Mirror of qdrant ``models.DeleteAlias`` — alias removal."""

    alias_name: str


@dataclass
class DeleteAliasOperation:
    """Mirror of qdrant ``models.DeleteAliasOperation`` — alias-delete op envelope."""

    delete_alias: DeleteAlias


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
    "CreateAlias",
    "CreateAliasOperation",
    "DeleteAlias",
    "DeleteAliasOperation",
]
