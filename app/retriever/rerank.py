"""Utility helpers for cross-encoder based reranking."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Mapping, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency guard
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - fallback when dependency missing
    CrossEncoder = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder as _CrossEncoder

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def is_rerank_enabled(
    env: Mapping[str, str] | None = None,
    default: bool = False,
) -> bool:
    """Return whether reranking should be enabled based on environment values."""

    source = env or os.environ
    value = source.get("RERANK_ENABLED")
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "yes", "on"}


def get_rerank_top_k(
    env: Mapping[str, str] | None = None,
    default: int = 10,
) -> int:
    """Return the configured ``top_k`` limit for reranking results."""

    source = env or os.environ
    value = source.get("RERANK_TOP_K", source.get("RERANK_TOPK"))
    if value in {None, ""}:
        return max(1, default)
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, parsed)


class CrossEncoderReranker:
    """Rerank hits using a cross-encoder model."""

    def __init__(
        self,
        model: "_CrossEncoder | None" = None,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
    ) -> None:
        if model is None:
            if CrossEncoder is None:  # pragma: no cover - dependency missing at runtime
                raise RuntimeError("sentence-transformers is required for reranking")
            model = CrossEncoder(model_name)
        self._model = model
        self._batch_size = max(1, int(batch_size))

    def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return the highest scoring hits for ``query`` with updated scores."""

        if not hits:
            return []

        limit = max(1, min(int(top_k), len(hits)))
        pairs = [(query, str(hit.get("text") or "")) for hit in hits]

        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            if not batch:
                continue
            batch_scores = self._model.predict(batch)  # type: ignore[attr-defined]
            scores.extend(float(score) for score in batch_scores)

        if len(scores) < len(hits):
            scores.extend([0.0] * (len(hits) - len(scores)))

        ranked = [dict(hit, score=float(score)) for hit, score in zip(hits, scores)]
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[:limit]


def apply_rerank(
    query: str,
    hits: Sequence[dict[str, object]],
    top_k: int,
    enabled: bool,
    reranker: CrossEncoderReranker | None = None,
) -> list[dict[str, object]]:
    """Conditionally rerank hits based on the configuration flags."""

    if not hits:
        return []

    limit = top_k if top_k and top_k > 0 else len(hits)
    if enabled and reranker is not None:
        return reranker.rerank(query, hits, limit)

    limit = min(max(1, limit), len(hits))
    return list(hits)[:limit]


@lru_cache(maxsize=1)
def get_reranker(model_name: str = DEFAULT_MODEL_NAME) -> CrossEncoderReranker:
    """Return a cached :class:`CrossEncoderReranker` instance."""

    return CrossEncoderReranker(model_name=model_name)


__all__ = [
    "CrossEncoderReranker",
    "apply_rerank",
    "get_rerank_top_k",
    "get_reranker",
    "is_rerank_enabled",
]
