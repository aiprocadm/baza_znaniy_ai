"""Cross-encoder based reranking utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Wrapper around :class:`sentence_transformers.CrossEncoder`."""

    def __init__(
        self,
        model: "CrossEncoder | None" = None,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore import
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("sentence-transformers is required for reranking") from exc

        if model is None:
            model = CrossEncoder(model_name)

        self._model = model
        self._batch_size = max(1, int(batch_size))

    def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, object]],
        top_k: int,
    ) -> List[dict[str, object]]:
        """Return the highest scoring hits ordered by cross-encoder scores."""

        if not hits:
            return []

        limit = max(1, min(int(top_k), len(hits)))

        pairs: List[tuple[str, str]] = []
        for hit in hits:
            text = str(hit.get("text") or "")
            pairs.append((query, text))

        scores: List[float] = []
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


__all__ = ["CrossEncoderReranker"]
