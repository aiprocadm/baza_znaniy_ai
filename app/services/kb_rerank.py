"""Cross-encoder reranker for the MVP knowledge base.

Thin wrapper around :class:`app.retriever.rerank.CrossEncoderReranker`
that exposes MVP-specific env-variables (``KB_RERANK_*``) and works on
:class:`app.services.kb_store.SearchHit` dataclasses instead of raw
dicts.

Pipeline:
1. ``KnowledgeBaseStore.search`` returns up to ``KB_RERANK_CANDIDATES``
   bi-encoder hits (cosine over embeddings).
2. If reranking is enabled, the cross-encoder scores each
   ``(query, hit.text)`` pair and the top ``top_n`` survive.

Two env contracts intentionally coexist — legacy ``RERANK_*`` drives
``/api/v1/*`` (multi-tenant pipeline), MVP ``KB_RERANK_*`` drives
``/api/kb/*``. They can be configured independently.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

from app.services._envutil import env as _env
from app.services.kb_store import SearchHit

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_CANDIDATES = 20
DEFAULT_TOP_N = 5

_RERANKER_LOCK = threading.Lock()
_RERANKER_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class RerankConfig:
    """Resolved configuration for the MVP reranker."""

    enabled: bool
    model_name: str
    candidates: int
    top_n: int
    batch_size: int
    # When True, a reranker load/scoring failure is raised instead of degrading
    # to bi-encoder order. Serving keeps the graceful default (False) so a user
    # query never 500s; the eval/gate path forces it on so a broken model fails
    # loud rather than masquerading as base-identical metrics.
    strict: bool = False


@dataclass(frozen=True)
class RerankResult:
    """Outcome of a single rerank call (used for diagnostics)."""

    hits: List[SearchHit]
    model: str
    elapsed_ms: float
    candidates: int


def _bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Optional[str], default: int, *, low: int = 1, high: int = 200) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def load_config(env: Mapping[str, str] | None = None) -> RerankConfig:
    """Read reranker configuration from environment variables."""

    return RerankConfig(
        enabled=_bool(_env("KB_RERANK_ENABLED", env), default=False),
        model_name=_env("KB_RERANK_MODEL", env) or DEFAULT_MODEL_NAME,
        candidates=_int(_env("KB_RERANK_CANDIDATES", env), DEFAULT_CANDIDATES, low=1, high=200),
        top_n=_int(_env("KB_RERANK_TOPN", env), DEFAULT_TOP_N, low=1, high=50),
        batch_size=_int(_env("KB_RERANK_BATCH", env), 32, low=1, high=256),
        strict=_bool(_env("KB_RERANK_STRICT", env), default=False),
    )


def _get_reranker(config: RerankConfig):
    """Return a cached :class:`CrossEncoderReranker` for ``config.model_name``.

    Lazy: the heavy ``sentence-transformers`` import and the ~80–600 MB
    model download only happen on the first call.
    """

    cached = _RERANKER_CACHE.get(config.model_name)
    if cached is not None:
        return cached
    with _RERANKER_LOCK:
        cached = _RERANKER_CACHE.get(config.model_name)
        if cached is not None:
            return cached
        try:
            from app.retriever.rerank import CrossEncoderReranker
        except ImportError as exc:  # pragma: no cover - missing optional dep
            raise RuntimeError(
                "sentence-transformers is required for KB_RERANK_ENABLED=true"
            ) from exc
        reranker = CrossEncoderReranker(model_name=config.model_name, batch_size=config.batch_size)
        _RERANKER_CACHE[config.model_name] = reranker
        return reranker


def reset_cache() -> None:
    """Drop the cached reranker (used in tests)."""

    with _RERANKER_LOCK:
        _RERANKER_CACHE.clear()


def _hits_to_dicts(hits: Sequence[SearchHit]) -> list[dict]:
    return [
        {
            "document_id": hit.document_id,
            "document_title": hit.document_title,
            "chunk_index": hit.chunk_index,
            "text": hit.text,
            "score": hit.score,
            "source": hit.source,
            "filename": hit.filename,
        }
        for hit in hits
    ]


def _dict_to_hit(payload: Mapping[str, object], fallback_score: float) -> SearchHit:
    raw_score = payload.get("score", fallback_score)
    try:
        score = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score = fallback_score
    # ``payload`` values are typed ``object`` (Mapping[str, object]). The
    # ``or <int>`` fallback means int() only ever sees the original runtime
    # value or the literal default; the inline ignores match the existing
    # ``score`` pattern above and change no runtime behaviour. ``raw_filename``
    # is hoisted so the isinstance guard can narrow it to ``str | None``.
    raw_filename = payload.get("filename")
    return SearchHit(
        document_id=int(payload.get("document_id") or 0),  # type: ignore[call-overload]
        document_title=str(payload.get("document_title") or ""),
        chunk_index=int(payload.get("chunk_index") or 0),  # type: ignore[call-overload]
        text=str(payload.get("text") or ""),
        score=score,
        source=str(payload.get("source") or "text"),
        filename=raw_filename if isinstance(raw_filename, str) else None,
    )


def rerank_hits(
    query: str,
    hits: Sequence[SearchHit],
    *,
    config: Optional[RerankConfig] = None,
    top_n: Optional[int] = None,
) -> RerankResult:
    """Apply the cross-encoder reranker, return top hits + diagnostics.

    Returns the original hits unchanged (truncated to ``top_n``) when
    reranking is disabled or no candidates exist. Cross-encoder scores
    overwrite the per-hit ``score`` field so callers see the reranker's
    confidence (not the original cosine).
    """

    effective_config = config or load_config()
    effective_top_n = top_n if top_n is not None else effective_config.top_n
    effective_top_n = max(1, effective_top_n)

    if not hits:
        return RerankResult(
            hits=[], model=effective_config.model_name, elapsed_ms=0.0, candidates=0
        )

    if not effective_config.enabled:
        return RerankResult(
            hits=list(hits[:effective_top_n]),
            model=effective_config.model_name,
            elapsed_ms=0.0,
            candidates=len(hits),
        )

    start = time.perf_counter()
    try:
        reranker = _get_reranker(effective_config)
        reranked = reranker.rerank(query, _hits_to_dicts(hits), effective_top_n)
    except Exception:
        if effective_config.strict:
            # Eval/gate path: a silent bi-encoder fallback would feed the gate
            # base-identical metrics and discard a good model as a false NO-GO.
            LOGGER.exception("Reranker %s failed (strict)", effective_config.model_name)
            raise
        LOGGER.exception("Reranker %s failed; using bi-encoder order", effective_config.model_name)
        return RerankResult(
            hits=list(hits[:effective_top_n]),
            model=effective_config.model_name,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            candidates=len(hits),
        )

    converted = [_dict_to_hit(item, fallback_score=0.0) for item in reranked]
    return RerankResult(
        hits=converted,
        model=effective_config.model_name,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        candidates=len(hits),
    )


def reranker_status(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Diagnostic snapshot for ``GET /api/kb/health``."""

    config = load_config(env=env)
    return {
        "enabled": config.enabled,
        "model": config.model_name,
        "candidates": config.candidates,
        "top_n": config.top_n,
        "loaded": config.model_name in _RERANKER_CACHE,
    }


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TOP_N",
    "RerankConfig",
    "RerankResult",
    "load_config",
    "rerank_hits",
    "reranker_status",
    "reset_cache",
]
