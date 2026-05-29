"""Unified retrieval-degradation contract shared by both retrieval paths.

Dependency-light by design (only an optional ``prometheus_client`` import):
both the heavy ``/api/v1`` vector path and the light MVP ``/api/kb`` SQLite
path import this without coupling either to the other's dependencies.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum


class RetrievalSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"  # results may be incomplete
    CRITICAL = "critical"  # results likely useless or empty


class RetrievalReason(str, Enum):
    HEALTHY = "healthy"
    VECTOR_BACKEND_DOWN = "vector_backend_down"  # Qdrant/FAISS failed -> grep fallback
    HASHING_EMBEDDER = "hashing_embedder"  # near-random matches
    EMBEDDING_DIM_MISMATCH = "embedding_dim_mismatch"  # index incoherent with active embedder
    SEARCH_TRUNCATED = "search_truncated"  # hard-limit hit


_SEVERITY: dict[RetrievalReason, RetrievalSeverity] = {
    RetrievalReason.VECTOR_BACKEND_DOWN: RetrievalSeverity.CRITICAL,
    RetrievalReason.HASHING_EMBEDDER: RetrievalSeverity.CRITICAL,
    RetrievalReason.EMBEDDING_DIM_MISMATCH: RetrievalSeverity.CRITICAL,
    RetrievalReason.SEARCH_TRUNCATED: RetrievalSeverity.WARNING,
}


def severity_of(reasons) -> RetrievalSeverity:
    sev = [_SEVERITY[r] for r in reasons if r in _SEVERITY]
    if RetrievalSeverity.CRITICAL in sev:
        return RetrievalSeverity.CRITICAL
    if RetrievalSeverity.WARNING in sev:
        return RetrievalSeverity.WARNING
    return RetrievalSeverity.OK


@dataclass(frozen=True)
class RetrievalReport:
    source: str  # "sqlite" | "vector" | "fallback"
    reasons: tuple[RetrievalReason, ...] = ()
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.reasons)

    @property
    def severity(self) -> RetrievalSeverity:
        return severity_of(self.reasons)


# ---------------------------------------------------------------------------
# Gauge (optional prometheus_client)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - prometheus_client is optional in minimal MVP envs
    from prometheus_client import Gauge

    _RETRIEVAL_DEGRADED = Gauge(
        "kb_retrieval_degraded",
        "Whether retrieval is currently degraded (1) per reason.",
        labelnames=("reason", "severity"),
    )
except Exception:  # pragma: no cover - gauge becomes a no-op when unavailable
    _RETRIEVAL_DEGRADED = None


def _set_gauge(reason: RetrievalReason, active: bool) -> None:
    if _RETRIEVAL_DEGRADED is None:
        return
    severity = _SEVERITY.get(reason, RetrievalSeverity.WARNING).value
    _RETRIEVAL_DEGRADED.labels(reason=reason.value, severity=severity).set(
        1.0 if active else 0.0
    )


# ---------------------------------------------------------------------------
# Registry, ContextVar, snapshot
# ---------------------------------------------------------------------------

# Which reasons each search source is authoritative for, so a clean run
# from that source clears them immediately (TTL is only a backstop).
_SOURCE_REASONS: dict[str, set[RetrievalReason]] = {
    "sqlite": {
        RetrievalReason.HASHING_EMBEDDER,
        RetrievalReason.EMBEDDING_DIM_MISMATCH,
        RetrievalReason.SEARCH_TRUNCATED,
    },
    "vector": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
    "fallback": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
}

_DEFAULT_TTL = 300.0
_LOCK = threading.Lock()
_REGISTRY: dict[RetrievalReason, tuple[float, str]] = {}  # reason -> (monotonic_ts, detail)
_CURRENT: ContextVar[RetrievalReport | None] = ContextVar("retrieval_report", default=None)


def report(rep: RetrievalReport) -> None:
    """Record *rep* for health/metrics and expose it to this request context."""

    _CURRENT.set(rep)
    active = set(rep.reasons)
    governed = _SOURCE_REASONS.get(rep.source, set())
    now = time.monotonic()
    with _LOCK:
        for reason in active:
            _REGISTRY[reason] = (now, rep.detail)
        for reason in governed - active:
            _REGISTRY.pop(reason, None)

    for reason in active:
        _set_gauge(reason, True)
    for reason in governed - active:
        _set_gauge(reason, False)


def current_report() -> RetrievalReport | None:
    """Return the report recorded earlier in this request context, if any."""

    return _CURRENT.get()


def snapshot(ttl_seconds: float = _DEFAULT_TTL, extra: tuple = ()) -> dict:
    """Current degradations within *ttl_seconds*, merged with active probes.

    *extra* is an iterable of ``(RetrievalReason, detail)`` from cheap
    config-level probes a health endpoint runs (e.g. embedder == hash).
    """

    now = time.monotonic()
    reasons: list[dict] = []
    seen: set[RetrievalReason] = set()
    with _LOCK:
        items = list(_REGISTRY.items())
    for reason, (ts, detail) in items:
        if now - ts <= ttl_seconds:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": round(now - ts, 1),
                }
            )
            seen.add(reason)
    for reason, detail in extra:
        if reason not in seen:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": 0.0,
                }
            )
            seen.add(reason)
    return {"degraded": bool(reasons), "severity": severity_of(seen).value, "reasons": reasons}


def reset() -> None:
    """Clear all recorded state (test helper)."""

    with _LOCK:
        _REGISTRY.clear()
    _CURRENT.set(None)
