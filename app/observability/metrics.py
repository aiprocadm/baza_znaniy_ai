"""Prometheus metrics and helpers for instrumenting core operations."""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Histogram

_DEFAULT_STATUS: Final[str] = "unknown"
_DEFAULT_EXTENSION: Final[str] = "unknown"
_DEFAULT_SOURCE: Final[str] = "unspecified"


def _normalise(value: str | None, default: str) -> str:
    value = (value or "").strip()
    return value or default


DOCUMENT_PARSE_TOTAL = Counter(
    "kb_document_parse_total",
    "Total number of document parse attempts.",
    labelnames=("status", "extension"),
)
DOCUMENT_PARSE_DURATION_SECONDS = Histogram(
    "kb_document_parse_duration_seconds",
    "Time spent parsing documents.",
    labelnames=("status",),
)
DOCUMENT_PARSE_CHUNKS_TOTAL = Counter(
    "kb_document_chunks_total",
    "Number of document chunks produced during parsing.",
    labelnames=("extension",),
)

INDEX_OPERATIONS_TOTAL = Counter(
    "kb_index_operations_total",
    "Total number of index operations attempted.",
    labelnames=("status", "backend"),
)
INDEX_DURATION_SECONDS = Histogram(
    "kb_index_duration_seconds",
    "Duration of index operations.",
    labelnames=("status", "backend"),
)
INDEX_CHUNKS_TOTAL = Counter(
    "kb_index_chunks_total",
    "Number of chunks processed by the index backends.",
    labelnames=("backend", "status"),
)

SEARCH_QUERIES_TOTAL = Counter(
    "kb_search_queries_total",
    "Total number of similarity search queries.",
    labelnames=("source", "status"),
)
SEARCH_DURATION_SECONDS = Histogram(
    "kb_search_duration_seconds",
    "Similarity search latency in seconds.",
    labelnames=("source", "status"),
)
SEARCH_HITS_TOTAL = Counter(
    "kb_search_hits_total",
    "Total hits returned by search operations.",
    labelnames=("source", "status"),
)

CHAT_COMPLETIONS_TOTAL = Counter(
    "kb_chat_completions_total",
    "Total number of chat completions handled by the service.",
    labelnames=("status",),
)
CHAT_LATENCY_SECONDS = Histogram(
    "kb_chat_latency_seconds",
    "Latency of chat completions in seconds.",
    labelnames=("status",),
)
CHAT_CONTEXT_HITS_TOTAL = Counter(
    "kb_chat_context_hits_total",
    "Number of context chunks retrieved for chat responses.",
    labelnames=("status",),
)
CHAT_CITATIONS_TOTAL = Counter(
    "kb_chat_citations_total",
    "Number of citations returned alongside chat responses.",
    labelnames=("status",),
)


def record_document_parse(extension: str | None, status: str, chunks: int, duration: float) -> None:
    """Record metrics for a document parsing attempt."""

    ext_label = _normalise(extension, _DEFAULT_EXTENSION)
    status_label = _normalise(status, _DEFAULT_STATUS)

    DOCUMENT_PARSE_TOTAL.labels(status=status_label, extension=ext_label).inc()
    DOCUMENT_PARSE_DURATION_SECONDS.labels(status=status_label).observe(max(duration, 0.0))
    if chunks > 0 and status_label == "success":
        DOCUMENT_PARSE_CHUNKS_TOTAL.labels(extension=ext_label).inc(chunks)


def record_index_operation(
    status: str,
    backend: str | None,
    chunks: int,
    duration: float,
) -> None:
    """Record metrics for an indexing attempt."""

    backend_label = _normalise(backend, _DEFAULT_SOURCE)
    status_label = _normalise(status, _DEFAULT_STATUS)

    INDEX_OPERATIONS_TOTAL.labels(status=status_label, backend=backend_label).inc()
    INDEX_DURATION_SECONDS.labels(status=status_label, backend=backend_label).observe(
        max(duration, 0.0)
    )
    if chunks > 0:
        INDEX_CHUNKS_TOTAL.labels(backend=backend_label, status=status_label).inc(chunks)


def record_search_operation(
    source: str | None,
    status: str,
    duration: float,
    hits: int,
) -> None:
    """Record metrics for similarity search operations."""

    source_label = _normalise(source, _DEFAULT_SOURCE)
    status_label = _normalise(status, _DEFAULT_STATUS)

    SEARCH_QUERIES_TOTAL.labels(source=source_label, status=status_label).inc()
    SEARCH_DURATION_SECONDS.labels(source=source_label, status=status_label).observe(
        max(duration, 0.0)
    )
    if hits > 0:
        SEARCH_HITS_TOTAL.labels(source=source_label, status=status_label).inc(hits)


def record_chat_completion(
    status: str,
    duration: float,
    *,
    hits: int = 0,
    citations: int = 0,
) -> None:
    """Record metrics for chat completions."""

    status_label = _normalise(status, _DEFAULT_STATUS)

    CHAT_COMPLETIONS_TOTAL.labels(status=status_label).inc()
    CHAT_LATENCY_SECONDS.labels(status=status_label).observe(max(duration, 0.0))
    if hits > 0:
        CHAT_CONTEXT_HITS_TOTAL.labels(status=status_label).inc(hits)
    if citations > 0:
        CHAT_CITATIONS_TOTAL.labels(status=status_label).inc(citations)
