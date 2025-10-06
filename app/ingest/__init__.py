"""Ingestion utilities and services."""

from importlib import import_module
from typing import Any

from .chunking import (
    _CharTokenizer,
    _chunk,
    _clean,
    _get_tokenizer,
    iter_document_pages,
    parse_and_chunk,
)

__all__ = [
    "_CharTokenizer",
    "_chunk",
    "_clean",
    "_get_tokenizer",
    "iter_document_pages",
    "parse_and_chunk",
    "IngestJob",
    "IngestQueueFullError",
    "IngestService",
    "IngestWorker",
]


def __getattr__(name: str) -> Any:
    if name in {"IngestJob", "IngestQueueFullError", "IngestService", "IngestWorker"}:
        try:
            service = import_module("app.ingest.service")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise AttributeError(
                "app.ingest.service is unavailable because optional dependencies were "
                "not installed. Install the project's runtime requirements to use the "
                "ingestion service."
            ) from exc
        value = getattr(service, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
