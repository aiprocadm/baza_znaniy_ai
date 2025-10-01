"""Tests for Prometheus metrics helper functions."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram

from app.observability import metrics


_METRIC_TYPES = (Counter, Histogram)


@pytest.fixture
def isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> CollectorRegistry:
    """Provide a registry-isolated copy of the metrics module globals."""

    registry = CollectorRegistry()

    for name in dir(metrics):
        metric = getattr(metrics, name)
        if isinstance(metric, _METRIC_TYPES):
            metric_cls = type(metric)
            kwargs = {
                "labelnames": metric._labelnames,  # type: ignore[attr-defined]
                "registry": registry,
            }
            replacement = metric_cls(
                metric._name,  # type: ignore[attr-defined]
                metric._documentation,  # type: ignore[attr-defined]
                **kwargs,
            )
            monkeypatch.setattr(metrics, name, replacement)

    return registry


def test_record_document_parse(isolated_metrics: CollectorRegistry) -> None:
    metrics.record_document_parse("  pdf  ", " success ", chunks=3, duration=1.5)
    metrics.record_document_parse(None, "", chunks=2, duration=-2.0)

    success_total = isolated_metrics.get_sample_value(
        "kb_document_parse_total", {"status": "success", "extension": "pdf"}
    )
    assert success_total == pytest.approx(1.0)

    unknown_total = isolated_metrics.get_sample_value(
        "kb_document_parse_total", {"status": "unknown", "extension": "unknown"}
    )
    assert unknown_total == pytest.approx(1.0)

    success_duration = isolated_metrics.get_sample_value(
        "kb_document_parse_duration_seconds_sum", {"status": "success"}
    )
    assert success_duration == pytest.approx(1.5)

    unknown_duration = isolated_metrics.get_sample_value(
        "kb_document_parse_duration_seconds_sum", {"status": "unknown"}
    )
    assert unknown_duration == pytest.approx(0.0)

    chunk_total = isolated_metrics.get_sample_value(
        "kb_document_chunks_total", {"extension": "pdf"}
    )
    assert chunk_total == pytest.approx(3.0)

    assert (
        isolated_metrics.get_sample_value(
            "kb_document_chunks_total", {"extension": "unknown"}
        )
        is None
    )


def test_record_document_ocr_pages(isolated_metrics: CollectorRegistry) -> None:
    metrics.record_document_ocr_pages(pages=3, status=" success ", extension="  pdf  ")
    metrics.record_document_ocr_pages(pages=0, status=" failure ", extension=None)

    success_total = isolated_metrics.get_sample_value(
        "kb_document_ocr_pages_total", {"status": "success", "extension": "pdf"}
    )
    assert success_total == pytest.approx(3.0)

    failure_total = isolated_metrics.get_sample_value(
        "kb_document_ocr_pages_total", {"status": "failure", "extension": "unknown"}
    )
    assert failure_total == pytest.approx(1.0)


def test_record_index_operation(isolated_metrics: CollectorRegistry) -> None:
    metrics.record_index_operation(" success ", "  weaviate  ", chunks=5, duration=2.25)
    metrics.record_index_operation("", None, chunks=0, duration=-3.0)

    success_total = isolated_metrics.get_sample_value(
        "kb_index_operations_total", {"status": "success", "backend": "weaviate"}
    )
    assert success_total == pytest.approx(1.0)

    unknown_total = isolated_metrics.get_sample_value(
        "kb_index_operations_total", {"status": "unknown", "backend": "unspecified"}
    )
    assert unknown_total == pytest.approx(1.0)

    success_duration = isolated_metrics.get_sample_value(
        "kb_index_duration_seconds_sum", {"status": "success", "backend": "weaviate"}
    )
    assert success_duration == pytest.approx(2.25)

    unknown_duration = isolated_metrics.get_sample_value(
        "kb_index_duration_seconds_sum", {"status": "unknown", "backend": "unspecified"}
    )
    assert unknown_duration == pytest.approx(0.0)

    chunk_total = isolated_metrics.get_sample_value(
        "kb_index_chunks_total", {"backend": "weaviate", "status": "success"}
    )
    assert chunk_total == pytest.approx(5.0)

    assert (
        isolated_metrics.get_sample_value(
            "kb_index_chunks_total", {"backend": "unspecified", "status": "unknown"}
        )
        is None
    )


def test_record_search_operation(isolated_metrics: CollectorRegistry) -> None:
    metrics.record_search_operation("  ui  ", " success ", duration=0.75, hits=4)
    metrics.record_search_operation(None, " failure ", duration=-1.0, hits=0)

    success_total = isolated_metrics.get_sample_value(
        "kb_search_queries_total", {"source": "ui", "status": "success"}
    )
    assert success_total == pytest.approx(1.0)

    failure_total = isolated_metrics.get_sample_value(
        "kb_search_queries_total", {"source": "unspecified", "status": "failure"}
    )
    assert failure_total == pytest.approx(1.0)

    success_duration = isolated_metrics.get_sample_value(
        "kb_search_duration_seconds_sum", {"source": "ui", "status": "success"}
    )
    assert success_duration == pytest.approx(0.75)

    failure_duration = isolated_metrics.get_sample_value(
        "kb_search_duration_seconds_sum", {"source": "unspecified", "status": "failure"}
    )
    assert failure_duration == pytest.approx(0.0)

    hits_total = isolated_metrics.get_sample_value(
        "kb_search_hits_total", {"source": "ui", "status": "success"}
    )
    assert hits_total == pytest.approx(4.0)

    assert (
        isolated_metrics.get_sample_value(
            "kb_search_hits_total", {"source": "unspecified", "status": "failure"}
        )
        is None
    )


def test_record_chat_completion(isolated_metrics: CollectorRegistry) -> None:
    metrics.record_chat_completion(" success ", duration=3.5, hits=3, citations=2)
    metrics.record_chat_completion("", duration=-0.5)

    success_total = isolated_metrics.get_sample_value(
        "kb_chat_completions_total", {"status": "success"}
    )
    assert success_total == pytest.approx(1.0)

    unknown_total = isolated_metrics.get_sample_value(
        "kb_chat_completions_total", {"status": "unknown"}
    )
    assert unknown_total == pytest.approx(1.0)

    success_duration = isolated_metrics.get_sample_value(
        "kb_chat_latency_seconds_sum", {"status": "success"}
    )
    assert success_duration == pytest.approx(3.5)

    unknown_duration = isolated_metrics.get_sample_value(
        "kb_chat_latency_seconds_sum", {"status": "unknown"}
    )
    assert unknown_duration == pytest.approx(0.0)

    hits_total = isolated_metrics.get_sample_value(
        "kb_chat_context_hits_total", {"status": "success"}
    )
    assert hits_total == pytest.approx(3.0)

    citations_total = isolated_metrics.get_sample_value(
        "kb_chat_citations_total", {"status": "success"}
    )
    assert citations_total == pytest.approx(2.0)

    assert (
        isolated_metrics.get_sample_value(
            "kb_chat_context_hits_total", {"status": "unknown"}
        )
        is None
    )
    assert (
        isolated_metrics.get_sample_value(
            "kb_chat_citations_total", {"status": "unknown"}
        )
        is None
    )
