"""/ops/health/dependencies merges retrieval degradation state."""

from __future__ import annotations

import app.api.v1.ops as ops
import app.observability.retrieval_health as retrieval_health


class _OkStore:
    def ensure_ready(self) -> None:
        return None


def test_dependencies_reports_retrieval_degradation(monkeypatch):
    retrieval_health.reset()
    monkeypatch.setattr(ops, "get_vector_store", lambda: _OkStore())
    retrieval_health.report(
        retrieval_health.RetrievalReport(
            source="fallback",
            reasons=(retrieval_health.RetrievalReason.VECTOR_BACKEND_DOWN,),
        )
    )

    result = ops.dependencies()

    assert result["checks"]["vector_store"] == "ok"
    assert result["checks"]["retrieval"] == "critical"
    assert result["status"] == "degraded"


def test_dependencies_ok_when_no_degradation(monkeypatch):
    retrieval_health.reset()
    monkeypatch.setattr(ops, "get_vector_store", lambda: _OkStore())

    result = ops.dependencies()

    assert result["checks"]["retrieval"] == "ok"
    assert result["status"] == "ok"


def test_dependencies_warning_severity_still_degrades_status(monkeypatch):
    retrieval_health.reset()
    monkeypatch.setattr(ops, "get_vector_store", lambda: _OkStore())
    retrieval_health.report(
        retrieval_health.RetrievalReport(
            source="sqlite",
            reasons=(retrieval_health.RetrievalReason.SEARCH_TRUNCATED,),
        )
    )

    result = ops.dependencies()

    assert result["checks"]["retrieval"] == "warning"
    assert result["status"] == "degraded"
