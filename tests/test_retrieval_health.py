"""Tests for the unified retrieval-degradation contract."""

from __future__ import annotations

from prometheus_client import REGISTRY

import app.observability.retrieval_health as rh
from app.observability.retrieval_health import (
    RetrievalReason,
    RetrievalReport,
    RetrievalSeverity,
)


def test_report_severity_critical_dominates_warning():
    rep = RetrievalReport(
        source="sqlite",
        reasons=(RetrievalReason.SEARCH_TRUNCATED, RetrievalReason.HASHING_EMBEDDER),
    )
    assert rep.degraded is True
    assert rep.severity is RetrievalSeverity.CRITICAL


def test_clean_report_is_not_degraded():
    rep = RetrievalReport(source="vector")
    assert rep.degraded is False
    assert rep.severity is RetrievalSeverity.OK


def test_report_then_snapshot_lists_active_reason():
    rh.reset()
    rh.report(
        RetrievalReport(
            source="fallback",
            reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,),
            detail="boom",
        )
    )
    snap = rh.snapshot()
    assert snap["degraded"] is True
    assert snap["severity"] == "critical"
    assert snap["reasons"][0]["reason"] == "vector_backend_down"
    assert snap["reasons"][0]["detail"] == "boom"


def test_clean_run_clears_that_sources_reasons():
    rh.reset()
    rh.report(RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,)))
    rh.report(RetrievalReport(source="vector"))  # clean run
    assert rh.snapshot()["degraded"] is False


def test_current_report_returns_last_report_for_this_context():
    rh.reset()
    rep = RetrievalReport(source="sqlite", reasons=(RetrievalReason.HASHING_EMBEDDER,))
    rh.report(rep)
    assert rh.current_report() is rep


def test_ttl_backstop_drops_stale_reason(monkeypatch):
    rh.reset()
    clock = {"t": 1000.0}
    monkeypatch.setattr(rh.time, "monotonic", lambda: clock["t"])
    rh.report(RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,)))
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is True
    clock["t"] = 1000.0 + 301.0
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is False


def test_snapshot_includes_extra_active_probes():
    rh.reset()
    snap = rh.snapshot(extra=((RetrievalReason.HASHING_EMBEDDER, "embedder=hash"),))
    assert snap["degraded"] is True
    assert snap["reasons"][0]["reason"] == "hashing_embedder"


def test_gauge_set_on_report_and_cleared_on_clean_run():
    rh.reset()
    rh.report(RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,)))
    assert (
        REGISTRY.get_sample_value(
            "kb_retrieval_degraded",
            {"reason": "vector_backend_down", "severity": "critical"},
        )
        == 1.0
    )
    rh.report(RetrievalReport(source="vector"))  # clean run clears governed reason
    assert (
        REGISTRY.get_sample_value(
            "kb_retrieval_degraded",
            {"reason": "vector_backend_down", "severity": "critical"},
        )
        == 0.0
    )


def test_snapshot_does_not_crash_on_unmapped_reason(monkeypatch):
    rh.reset()
    # A reason present in the registry but absent from _SEVERITY (e.g. a future
    # RetrievalReason added before its severity mapping) must not crash snapshot().
    monkeypatch.setitem(rh._REGISTRY, rh.RetrievalReason.HEALTHY, (rh.time.monotonic(), "x"))
    snap = rh.snapshot()
    assert isinstance(snap["reasons"], list)


def test_report_payload_none_when_missing_or_clean():
    assert rh.report_payload(None) is None
    assert rh.report_payload(RetrievalReport(source="sqlite")) is None


def test_report_payload_mirrors_degraded_report():
    rep = RetrievalReport(
        source="sqlite",
        reasons=(RetrievalReason.HASHING_EMBEDDER, RetrievalReason.SEARCH_TRUNCATED),
        detail="embedder=hash",
    )
    payload = rh.report_payload(rep)
    assert payload["degraded"] is True
    assert payload["severity"] == "critical"  # CRITICAL dominates WARNING
    by_reason = {r["reason"]: r for r in payload["reasons"]}
    assert by_reason["hashing_embedder"]["severity"] == "critical"
    assert by_reason["search_truncated"]["severity"] == "warning"
    assert by_reason["hashing_embedder"]["detail"] == "embedder=hash"
