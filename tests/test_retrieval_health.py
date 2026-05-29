"""Tests for the unified retrieval-degradation contract."""

from __future__ import annotations

import pytest
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
    rh.report(
        RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,))
    )
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
    rh.report(
        RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,))
    )
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is True
    clock["t"] = 1000.0 + 301.0
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is False


def test_snapshot_includes_extra_active_probes():
    rh.reset()
    snap = rh.snapshot(extra=((RetrievalReason.HASHING_EMBEDDER, "embedder=hash"),))
    assert snap["degraded"] is True
    assert snap["reasons"][0]["reason"] == "hashing_embedder"
