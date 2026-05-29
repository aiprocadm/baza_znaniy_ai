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
