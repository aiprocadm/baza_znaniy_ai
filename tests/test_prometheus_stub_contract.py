"""Contract tests for the in-tree ``prometheus_client`` stub.

The stub lives at ``tests/stubs/prometheus_client/`` and is used when the
real package is not importable. The real ``prometheus_client`` lets you
call ``.inc()``/``.dec()``/``.set()``/``.observe()`` directly on an
*unlabeled* metric (i.e. one constructed without ``labelnames``). The
production code in ``app/core/app.py`` and ``app/observability/metrics.py``
relies on this shape, so the stub must mirror it. Regressions here
silently break ~30 API tests with ``AttributeError``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

STUBS_PATH = Path(__file__).resolve().parent / "stubs"


@pytest.fixture
def stub_module(monkeypatch):
    """Force-import the stub regardless of whether real prometheus_client is installed."""

    # Remove any cached prometheus_client (real or stub) and put stubs first.
    for name in list(sys.modules):
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(STUBS_PATH))
    module = importlib.import_module("prometheus_client")
    assert "stubs" in (module.__file__ or ""), (
        "Test fixture should resolve to the stub, not the real package"
    )
    return module


def test_unlabeled_gauge_supports_inc_dec_set(stub_module):
    gauge = stub_module.Gauge("test_gauge", "doc")
    gauge.inc()
    gauge.inc(2)
    gauge.dec()
    # After +1 +2 -1 the value should be 2
    registry = stub_module._DEFAULT_REGISTRY
    assert registry.get_sample_value("test_gauge") == 2.0

    gauge.set(42.0)
    assert registry.get_sample_value("test_gauge") == 42.0


def test_unlabeled_counter_supports_inc(stub_module):
    counter = stub_module.Counter("test_counter", "doc")
    counter.inc()
    counter.inc(5)
    assert stub_module._DEFAULT_REGISTRY.get_sample_value("test_counter") == 6.0


def test_unlabeled_histogram_supports_observe(stub_module):
    hist = stub_module.Histogram("test_hist", "doc")
    hist.observe(1.5)
    hist.observe(2.5)
    registry = stub_module._DEFAULT_REGISTRY
    assert registry.get_sample_value("test_hist_count") == 2.0
    assert registry.get_sample_value("test_hist_sum") == 4.0


def test_labeled_metric_still_requires_labels_call(stub_module):
    """Labeled metrics retain the original .labels(...) discipline."""

    counter = stub_module.Counter("test_labeled", "doc", labelnames=("kind",))
    counter.labels(kind="a").inc()
    assert (
        stub_module._DEFAULT_REGISTRY.get_sample_value("test_labeled", {"kind": "a"})
        == 1.0
    )
