"""Minimal stub of :mod:`prometheus_client` for tests."""

from __future__ import annotations

from typing import Any


CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


def generate_latest() -> bytes:
    return b""


class _Metric:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def labels(self, *args: Any, **kwargs: Any) -> "_Metric":  # pragma: no cover - stub
        return self

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - stub
        return None

    def observe(self, value: float) -> None:  # pragma: no cover - stub
        return None


class Counter(_Metric):
    """Test double for :class:`prometheus_client.Counter`."""


class Histogram(_Metric):
    """Test double for :class:`prometheus_client.Histogram`."""


__all__ = ["Counter", "Histogram", "CONTENT_TYPE_LATEST", "generate_latest"]

