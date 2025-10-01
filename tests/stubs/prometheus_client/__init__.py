"""Minimal stub of :mod:`prometheus_client` for unit tests."""

from __future__ import annotations

from typing import Any

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


def generate_latest() -> bytes:
    return b""


class _Metric:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def labels(self, *args: Any, **kwargs: Any) -> "_Metric":
        return self

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - no-op
        return None

    def observe(self, value: float) -> None:  # pragma: no cover - no-op
        return None


class Counter(_Metric):
    pass


class Histogram(_Metric):
    pass
