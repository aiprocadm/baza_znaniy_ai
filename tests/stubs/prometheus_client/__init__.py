"""Lightweight Prometheus client stub used for unit testing."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

__all__ = [
    "CollectorRegistry",
    "Counter",
    "Histogram",
    "CONTENT_TYPE_LATEST",
    "generate_latest",
]


CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class CollectorRegistry:
    """Minimal registry storing metric samples for assertions."""

    def __init__(self) -> None:
        self._metrics: dict[str, _MetricBase] = {}

    def register(self, metric: "_MetricBase") -> None:
        self._metrics[metric._name] = metric
        metric._registry = self

    def get_sample_value(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        labels = labels or {}
        if name.endswith("_sum"):
            base = name[: -len("_sum")]
            metric = self._metrics.get(base)
            if isinstance(metric, Histogram):
                return metric._sums.get(metric._canonicalise(labels))
            return None
        if name.endswith("_count"):
            base = name[: -len("_count")]
            metric = self._metrics.get(base)
            if isinstance(metric, Histogram):
                return metric._counts.get(metric._canonicalise(labels))
            return None
        metric = self._metrics.get(name)
        if metric is None:
            return None
        return metric._samples.get(metric._canonicalise(labels))


_DEFAULT_REGISTRY = CollectorRegistry()


def generate_latest(registry: CollectorRegistry | None = None) -> bytes:
    """Return a very small payload suitable for tests."""

    registry = registry or _DEFAULT_REGISTRY
    lines: list[str] = []
    for metric in registry._metrics.values():
        lines.extend(metric._export_lines())
    return "\n".join(lines).encode("utf-8")


class _MetricBase:
    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: Iterable[str] | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._name = name
        self._documentation = documentation
        self._labelnames = tuple(labelnames or ())
        self._registry: CollectorRegistry | None = None
        self._samples: dict[tuple[str, ...], float] = defaultdict(float)
        if registry is None:
            _DEFAULT_REGISTRY.register(self)
        else:
            registry.register(self)

    # ``*args`` support is provided for compatibility with the real client.
    def labels(self, *args: str, **kwargs: str) -> "_MetricChild":  # pragma: no cover - overwritten
        raise NotImplementedError

    def _canonicalise(self, labels: dict[str, str]) -> tuple[str, ...]:
        return tuple(labels.get(name, "") for name in self._labelnames)

    def _export_lines(self) -> list[str]:  # pragma: no cover - seldom used in tests
        return []


class _MetricChild:
    def __init__(self, metric: "_MetricBase", labels: tuple[str, ...]) -> None:
        self._metric = metric
        self._labels = labels

    def inc(self, amount: float = 1.0) -> None:
        self._metric._samples[self._labels] += amount

    def observe(self, value: float) -> None:
        self._metric._samples[self._labels] += value
        if isinstance(self._metric, Histogram):
            self._metric._sums[self._labels] += value
            self._metric._counts[self._labels] += 1.0


class Counter(_MetricBase):
    def labels(self, *args: str, **kwargs: str) -> _MetricChild:
        labels = self._canonicalise(dict(zip(self._labelnames, args, strict=False)) | kwargs)
        return _MetricChild(self, labels)


class Histogram(_MetricBase):
    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: Iterable[str] | None = None,
        registry: CollectorRegistry | None = None,
        buckets: Iterable[float] | None = None,  # pragma: no cover - compatibility only
    ) -> None:
        super().__init__(name, documentation, labelnames=labelnames, registry=registry)
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)
        self._counts: dict[tuple[str, ...], float] = defaultdict(float)
        self._buckets = tuple(buckets or ())

    def labels(self, *args: str, **kwargs: str) -> _MetricChild:
        labels = self._canonicalise(dict(zip(self._labelnames, args, strict=False)) | kwargs)
        return _MetricChild(self, labels)

    def _export_lines(self) -> list[str]:  # pragma: no cover - compatibility only
        lines: list[str] = []
        for labels, value in self._samples.items():
            label_str = ",".join(f"{name}=\"{label}\"" for name, label in zip(self._labelnames, labels, strict=False) if label)
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"# HELP {self._name} {self._documentation}")
            lines.append(f"# TYPE {self._name} histogram")
            lines.append(f"{self._name}_sum{suffix} {self._sums[labels]}")
            lines.append(f"{self._name}_count{suffix} {self._counts[labels]}")
            for bucket in self._buckets:
                lines.append(f"{self._name}_bucket{suffix},le=\"{bucket}\" {value}")
        return lines


class _HistogramChild(_MetricChild):
    def observe(self, amount: float) -> None:
        value = float(amount)
        self._metric._counts[self._label_values] = self._metric._counts.get(self._label_values, 0.0) + 1.0
        self._metric._sums[self._label_values] = self._metric._sums.get(self._label_values, 0.0) + value
        # Histograms expose both ``_count`` and ``_sum`` samples.  The base sample tracks
        # the observation count to keep the registry data consistent.
        self._metric._samples[self._label_values] = self._metric._counts[self._label_values]

