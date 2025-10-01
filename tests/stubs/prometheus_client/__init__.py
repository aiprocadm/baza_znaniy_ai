        codex/update-upload-file-handling-and-tests
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

    def _canonicalise(self, labels: dict[str, str] | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(labels, tuple):
            if len(labels) != len(self._labelnames):
                raise ValueError("Incorrect number of label values provided")
            return labels
        return tuple(labels.get(name, "") for name in self._labelnames)

    def _export_lines(self) -> list[str]:  # pragma: no cover - used only in optional endpoints
        lines = [f"# HELP {self._name} {self._documentation}", f"# TYPE {self._name} {self._type}"]
        for label_values, value in self._samples.items():
            label_str = ""
            if self._labelnames:
                pieces = [f"{key}=\"{val}\"" for key, val in zip(self._labelnames, label_values)]
                label_str = "{" + ",".join(pieces) + "}"
            lines.append(f"{self._name}{label_str} {value}")
        return lines


class _MetricChild:
    def __init__(self, metric: _MetricBase, label_values: tuple[str, ...]) -> None:
        self._metric = metric
        self._label_values = label_values


class Counter(_MetricBase):
    _type = "counter"

    def labels(self, *args: str, **kwargs: str) -> "_CounterChild":
        if args and kwargs:
            raise ValueError("Mixing args and kwargs for labels is not supported")
        if args:
            label_values = self._canonicalise(tuple(args))
        else:
            label_values = self._canonicalise(kwargs)
        return _CounterChild(self, label_values)


class _CounterChild(_MetricChild):
    def inc(self, amount: float = 1.0) -> None:
        self._metric._samples[self._label_values] = self._metric._samples.get(self._label_values, 0.0) + float(amount)


class Histogram(_MetricBase):
    _type = "histogram"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)
        self._counts: dict[tuple[str, ...], float] = defaultdict(float)

    def labels(self, *args: str, **kwargs: str) -> "_HistogramChild":
        if args and kwargs:
            raise ValueError("Mixing args and kwargs for labels is not supported")
        if args:
            label_values = self._canonicalise(tuple(args))
        else:
            label_values = self._canonicalise(kwargs)
        return _HistogramChild(self, label_values)

    def _export_lines(self) -> list[str]:  # pragma: no cover - used only in optional endpoints
        lines = []
        for label_values in self._counts:
            label_str = ""
            if self._labelnames:
                pieces = [f"{key}=\"{val}\"" for key, val in zip(self._labelnames, label_values)]
                label_str = "{" + ",".join(pieces) + "}"
            lines.append(f"{self._name}_count{label_str} {self._counts.get(label_values, 0.0)}")
            lines.append(f"{self._name}_sum{label_str} {self._sums.get(label_values, 0.0)}")
        return lines


class _HistogramChild(_MetricChild):
    def observe(self, amount: float) -> None:
        value = float(amount)
        self._metric._counts[self._label_values] = self._metric._counts.get(self._label_values, 0.0) + 1.0
        self._metric._sums[self._label_values] = self._metric._sums.get(self._label_values, 0.0) + value
        # Histograms expose both ``_count`` and ``_sum`` samples.  The base sample
        # tracks the observation count to keep the registry data consistent.
        self._metric._samples[self._label_values] = self._metric._counts[self._label_values]
        main
