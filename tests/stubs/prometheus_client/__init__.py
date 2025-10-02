"""Minimal Prometheus client stub tailored for unit tests."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, MutableMapping, TypeVar

__all__ = [
    "CollectorRegistry",
    "Counter",
    "Histogram",
    "CONTENT_TYPE_LATEST",
    "generate_latest",
]

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

LabelValues = tuple[str, ...]
MetricSamples = MutableMapping[LabelValues, float]


class CollectorRegistry:
    """Registry that stores metric instances for lookup during tests."""

    def __init__(self) -> None:
        self._metrics: dict[str, _MetricBase] = {}

    def register(self, metric: "_MetricBase") -> None:
        self._metrics[metric._name] = metric
        metric._registry = self

    def get_sample_value(
        self, name: str, labels: Mapping[str, str] | None = None
    ) -> float | None:
        labels = labels or {}
        if name.endswith("_sum"):
            metric_name = name[: -len("_sum")]
            metric = self._metrics.get(metric_name)
            if isinstance(metric, Histogram):
                key = metric._canonicalise(labels)
                return metric._sums.get(key)
            return None
        if name.endswith("_count"):
            metric_name = name[: -len("_count")]
            metric = self._metrics.get(metric_name)
            if isinstance(metric, Histogram):
                key = metric._canonicalise(labels)
                return metric._counts.get(key)
            return None
        metric = self._metrics.get(name)
        if metric is None:
            return None
        key = metric._canonicalise(labels)
        return metric._samples.get(key)


_DEFAULT_REGISTRY = CollectorRegistry()


def generate_latest(registry: CollectorRegistry | None = None) -> bytes:
    """Render the stored samples in Prometheus exposition format."""

    registry = registry or _DEFAULT_REGISTRY
    lines: list[str] = []
    for metric in registry._metrics.values():
        lines.extend(metric._export_lines())
    return "\n".join(lines).encode("utf-8")


_ChildT = TypeVar("_ChildT", bound="_MetricChild")


class _MetricBase:
    """Shared behaviour for Counter and Histogram test doubles."""

    _type: str = ""
    _child_type: type[_MetricChild]

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
        self._samples: MetricSamples = defaultdict(float)
        registry = registry or _DEFAULT_REGISTRY
        registry.register(self)

    def labels(self, *args: str, **kwargs: str) -> _MetricChild:
        if len(args) > len(self._labelnames):  # pragma: no cover - defensive
            raise ValueError("Too many positional label values provided")
        provided = dict(zip(self._labelnames, args))
        provided.update(kwargs)
        values = tuple(provided.get(name, "") for name in self._labelnames)
        return self._child_type(self, values)

    def _canonicalise(self, labels: Mapping[str, str]) -> LabelValues:
        return tuple(labels.get(name, "") for name in self._labelnames)

    def _format_labels(self, values: LabelValues) -> str:
        if not self._labelnames:
            return ""
        parts = [
            f'{name}="{value}"'
            for name, value in zip(self._labelnames, values, strict=False)
            if value
        ]
        return f"{{{','.join(parts)}}}" if parts else ""

    def _export_lines(self) -> list[str]:  # pragma: no cover - compatibility helper
        if not self._samples:
            return []
        lines = [
            f"# HELP {self._name} {self._documentation}",
            f"# TYPE {self._name} {self._type}",
        ]
        for labels, value in sorted(self._samples.items()):
            lines.append(f"{self._name}{self._format_labels(labels)} {value}")
        return lines


class _MetricChild:
    def __init__(self, metric: _MetricBase, label_values: LabelValues) -> None:
        self._metric = metric
        self._label_values = label_values

    def inc(self, amount: float = 1.0) -> None:
        self._metric._samples[self._label_values] += float(amount)

    def observe(self, value: float) -> None:
        self._metric._samples[self._label_values] += float(value)


class Counter(_MetricBase):
    _type = "counter"
    _child_type = _MetricChild


class _HistogramChild(_MetricChild):
    def observe(self, value: float) -> None:
        observation = float(value)
        metric = self._metric
        metric._counts[self._label_values] += 1.0
        metric._sums[self._label_values] += observation
        metric._samples[self._label_values] = metric._counts[self._label_values]


class Histogram(_MetricBase):
    _type = "histogram"
    _child_type = _HistogramChild

    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: Iterable[str] | None = None,
        registry: CollectorRegistry | None = None,
        buckets: Iterable[float] | None = None,  # pragma: no cover - compatibility
    ) -> None:
        super().__init__(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
        )
        self._counts: MetricSamples = defaultdict(float)
        self._sums: MetricSamples = defaultdict(float)
        self._buckets = tuple(buckets or ())

    def labels(self, *args: str, **kwargs: str) -> _HistogramChild:
        return super().labels(*args, **kwargs)  # type: ignore[return-value]

    def _export_lines(self) -> list[str]:  # pragma: no cover - compatibility helper
        if not self._counts:
            return []
        lines = [
            f"# HELP {self._name} {self._documentation}",
            f"# TYPE {self._name} histogram",
        ]
        for labels in sorted(self._counts):
            suffix = self._format_labels(labels)
            lines.append(f"{self._name}_count{suffix} {self._counts[labels]}")
            lines.append(f"{self._name}_sum{suffix} {self._sums[labels]}")
        return lines
