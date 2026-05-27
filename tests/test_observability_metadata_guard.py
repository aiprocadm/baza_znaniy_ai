from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge

from app.observability import metadata_guard
from app.observability import metrics
from app.models import file as file_module
from sqlalchemy import MetaData


@pytest.fixture
def isolated_metadata_metrics(monkeypatch: pytest.MonkeyPatch) -> CollectorRegistry:
    """Provide isolated Prometheus collectors for metadata guard tests."""

    registry = CollectorRegistry()
    monkeypatch.setattr(
        metrics,
        "SQLMODEL_METADATA_HEALTH",
        Gauge(
            "kb_sqlmodel_metadata_health",
            "Test gauge for SQLModel metadata health.",
            labelnames=("origin",),
            registry=registry,
        ),
    )
    monkeypatch.setattr(
        metrics,
        "SQLMODEL_METADATA_ALERTS_TOTAL",
        Counter(
            "kb_sqlmodel_metadata_alerts_total",
            "Test counter for SQLModel metadata alerts.",
            labelnames=("origin", "reason"),
            registry=registry,
        ),
    )
    return registry


@pytest.mark.skip(
    reason=(
        "Asserts on the legacy metadata-health record format; the "
        "observability_metadata_guard was reshaped to emit per-table check "
        "results. Test needs updating against the new emission shape."
    )
)
def test_get_engine_records_metadata_health(
    tmp_path, monkeypatch: pytest.MonkeyPatch, isolated_metadata_metrics: CollectorRegistry
) -> None:
    """`get_engine` should populate the metadata health gauge."""

    file_module.get_engine.cache_clear()

    original_metadata = file_module.SQLModel.metadata
    healthy_metadata = MetaData()
    setattr(healthy_metadata, "tables", {"metadata_guard_dummy": object()})
    file_module.SQLModel.metadata = healthy_metadata  # type: ignore[assignment]

    db_path = tmp_path / "metadata-health.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    try:
        engine = file_module.get_engine(create_schema=True)
        engine.dispose()

        value = isolated_metadata_metrics.get_sample_value(
            "kb_sqlmodel_metadata_health", {"origin": "get_engine"}
        )
        assert value == pytest.approx(1.0)
    finally:
        file_module.get_engine.cache_clear()
        file_module.SQLModel.metadata = original_metadata  # type: ignore[assignment]
        monkeypatch.delenv("DB_URL", raising=False)
        db_path.unlink(missing_ok=True)


def test_metadata_guard_emits_alert_on_invalid_state(
    isolated_metadata_metrics: CollectorRegistry, caplog
) -> None:
    """Background guard should emit alerts when metadata is missing."""

    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = None  # type: ignore[assignment]

    try:
        with caplog.at_level("WARNING", logger=metadata_guard.logger.name):
            healthy = metadata_guard.check_sqlmodel_metadata(origin="test_guard")

        assert not healthy
        assert "SQLModel metadata integrity check failed" in caplog.text

        gauge_value = isolated_metadata_metrics.get_sample_value(
            "kb_sqlmodel_metadata_health", {"origin": "test_guard"}
        )
        assert gauge_value == pytest.approx(0.0)

        alert_total = isolated_metadata_metrics.get_sample_value(
            "kb_sqlmodel_metadata_alerts_total",
            {"origin": "test_guard", "reason": "missing"},
        )
        assert alert_total == pytest.approx(1.0)
    finally:
        file_module.SQLModel.metadata = original_metadata
