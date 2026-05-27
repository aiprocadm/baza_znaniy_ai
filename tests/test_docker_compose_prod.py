from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


def _docker_compose_cmd() -> list[str] | None:
    if shutil.which("docker") is None:
        return None
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    return ["docker", "compose"] if probe.returncode == 0 else None


def test_docker_compose_prod_boots_all_services() -> None:
    compose = _docker_compose_cmd()
    if compose is None:
        pytest.skip("docker compose is not available in this environment")

    cmd = compose + ["-f", "docker-compose.prod.yml", "config", "--services"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    services = set(out.stdout.split())

    assert {"qdrant", "kb_api", "kb_worker", "prometheus", "grafana"}.issubset(services)


def test_docker_compose_prod_metrics_targets_present() -> None:
    """Prometheus must be configured to scrape both kb_api and qdrant on /metrics."""

    prometheus_yml = Path("ops/prometheus/prometheus.yml")
    assert prometheus_yml.is_file(), f"missing {prometheus_yml}"

    config = yaml.safe_load(prometheus_yml.read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in config.get("scrape_configs", [])}

    for job_name, expected_target in (("api", "kb_api:8000"), ("qdrant", "qdrant:6333")):
        assert job_name in jobs, f"prometheus job {job_name!r} missing"
        job = jobs[job_name]
        assert job.get("metrics_path") == "/metrics"
        targets = {t for static in job.get("static_configs", []) for t in static.get("targets", [])}
        assert (
            expected_target in targets
        ), f"prometheus job {job_name!r} should scrape {expected_target}"
