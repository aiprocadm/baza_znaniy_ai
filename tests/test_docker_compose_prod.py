from __future__ import annotations

import shutil
import subprocess

import pytest


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
    compose = _docker_compose_cmd()
    if compose is None:
        pytest.skip("docker compose is not available in this environment")

    cmd = compose + ["-f", "docker-compose.prod.yml", "config"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rendered = out.stdout

    assert "kb_api:8000" in rendered
    assert "qdrant:6333" in rendered
    assert "/metrics" in rendered
