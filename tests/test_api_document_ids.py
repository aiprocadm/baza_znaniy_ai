"""Integration tests for API document identifiers."""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb"


def _load_service_app():
    """Load the KB service modules under an isolated package name."""

    package_name = "kb_service_app"
    package_path = SERVICE_ROOT / "app"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

    config_spec = importlib.util.spec_from_file_location(
        f"{package_name}.config", package_path / "config.py"
    )
    assert config_spec and config_spec.loader
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)
    config_module.get_settings.cache_clear()

    main_spec = importlib.util.spec_from_file_location(
        f"{package_name}.main", package_path / "main.py"
    )
    assert main_spec and main_spec.loader
    main_module = importlib.util.module_from_spec(main_spec)
    sys.modules[main_spec.name] = main_module
    main_spec.loader.exec_module(main_module)

    return main_module


def test_api_generates_url_safe_document_id(tmp_path, monkeypatch):
    """Documents with unsafe characters should get sanitized identifiers."""

    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DATA_DIR", str(storage_dir))

    main = _load_service_app()

    client = TestClient(main.app)

    response = client.post(
        "/documents",
        json={"content": "Important / document", "tags": ["safety"]},
    )

    assert response.status_code == 201
    payload = response.json()

    assert "id" in payload and payload["id"], "Document ID should be returned"
    assert "/" not in payload["id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", payload["id"])

    delete_response = client.delete(f"/documents/{payload['id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == payload["id"]

    # Ensure the document was actually removed.
    not_found = client.delete(f"/documents/{payload['id']}")
    assert not_found.status_code == 404
