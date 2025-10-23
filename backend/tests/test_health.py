from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.db.utils import init_db
from backend.app.main import create_app


def test_health_endpoint_returns_ok() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_returns_ready() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
