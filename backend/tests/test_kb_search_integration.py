from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.db.utils import init_db
from backend.app.main import create_app


def test_upload_index_search_and_citations_flow() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    upload_response = client.post(
        "/api/v1/upload",
        files={"file": ("guide.md", b"onboarding checklist and team contacts", "text/markdown")},
        headers={"Authorization": "Bearer kb_tenant_admin_token"},
    )
    assert upload_response.status_code == 200
    uploaded = upload_response.json()

    search_response = client.post("/api/v1/search", json={"query": "onboarding checklist", "top_k": 3}, headers={"Authorization": "Bearer kb_tenant_admin_token"})
    assert search_response.status_code == 200
    body = search_response.json()
    assert body["total"] >= 1
    first = body["results"][0]
    assert first["source"] == uploaded["id"]
    assert "onboarding" in first["snippet"].lower()
