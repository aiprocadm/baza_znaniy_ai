from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import select

from backend.app.domains.templating.renderer import _load_python_docx
from backend.app.core.config import get_settings, reset_settings_cache
from backend.app.db.session import get_session_factory, reset_engine, session_scope
from backend.app.db.utils import init_db
from backend.app.main import create_app
from backend.app.models import Document, DocumentVersion, Template
from backend.app.tasks import celery_app

DocxDocument = getattr(_load_python_docx(), "Document")


@pytest.fixture(autouse=True)
def _configure_env(monkeypatch, tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path/'documents.db'}"
    monkeypatch.setenv("BACKEND_DATABASE_URL", db_url)
    monkeypatch.setenv("BACKEND_S3_BUCKET", "test-documents")
    monkeypatch.setenv("BACKEND_S3_REGION", "us-east-1")
    monkeypatch.setenv("BACKEND_S3_ACCESS_KEY", "test")
    monkeypatch.setenv("BACKEND_S3_SECRET_KEY", "test")
    monkeypatch.setenv("BACKEND_CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("BACKEND_CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1")

    reset_settings_cache()
    reset_engine()
    celery_app.conf.update(
        broker_url=os.getenv("BACKEND_CELERY_BROKER_URL", "memory://"),
        result_backend=os.getenv("BACKEND_CELERY_RESULT_BACKEND", "cache+memory://"),
        task_always_eager=True,
        task_eager_propagates=True,
    )
    yield
    reset_engine()
    reset_settings_cache()
    celery_app.conf.update(task_always_eager=False)


@pytest.fixture()
def template_bytes() -> bytes:
    document = DocxDocument()
    document.add_paragraph("Hello {{ name }}!")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@mock_aws
@pytest.mark.usefixtures("template_bytes")
def test_generate_document(template_bytes: bytes) -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    with session_scope() as session:
        session.add(Template(id="welcome", name="Welcome Letter", content=template_bytes))

    response = client.post(
        "/api/v1/documents/generate",
        json={"template_id": "welcome", "document_name": "Greeting", "context": {"name": "Alice"}},
    )

    assert response.status_code == 202
    payload = response.json()
    assert "task_id" in payload
    assert payload["status_url"].endswith(payload["task_id"])

    session_factory = get_session_factory()
    with session_factory() as session:
        documents = session.execute(select(Document)).scalars().all()
        assert len(documents) == 1
        document = documents[0]
        assert document.name == "Greeting"

        versions = session.execute(select(DocumentVersion).where(DocumentVersion.document_id == document.id)).scalars().all()
        assert len(versions) == 1
        version = versions[0]
        assert version.version == 1
        assert version.context == {"name": "Alice"}

    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    s3_client.head_object(Bucket=settings.s3_bucket, Key=version.storage_key)
