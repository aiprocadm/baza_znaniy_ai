from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import reset_settings_cache
from backend.app.db.session import reset_engine
from backend.app.domains.templating.renderer import _load_python_docx
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
