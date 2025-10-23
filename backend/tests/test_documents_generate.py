import boto3
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import select

from backend.app.core.config import get_settings
from backend.app.db.session import get_session_factory, session_scope
from backend.app.db.utils import init_db
from backend.app.main import create_app
from backend.app.models import Document, DocumentVersion, Template


@mock_aws
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
