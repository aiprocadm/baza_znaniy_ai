import boto3
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import select

from backend.app.core.config import get_settings
from backend.app.db.session import get_session_factory, session_scope
from backend.app.db.utils import init_db
from backend.app.main import create_app
from backend.app.models import Document, DocumentVersion, Pack, PackItem, Template


@mock_aws
def test_run_pack_queues_multiple_tasks(template_bytes: bytes) -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    with session_scope() as session:
        session.add_all(
            [
                Template(id="welcome", name="Welcome Letter", content=template_bytes),
                Template(id="farewell", name="Farewell Letter", content=template_bytes),
            ]
        )
        pack = Pack(name="Onboarding Pack")
        session.add(pack)
        session.flush()
        pack_id = pack.id
        session.add_all(
            [
                PackItem(
                    pack_id=pack.id,
                    template_id="welcome",
                    position=1,
                    document_name="Greeting",
                    context={"name": "Alice"},
                ),
                PackItem(
                    pack_id=pack.id,
                    template_id="farewell",
                    position=2,
                    document_name="Goodbye",
                    context={"name": "Bob"},
                ),
            ]
        )

    response = client.post("/api/v1/packs/run", json={"pack_id": pack_id})

    assert response.status_code == 202
    payload = response.json()
    assert payload["batch_id"]
    assert payload["status_url"].endswith(payload["batch_id"])

    session_factory = get_session_factory()
    with session_factory() as session:
        documents = session.execute(select(Document).order_by(Document.name)).scalars().all()
        assert len(documents) == 2
        names = {doc.name for doc in documents}
        assert names == {"Greeting", "Goodbye"}

        versions = session.execute(select(DocumentVersion)).scalars().all()
        assert len(versions) == 2
        contexts = {version.document.name: version.context for version in versions}
        assert contexts == {"Greeting": {"name": "Alice"}, "Goodbye": {"name": "Bob"}}

    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    for document_version in versions:
        s3_client.head_object(Bucket=settings.s3_bucket, Key=document_version.storage_key)
