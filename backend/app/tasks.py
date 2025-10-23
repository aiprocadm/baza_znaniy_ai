from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy import func, select

from backend.app.core.config import get_settings
from backend.app.db.session import session_scope
from backend.app.domains.templating.renderer import render_docx
from backend.app.models import Document, DocumentVersion, Template
from backend.app.services.storage import S3DocumentStorage

settings = get_settings()

celery_app = Celery(
    "backend.app",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.conf.task_always_eager = settings.celery_task_eager
celery_app.conf.task_eager_propagates = True

logger = get_task_logger(__name__)
logging.getLogger("boto3").setLevel(logging.WARNING)


@celery_app.task(name="generate_document_task", bind=True)
def generate_document_task(
    self,
    *,
    template_id: str,
    context: dict[str, Any],
    document_name: str | None = None,
) -> dict[str, Any]:
    storage = S3DocumentStorage.from_settings()
    with session_scope() as session:
        template = session.get(Template, template_id)
        if template is None:
            msg = f"Template {template_id} not found"
            logger.error(msg)
            raise ValueError(msg)

        rendered = render_docx(template.content, context)
        name = document_name or template.name
        document = Document(template_id=template.id, name=name)
        session.add(document)
        session.flush()

        last_version = session.execute(
            select(func.max(DocumentVersion.version)).where(DocumentVersion.document_id == document.id)
        ).scalar()
        next_version = 1 if last_version is None else last_version + 1

        storage_key = storage.upload(f"documents/{document.id}/v{next_version}.docx", rendered)

        version = DocumentVersion(
            document_id=document.id,
            version=next_version,
            storage_key=storage_key,
            context=dict(context),
        )
        session.add(version)
        session.flush()

        logger.info(
            "Document generated",
            extra={
                "document_id": document.id,
                "document_version_id": version.id,
                "storage_key": storage_key,
            },
        )

        return {
            "document_id": document.id,
            "document_version_id": version.id,
            "storage_key": storage_key,
        }
