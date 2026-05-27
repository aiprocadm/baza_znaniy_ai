from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import uuid4

from typing import Any


@dataclass
class ReindexResult:
    job_id: str
    status: str
    copied: int
    alias: str
    source_collection: str
    temp_collection: str
    dry_run: bool = False


logger = logging.getLogger(__name__)


class ReindexService:
    """Safe reindex flow with temporary collection + atomic alias switch."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def reindex_document(
        self, *, document_id: str, idempotency_key: str | None = None, dry_run: bool = False
    ) -> ReindexResult:
        alias = self.store.settings.qdrant_collection
        source_collection = self.store.resolve_collection_name(alias)
        job_id = f"reindex-{uuid4().hex[:12]}"
        temp_collection = f"{alias}__tmp__{job_id}"

        logger.info(
            "reindex stage=start document_id=%s idempotency_key=%s source_collection=%s temp_collection=%s dry_run=%s",
            document_id,
            idempotency_key,
            source_collection,
            temp_collection,
            dry_run,
        )
        self.store.create_collection_like(temp_collection, source_collection)
        copied = 0
        switched = False
        try:
            logger.info(
                "reindex stage=copy document_id=%s idempotency_key=%s", document_id, idempotency_key
            )
            for payload in self.store.export_payloads_from_collection(source_collection):
                if str(payload.get("document_id") or "") != str(document_id):
                    continue
                self.store.import_payloads_to_collection(temp_collection, [payload])
                copied += 1

            if copied == 0:
                raise ValueError("No vectors found for document")

            logger.info(
                "reindex stage=verify document_id=%s copied=%s idempotency_key=%s",
                document_id,
                copied,
                idempotency_key,
            )
            self.store.validate_collection_not_empty(temp_collection)
            if dry_run:
                logger.info(
                    "reindex stage=rollback document_id=%s reason=dry_run idempotency_key=%s",
                    document_id,
                    idempotency_key,
                )
                self.store.delete_collection_safe(temp_collection)
                return ReindexResult(
                    job_id=job_id,
                    status="dry_run",
                    copied=copied,
                    alias=alias,
                    source_collection=source_collection,
                    temp_collection=temp_collection,
                    dry_run=True,
                )
            logger.info(
                "reindex stage=switch document_id=%s idempotency_key=%s",
                document_id,
                idempotency_key,
            )
            self.store.switch_alias(alias, temp_collection)
            switched = True
            return ReindexResult(
                job_id=job_id,
                status="completed",
                copied=copied,
                alias=alias,
                source_collection=source_collection,
                temp_collection=temp_collection,
                dry_run=False,
            )
        except Exception:
            logger.exception(
                "reindex stage=rollback document_id=%s idempotency_key=%s switched=%s",
                document_id,
                idempotency_key,
                switched,
            )
            if not switched:
                self.store.delete_collection_safe(temp_collection)
            raise
