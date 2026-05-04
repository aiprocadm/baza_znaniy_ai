from __future__ import annotations

from dataclasses import dataclass
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


class ReindexService:
    """Safe reindex flow with temporary collection + atomic alias switch."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def reindex_document(self, *, document_id: str) -> ReindexResult:
        alias = self.store.settings.qdrant_collection
        source_collection = self.store.resolve_collection_name(alias)
        job_id = f"reindex-{uuid4().hex[:12]}"
        temp_collection = f"{alias}__tmp__{job_id}"

        self.store.create_collection_like(temp_collection, source_collection)
        copied = 0
        switched = False
        try:
            for payload in self.store.export_payloads_from_collection(source_collection):
                if str(payload.get("document_id") or "") != str(document_id):
                    continue
                self.store.import_payloads_to_collection(temp_collection, [payload])
                copied += 1

            if copied == 0:
                raise ValueError("No vectors found for document")

            self.store.validate_collection_not_empty(temp_collection)
            self.store.switch_alias(alias, temp_collection)
            switched = True
            return ReindexResult(
                job_id=job_id,
                status="completed",
                copied=copied,
                alias=alias,
                source_collection=source_collection,
                temp_collection=temp_collection,
            )
        except Exception:
            if not switched:
                self.store.delete_collection_safe(temp_collection)
            raise
