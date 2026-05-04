from __future__ import annotations

import pytest

from app.services.reindex_service import ReindexService


class _Store:
    class settings:
        qdrant_collection = "kb"

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.switched: list[tuple[str, str]] = []

    def resolve_collection_name(self, alias_name: str) -> str:
        return "kb_v1"

    def create_collection_like(self, target_collection: str, source_collection: str) -> None:
        self.target = target_collection

    def export_payloads_from_collection(self, source_collection: str):
        yield {"id": "1", "document_id": "42", "text": "ok", "vector": [0.1, 0.2]}

    def import_payloads_to_collection(self, collection_name: str, payloads):
        raise RuntimeError("mid-reindex failure")

    def validate_collection_not_empty(self, collection_name: str) -> None:
        return None

    def switch_alias(self, alias_name: str, collection_name: str) -> None:
        self.switched.append((alias_name, collection_name))

    def delete_collection_safe(self, collection_name: str) -> None:
        self.deleted.append(collection_name)


def test_reindex_rolls_back_temp_collection_on_mid_failure() -> None:
    store = _Store()
    service = ReindexService(store)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="mid-reindex failure"):
        service.reindex_document(document_id="42")

    assert len(store.deleted) == 1
    assert store.switched == []
