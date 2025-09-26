"""Tests for the document memory persistence layer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.memory import DocumentMemory
from app.models import DocumentCreate


def test_document_memory_persists_and_loads(tmp_path) -> None:
    storage_path = tmp_path / "documents.json"
    memory = DocumentMemory(storage_path)

    created = memory.add(DocumentCreate(content="Hello world", tags=["greeting"]))

    assert storage_path.exists(), "DocumentMemory should create the storage file on add"

    reloaded_memory = DocumentMemory(storage_path)
    loaded = reloaded_memory.get(created.id)

    assert loaded is not None
    assert loaded.content == created.content
    assert loaded.tags == created.tags
