"""Tests for the document memory persistence layer."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Tuple

import pytest

from app.memory import DocumentCreate as LegacyDocumentCreate
from app.memory import DocumentMemory as LegacyDocumentMemory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb"


def _load_service_memory() -> Tuple[Type, Type]:
    package_name = "kb_service_memory_tests"
    package_path = SERVICE_ROOT / "app"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

    models_spec = importlib.util.spec_from_file_location(
        f"{package_name}.models", package_path / "models.py"
    )
    assert models_spec and models_spec.loader
    models_module = importlib.util.module_from_spec(models_spec)
    sys.modules[models_spec.name] = models_module
    models_spec.loader.exec_module(models_module)

    memory_spec = importlib.util.spec_from_file_location(
        f"{package_name}.memory", package_path / "memory.py"
    )
    assert memory_spec and memory_spec.loader
    memory_module = importlib.util.module_from_spec(memory_spec)
    sys.modules[memory_spec.name] = memory_module
    memory_spec.loader.exec_module(memory_module)

    return memory_module.DocumentMemory, models_module.DocumentCreate


ServiceDocumentMemory, ServiceDocumentCreate = _load_service_memory()


@pytest.mark.parametrize(
    "memory_cls, payload_cls",
    [
        (LegacyDocumentMemory, LegacyDocumentCreate),
        (ServiceDocumentMemory, ServiceDocumentCreate),
    ],
)
def test_document_memory_persists_and_loads(tmp_path: Path, memory_cls: type, payload_cls: type) -> None:
    storage_path = tmp_path / f"{memory_cls.__module__.replace('.', '_')}_documents.json"
    memory = memory_cls(storage_path)

    created = memory.add(payload_cls(content="Hello world", tags=["greeting"]))

    assert storage_path.exists(), "DocumentMemory should create the storage file on add"

    reloaded_memory = memory_cls(storage_path)
    loaded = reloaded_memory.get(created.id)

    assert loaded is not None
    assert loaded.content == created.content
    assert loaded.tags == created.tags


@pytest.mark.parametrize(
    "memory_cls",
    [LegacyDocumentMemory, ServiceDocumentMemory],
)
def test_document_memory_recovers_from_invalid_json(tmp_path: Path, memory_cls: type) -> None:
    storage_path = tmp_path / "documents.json"
    storage_path.write_text("{not: valid json}", encoding="utf-8")

    memory = memory_cls(storage_path)

    assert memory.all() == []
    with storage_path.open(encoding="utf-8") as handle:
        content = handle.read().strip()
    assert content.startswith("[")


@pytest.mark.parametrize(
    "memory_cls, payload_cls",
    [
        (LegacyDocumentMemory, LegacyDocumentCreate),
        (ServiceDocumentMemory, ServiceDocumentCreate),
    ],
)
def test_document_ids_remain_unique_after_deletion(
    tmp_path: Path, memory_cls: type, payload_cls: type
) -> None:
    storage_path = tmp_path / f"{memory_cls.__module__.replace('.', '_')}_unique.json"
    memory = memory_cls(storage_path)

    first = memory.add(payload_cls(content="Same prefix", tags=[]))
    second = memory.add(payload_cls(content="Same prefix", tags=[]))

    assert first.id != second.id

    assert memory.remove(first.id)

    third = memory.add(payload_cls(content="Same prefix", tags=[]))

    assert third.id not in {first.id, second.id}

    remaining = memory.get(second.id)
    assert remaining is not None
    assert remaining.content == "Same prefix"
