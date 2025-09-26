"""Tests for the document memory persistence layer."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from app.memory import DocumentMemory as LegacyDocumentMemory
from app.models import DocumentCreate as LegacyDocumentCreate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb"


def _load_service_memory() -> tuple[type, type]:
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
def test_document_memory_persists_and_loads(tmp_path, memory_cls, payload_cls) -> None:
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
    "memory_cls, payload_cls",
    [
        (LegacyDocumentMemory, LegacyDocumentCreate),
        (ServiceDocumentMemory, ServiceDocumentCreate),
    ],
)
def test_document_ids_remain_unique_after_deletion(tmp_path, memory_cls, payload_cls) -> None:
    storage_path = tmp_path / f"{memory_cls.__module__.replace('.', '_')}_unique.json"
    memory = memory_cls(storage_path)

    first = memory.add(payload_cls(content="Same prefix", tags=[]))
    second = memory.add(payload_cls(content="Same prefix", tags=[]))

    assert first.id != second.id

    assert memory.remove(first.id)

    third = memory.add(payload_cls(content="Same prefix", tags=[]))

    assert third.id not in {first.id, second.id}

    # Ensure the second document still exists and was not overwritten.
    remaining = memory.get(second.id)
    assert remaining is not None
    assert remaining.content == "Same prefix"
