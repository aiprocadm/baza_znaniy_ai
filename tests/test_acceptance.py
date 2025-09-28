"""End-to-end smoke test that covers demo document ingestion."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from tests.demo_assets import ensure_demo_assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"
SAMPLE_FILES_DIR = PROJECT_ROOT / "srv" / "projects" / "kb" / "data" / "files"


def _load_ingest_module():
    package_name = "kb_service_acceptance_ingest"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    ingest_spec = importlib.util.spec_from_file_location(
        f"{package_name}.ingest", SERVICE_ROOT / "ingest.py"
    )
    assert ingest_spec and ingest_spec.loader
    ingest_module = importlib.util.module_from_spec(ingest_spec)
    sys.modules[ingest_spec.name] = ingest_module
    ingest_spec.loader.exec_module(ingest_module)
    return ingest_module


ingest = _load_ingest_module()
parse_and_chunk = ingest.parse_and_chunk


def test_sample_demo_documents_are_ingestable():
    ensure_demo_assets(SAMPLE_FILES_DIR)

    sample_files = sorted(SAMPLE_FILES_DIR.glob("demo_*"))
    assert sample_files, "demo assets should be generated"

    extensions = {path.suffix for path in sample_files}
    assert extensions == {".pdf", ".docx", ".txt"}

    for path in sample_files:
        data = path.read_bytes()
        chunks = parse_and_chunk(path.name, data)

        assert chunks, f"{path.name} should produce chunks"
        assert all(chunk["file"] == path.name for chunk in chunks)
        assert all(isinstance(chunk["text"], str) and chunk["text"] for chunk in chunks)
        assert all(isinstance(chunk["page"], int) and chunk["page"] >= 1 for chunk in chunks)
