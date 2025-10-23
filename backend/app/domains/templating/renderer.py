from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from importlib import import_module
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


def _load_python_docx() -> ModuleType:
    existing = sys.modules.get("docx")
    if existing is not None and "site-packages" in (getattr(existing, "__file__", "") or ""):
        return existing

    try:
        distribution = importlib.metadata.distribution("python-docx")
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover - should not happen
        raise RuntimeError("python-docx must be installed") from exc

    package_dir = Path(distribution.locate_file("docx"))
    search_path = str(package_dir.parent)

    saved_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "docx" or name.startswith("docx.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, search_path)
    try:
        module = import_module("docx")
        submodules = {
            name: sys.modules[name]
            for name in list(sys.modules)
            if name.startswith("docx.")
        }
        module.__dict__["__submodules__"] = submodules
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "docx" or name.startswith("docx."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
    return module


def render_docx(template_bytes: bytes, context: Mapping[str, Any]) -> bytes:
    """Render a DOCX template with the provided context using docxtpl."""

    python_docx = _load_python_docx()
    previous_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "docx" or name.startswith("docx.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.modules["docx"] = python_docx
    for name, module in python_docx.__dict__.get("__submodules__", {}).items():
        sys.modules[name] = module
    try:
        DocxTemplate = import_module("docxtpl").DocxTemplate

        buffer = BytesIO(template_bytes)
        template = DocxTemplate(buffer)
        template.render(context)
        output = BytesIO()
        template.save(output)
        return output.getvalue()
    finally:
        for name in list(sys.modules):
            if name == "docx" or name.startswith("docx."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
