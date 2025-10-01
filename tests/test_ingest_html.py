"""Unit tests for HTML ingestion helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest


def _load_html_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "app" / "ingest" / "html.py"
    spec = importlib.util.spec_from_file_location("app.ingest.html_test", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to load html module for tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


html_module = _load_html_module()


def _settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "html2text_bodywidth": 0,
        "html2text_links": False,
        "html2text_ignore_images": True,
        "html2text_ignore_emphasis": True,
        "html2text_inline_links": False,
        "html2text_single_line_break": False,
        "html2text_wrap_links": True,
        "html2text_unicode_snob": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_html_to_text_sections_fallback_without_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(html_module, "html2text", None)

    html = "<html><body><h1>Heading</h1><p>Paragraph</p><div>Another block</div></body></html>"
    sections = html_module.html_to_text_sections(html)

    assert sections == ["Heading", "Paragraph", "Another block"]


def test_html_to_text_sections_configures_html2text(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: List[object] = []

    class DummyHTML2Text:
        def __init__(self) -> None:
            self.body_width = None
            self.ignore_links = None
            self.ignore_images = None
            self.ignore_emphasis = None
            self.inline_links = None
            self.single_line_break = None
            self.wrap_links = None
            self.unicode_snob = None
            instances.append(self)

        def handle(self, _: str) -> str:
            parts = ["Heading"]
            link_fragment = "Example https://example.com" if not self.ignore_links else "Example"
            parts.append(link_fragment)
            if not self.ignore_images:
                parts.append("Image ALT text")
            return "\n\n".join(parts)

    monkeypatch.setattr(
        html_module,
        "html2text",
        SimpleNamespace(HTML2Text=DummyHTML2Text),
    )

    settings = _settings(
        html2text_bodywidth=42,
        html2text_links=True,
        html2text_ignore_images=False,
        html2text_ignore_emphasis=False,
        html2text_inline_links=True,
        html2text_single_line_break=True,
        html2text_wrap_links=False,
        html2text_unicode_snob=True,
    )

    sections = html_module.html_to_text_sections("<p>Example</p>", settings=settings)

    assert sections[0] == "Heading"
    assert "https://example.com" in " ".join(sections)
    assert "ALT text" in " ".join(sections)

    assert instances, "converter should have been instantiated"
    converter = instances[-1]
    assert converter.body_width == 42
    assert converter.ignore_links is False
    assert converter.ignore_images is False
    assert converter.ignore_emphasis is False
    assert converter.inline_links is True
    assert converter.single_line_break is True
    assert converter.wrap_links is False
    assert converter.unicode_snob is True


def test_html_to_text_sections_excludes_links_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: List[object] = []

    class DummyHTML2Text:
        def __init__(self) -> None:
            self.ignore_links = None
            self.ignore_images = None
            instances.append(self)

        def handle(self, _: str) -> str:
            include_link = "https://example.com" if not self.ignore_links else ""
            image_fragment = "ALT" if not self.ignore_images else ""
            fragments = ["Heading", include_link, image_fragment]
            return "\n\n".join(filter(None, fragments))

    monkeypatch.setattr(
        html_module,
        "html2text",
        SimpleNamespace(HTML2Text=DummyHTML2Text),
    )

    no_links = html_module.html_to_text_sections(
        "<p>Example</p>",
        settings=_settings(html2text_links=False, html2text_ignore_images=False),
    )
    with_links = html_module.html_to_text_sections(
        "<p>Example</p>",
        settings=_settings(html2text_links=True, html2text_ignore_images=False),
    )
    without_images = html_module.html_to_text_sections(
        "<p>Example</p>",
        settings=_settings(html2text_links=True, html2text_ignore_images=True),
    )

    assert "https://example.com" not in " ".join(no_links)
    assert "https://example.com" in " ".join(with_links)
    assert "ALT" in " ".join(with_links)
    assert "ALT" not in " ".join(without_images)
    assert instances, "expected html2text converter to be created"
