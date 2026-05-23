"""Verify the i18n loader JS and ru.json are well-formed and consistent."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WWW = ROOT / "data" / "www"
I18N = WWW / "i18n"


def test_ru_json_exists_and_valid():
    """ru.json must exist and parse as a flat dict of string→string."""
    path = I18N / "ru.json"
    assert path.exists(), f"missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    for k, v in data.items():
        assert isinstance(k, str), f"non-string key: {k!r}"
        assert isinstance(v, str), f"non-string value for {k!r}: {v!r}"


def test_loader_js_exists_and_uses_data_i18n_attribute():
    """_loader.js must reference data-i18n attribute selector."""
    path = I18N / "_loader.js"
    assert path.exists(), f"missing {path}"
    content = path.read_text(encoding="utf-8")
    assert "data-i18n" in content
    assert "querySelectorAll" in content


def test_ru_json_has_minimum_keys():
    """Sanity check: ru.json must have keys for header, common actions, admin sections."""
    data = json.loads((I18N / "ru.json").read_text(encoding="utf-8"))
    expected_keys = {
        "app.title",
        "header.subtitle",
        "tab.documents",
        "tab.search",
        "tab.qa",
        "action.upload",
        "action.search",
        "action.ask",
        "admin.title",
        "admin.header",
        "admin.section.upload",
        "citation.with_page",
        "citation.no_page",
        "citation.text_doc",
        "modal.viewer_title",
        "action.close",
        "viewer.page",
        "viewer.prev",
        "viewer.next",
        "viewer.error.not_available",
        "viewer.error.file_deleted",
        "viewer.error.load_failed",
        "viewer.fallback.text_only",
        "viewer.fallback.scan_no_text",
        "viewer.fallback.page_out_of_range",
    }
    missing = expected_keys - data.keys()
    assert not missing, f"missing keys: {missing}"


def test_loader_supports_interpolation():
    """_loader.js must expose a t() helper that substitutes {var} tokens."""
    content = (I18N / "_loader.js").read_text(encoding="utf-8")
    assert "t(" in content or "window.t" in content
    # The substitution pattern uses simple {key} braces — verify it's wired
    assert "{" in content and "replace" in content, (
        "_loader.js should support {var}-style interpolation in t()"
    )
