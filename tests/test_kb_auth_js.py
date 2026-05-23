"""Verify kb-auth.js exports the expected API and index.html uses it."""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WWW = ROOT / "data" / "www"
JS = WWW / "js"


def test_kb_auth_js_exists():
    assert (JS / "kb-auth.js").exists(), "kb-auth.js missing"


def test_kb_auth_js_exports_window_namespace():
    content = (JS / "kb-auth.js").read_text(encoding="utf-8")
    assert "window.kbAuth" in content, "kb-auth.js must define window.kbAuth"
    for fn in ("getApiKey", "withAuthHeaders", "fetch"):
        assert fn in content, f"kb-auth.js missing {fn}"


def test_kb_auth_js_uses_correct_storage_key():
    content = (JS / "kb-auth.js").read_text(encoding="utf-8")
    # Must match the key used by the existing inline UI
    assert '"kb_mvp_api_key"' in content


def test_index_html_loads_kb_auth_js():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    assert "/js/kb-auth.js" in html, "index.html must script-include kb-auth.js"
    # And must reference window.kbAuth somewhere
    assert "kbAuth" in html
