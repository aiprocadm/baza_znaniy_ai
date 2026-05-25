"""Tests that the index.html debug-pill row is hidden by default."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "data" / "www" / "index.html"


def test_pill_row_has_data_debug_attribute():
    """The pill-row container must be marked with data-debug-pill='1'."""
    html = HTML.read_text(encoding="utf-8")
    assert re.search(
        r'<div\s+class="pill-row"[^>]*data-debug-pill="1"',
        html,
    ), "pill-row must have data-debug-pill='1' attribute"


def test_inline_js_hides_pill_row_when_debug_not_in_query():
    """The inline JS must check URLSearchParams and hide [data-debug-pill] by default."""
    html = HTML.read_text(encoding="utf-8")
    assert "URLSearchParams" in html
    assert "data-debug-pill" in html
    # Must reference 'debug' as the query param name
    assert "'debug'" in html or '"debug"' in html


def test_debug_query_param_keeps_pills_visible():
    """When ?debug=1 is present, the JS must NOT hide the pills (i.e. assigns display='')."""
    html = HTML.read_text(encoding="utf-8")
    # Look for the conditional: if !has('debug') → display='none'
    pattern = re.compile(
        r"URLSearchParams[\s\S]{0,200}?debug[\s\S]{0,200}?(none|hidden)",
        re.IGNORECASE,
    )
    assert pattern.search(html), "expected conditional hide on missing ?debug"
