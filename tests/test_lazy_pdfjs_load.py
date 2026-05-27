"""Verify the initial HTML payload does NOT include PDF.js inline."""

from __future__ import annotations

from pathlib import Path


HTML = Path(__file__).resolve().parents[1] / "data" / "www" / "index.html"


def test_index_html_does_not_inline_pdfjs():
    text = HTML.read_text(encoding="utf-8")
    assert "/vendor/pdfjs/build/pdf.mjs" not in text, (
        "index.html must not eagerly load PDF.js — must be lazy via " "pdf-viewer.js dynamic import"
    )


def test_pdf_viewer_uses_dynamic_import():
    """pdf-viewer.js must reference the vendored PDF.js URL AND load it
    via dynamic ``import(...)`` — either inline or through a constant."""
    js = Path(__file__).resolve().parents[1] / "data" / "www" / "js" / "pdf-viewer.js"
    text = js.read_text(encoding="utf-8")

    assert "/vendor/pdfjs/build/pdf.mjs" in text, "vendored PDF.js URL must appear in pdf-viewer.js"

    inline_double = 'import("/vendor/pdfjs/build/pdf.mjs")' in text
    inline_single = "import('/vendor/pdfjs/build/pdf.mjs')" in text
    via_constant = "import(PDFJS_URL)" in text
    assert (
        inline_double or inline_single or via_constant
    ), "pdf-viewer.js must use dynamic import() for the PDF.js library"
