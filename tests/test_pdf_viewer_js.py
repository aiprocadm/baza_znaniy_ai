"""Smoke-test pdf-viewer.js structure and integration with kb-auth."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "data" / "www" / "js"


def test_pdf_viewer_js_exists():
    assert (JS / "pdf-viewer.js").exists(), "pdf-viewer.js missing"


def test_pdf_viewer_exports_window_namespace():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "window.kbPdfViewer" in content
    assert "openCitation" in content


def test_pdf_viewer_uses_kb_auth():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "window.kbAuth.fetch" in content or "kbAuth.fetch" in content


def test_pdf_viewer_handles_404_410():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "404" in content and "410" in content


def test_pdf_viewer_lazy_imports_pdfjs():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "import(" in content
    assert "/vendor/pdfjs/build/pdf.mjs" in content


def test_pdf_viewer_uses_find_phrase_search():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "phraseSearch" in content
    assert "highlightAll" in content


def test_index_html_loads_pdf_viewer_js():
    html = (ROOT / "data" / "www" / "index.html").read_text(encoding="utf-8")
    assert "/js/pdf-viewer.js" in html


def test_pdf_viewer_detects_no_text_layer():
    """After renderTextLayer, the viewer should count spans and remember
    whether the page has any extractable text (scan-PDF detection)."""
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert 'querySelectorAll("span")' in content, (
        "renderPage must count text-layer span elements"
    )
    assert "hasTextLayer" in content, (
        "viewer must remember the no-text-layer state on the controller state"
    )


def test_pdf_viewer_shows_scan_no_text_banner():
    """When the text layer is empty, triggerFind must bail out early and
    surface the viewer.fallback.scan_no_text i18n message."""
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "viewer.fallback.scan_no_text" in content, (
        "triggerFind must reference the scan_no_text i18n key"
    )
