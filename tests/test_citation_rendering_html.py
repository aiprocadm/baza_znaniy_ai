"""Verify citation buttons are present in chat rendering of index.html."""

from __future__ import annotations

from pathlib import Path


HTML = Path(__file__).resolve().parents[1] / "data" / "www" / "index.html"


def test_citation_template_exists():
    text = HTML.read_text(encoding="utf-8")
    assert "kb-citation" in text, "no .kb-citation class in index.html"
    assert "data-doc-id" in text, "data-doc-id attribute missing"
    assert "data-page" in text, "data-page attribute missing"
    assert "data-has-original" in text, "data-has-original attribute missing"
    assert "data-snippet" in text, "data-snippet attribute missing"


def test_citation_click_wires_to_pdf_viewer():
    text = HTML.read_text(encoding="utf-8")
    assert "kbPdfViewer.openCitation" in text, "click handler must call openCitation"


def test_citation_uses_i18n_keys_for_label():
    """The label for citation buttons should be generated via t() with
    citation.with_page/citation.no_page/citation.text_doc keys."""
    text = HTML.read_text(encoding="utf-8")
    keys_found = sum(
        1 for k in ("citation.with_page", "citation.no_page", "citation.text_doc") if k in text
    )
    assert keys_found >= 1, "citation.* i18n keys not used in index.html"
