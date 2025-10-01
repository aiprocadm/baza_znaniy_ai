from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

import app.ingest.ocr as ocr
from app.ingest import chunking


class _FakeImage:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def close(self) -> None:  # pragma: no cover - no-op for compatibility
        return None


class _FakeBitmap:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_pil(self) -> _FakeImage:
        return _FakeImage(self.payload)

    def close(self) -> None:  # pragma: no cover - compatibility shim
        return None


class _FakePage:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def render(self, scale: float) -> _FakeBitmap:
        if self.payload.get("render_error"):
            raise RuntimeError("render failed")
        return _FakeBitmap(self.payload)

    def close(self) -> None:  # pragma: no cover - compatibility shim
        return None


class _FakePdfDocument:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self._pages = pages
        self.closed = False

    def __len__(self) -> int:
        return len(self._pages)

    def get_page(self, index: int) -> _FakePage:
        payload = self._pages[index]
        if payload.get("load_error"):
            raise RuntimeError("load failed")
        return _FakePage(payload)

    def close(self) -> None:
        self.closed = True


class _FakeTesseract:
    def __init__(self) -> None:
        self.calls: list[tuple[_FakeImage, str]] = []
        self.pytesseract = SimpleNamespace(tesseract_cmd=None)

    def image_to_string(self, image: _FakeImage, *, config: str = "", **_: object) -> str:
        self.calls.append((image, config))
        if image.payload.get("ocr_error"):
            raise RuntimeError("boom")
        return str(image.payload.get("text", ""))


def test_iter_pdf_pages_with_ocr_handles_page_errors_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"text": "First\npage"},
        {"load_error": True},
        {"text": "Skipped"},
    ]

    fake_pdfium = SimpleNamespace(PdfDocument=lambda payload: _FakePdfDocument(pages))
    fake_tesseract = _FakeTesseract()

    monkeypatch.setattr(ocr, "pdfium", fake_pdfium)
    monkeypatch.setattr(ocr, "pytesseract", fake_tesseract)
    monkeypatch.setattr(ocr, "Image", SimpleNamespace())

    config = ocr.OCRConfig(tesseract_cmd="/usr/bin/tesseract", dpi=150, page_limit=2, timeout_seconds=None)

    results = list(ocr.iter_pdf_pages_with_ocr(b"%PDF-1.4", config=config))

    assert results == [(1, "First page"), (2, "")]
    assert fake_tesseract.pytesseract.tesseract_cmd == "/usr/bin/tesseract"
    assert fake_tesseract.calls and fake_tesseract.calls[0][1].strip() == "--dpi 150"


def test_parse_and_chunk_streams_pdf_with_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[bytes] = []

    def _fake_iter_pdf_pages(data: bytes, *, config: ocr.OCRConfig):
        captured_calls.append(data)
        yield 1, "Page one content"
        yield 2, ""

    monkeypatch.setenv("RAG_CHUNK", "64")
    monkeypatch.setenv("RAG_OVERLAP", "0")

    monkeypatch.setattr(chunking, "iter_pdf_pages_with_ocr", _fake_iter_pdf_pages)
    monkeypatch.setattr(chunking, "_iter_pdf_text", lambda _: iter(()))
    monkeypatch.setattr(chunking, "_ocr_config", lambda: ocr.OCRConfig(dpi=200, timeout_seconds=None))

    pdf_payload = io.BytesIO(b"fake-pdf")
    chunks = chunking.parse_and_chunk("demo.pdf", pdf_payload)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"] == "Page one content"
    assert chunks[0]["file"] == "demo.pdf"
    assert captured_calls and captured_calls[0].startswith(b"fake-pdf")
