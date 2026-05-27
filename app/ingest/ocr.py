"""Optical character recognition helpers for PDF ingestion."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

try:  # pragma: no cover - dependencies may be optional in slim environments
    import pytesseract
except Exception:  # pragma: no cover - handled gracefully by runtime checks
    pytesseract = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency is optional for environments without OCR
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - handled gracefully when missing
    pdfium = None  # type: ignore[assignment]

try:  # pragma: no cover - Pillow might be excluded from certain deployments
    from PIL import Image
except Exception:  # pragma: no cover - degrade when Pillow unavailable
    Image = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Raised when OCR processing cannot be performed."""


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for streaming OCR processing of PDF pages."""

    tesseract_cmd: str | None = None
    dpi: int = 300
    page_limit: int | None = None
    timeout_seconds: float | None = None
    language: str | None = None


def _ensure_dependencies() -> None:
    if pytesseract is None:
        raise OCRError("pytesseract is not installed")
    if Image is None:
        raise OCRError("Pillow is required for OCR processing")
    if pdfium is None:
        raise OCRError("pypdfium2 is required for OCR processing")


def _prepare_source(source: str | os.PathLike[str] | bytes | bytearray | BinaryIO) -> object:
    if isinstance(source, (str, os.PathLike)):
        return Path(source)
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        stream = source  # type: ignore[assignment]
        try:
            stream.seek(0)
        except Exception:
            pass
        payload = stream.read()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, memoryview):  # pragma: no cover - rarely triggered
            return payload.tobytes()
        return str(payload or "").encode("utf-8")
    raise TypeError("Unsupported PDF input type for OCR")


def _clean_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _run_tesseract(image: Image.Image, config: OCRConfig) -> str:
    assert pytesseract is not None  # for type-checkers

    kwargs: dict[str, object] = {}
    if config.language:
        kwargs["lang"] = config.language

    options: Sequence[str] = []
    if config.dpi:
        options = [f"--dpi {int(config.dpi)}"]

    try:
        raw = pytesseract.image_to_string(image, config=" ".join(options), **kwargs)
    except Exception as exc:  # pragma: no cover - passthrough for logging upstream
        raise OCRError(f"tesseract failed: {exc}") from exc
    return _clean_text(raw)


def iter_pdf_pages_with_ocr(
    source: str | os.PathLike[str] | bytes | bytearray | BinaryIO,
    *,
    config: OCRConfig,
) -> Iterator[tuple[int, str]]:
    """Yield OCR text for each page of ``source`` lazily."""

    _ensure_dependencies()

    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd  # type: ignore[attr-defined]

    payload = _prepare_source(source)

    try:
        document = pdfium.PdfDocument(payload)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - propagate for fallback
        raise OCRError(f"Failed to open PDF for OCR: {exc}") from exc

    start_time = time.monotonic()
    scale = max(int(config.dpi), 72) / 72 if config.dpi else 1.0

    try:
        page_total = len(document)
        limit = config.page_limit or page_total
        for index in range(min(page_total, limit)):
            if config.timeout_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > config.timeout_seconds:
                    LOGGER.warning("OCR timeout reached after %.2f seconds", elapsed)
                    break

            try:
                page = document.get_page(index)
            except Exception as exc:
                LOGGER.error("Failed to load PDF page %s for OCR: %s", index + 1, exc)
                yield index + 1, ""
                continue

            image: Image.Image | None = None
            bitmap = None
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                text = _run_tesseract(image, config)
            except OCRError as exc:
                LOGGER.error("OCR failed for page %s: %s", index + 1, exc)
                yield index + 1, ""
            except Exception:
                LOGGER.exception("Unexpected OCR failure on page %s", index + 1)
                yield index + 1, ""
            else:
                yield index + 1, text
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:  # pragma: no cover - Pillow housekeeping
                        pass
                if bitmap is not None and hasattr(bitmap, "close"):
                    try:
                        bitmap.close()
                    except Exception:  # pragma: no cover - resource cleanup
                        pass
                try:
                    page.close()
                except Exception:  # pragma: no cover - resource cleanup
                    pass
    finally:
        if hasattr(document, "close"):
            try:
                document.close()
            except Exception:  # pragma: no cover - cleanup best-effort
                pass


__all__ = ["OCRConfig", "OCRError", "iter_pdf_pages_with_ocr"]
