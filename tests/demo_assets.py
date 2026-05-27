from __future__ import annotations

"""Helpers for generating demo document fixtures on the fly."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict
from zipfile import ZIP_DEFLATED, ZipFile

__all__ = ["ensure_demo_assets"]


@lru_cache
def _build_pdf_bytes() -> bytes:
    """Generate a tiny valid PDF with predictable text."""

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objects = []

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    page = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    text_stream = b"BT /F1 18 Tf 72 720 Td (Demo contract ready for ingestion.) Tj ET\n"
    contents = (
        b"4 0 obj\n"
        + f"<< /Length {len(text_stream)} >>\n".encode()
        + b"stream\n"
        + text_stream
        + b"endstream\nendobj\n"
    )
    font = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    objects.extend([catalog, pages, page, contents, font])

    buffer = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(buffer))
        buffer.extend(obj)

    xref_offset = len(buffer)
    buffer.extend(f"xref\n0 {len(offsets)}\n".encode())
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode())

    buffer.extend(b"trailer\n")
    buffer.extend(f"<< /Size {len(offsets)} /Root 1 0 R >>\n".encode())
    buffer.extend(f"startxref\n{xref_offset}\n".encode())
    buffer.extend(b"%%EOF\n")
    return bytes(buffer)


_DOCX_PARTS: Dict[str, str] = {
    "[Content_Types].xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>\n'
        '  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>\n'
        "</Types>\n"
    ),
    "_rels/.rels": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>\n'
        '  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>\n'
        "</Relationships>\n"
    ),
    "word/_rels/document.xml.rels": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>\n'
    ),
    "word/document.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p>\n"
        "      <w:r>\n"
        "        <w:t>Demo overview document used for ingestion tests.</w:t>\n"
        "      </w:r>\n"
        "    </w:p>\n"
        "    <w:sectPr>\n"
        '      <w:pgSz w:w="11906" w:h="16838"/>\n'
        '      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>\n'
        "    </w:sectPr>\n"
        "  </w:body>\n"
        "</w:document>\n"
    ),
    "docProps/core.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        "  <dc:title>Demo Overview</dc:title>\n"
        "  <dc:subject>Knowledge base acceptance</dc:subject>\n"
        "  <dc:creator>kb_ai</dc:creator>\n"
        "  <cp:keywords>demo</cp:keywords>\n"
        "  <dc:description>Minimal document for automated ingestion checks.</dc:description>\n"
        "  <cp:lastModifiedBy>kb_ai</cp:lastModifiedBy>\n"
        '  <dcterms:created xsi:type="dcterms:W3CDTF">2024-01-01T00:00:00Z</dcterms:created>\n'
        '  <dcterms:modified xsi:type="dcterms:W3CDTF">2024-01-01T00:00:00Z</dcterms:modified>\n'
        "</cp:coreProperties>\n"
    ),
    "docProps/app.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">\n'
        "  <Application>kb_ai</Application>\n"
        "  <DocSecurity>0</DocSecurity>\n"
        "  <ScaleCrop>false</ScaleCrop>\n"
        "  <Company>kb_ai</Company>\n"
        "  <LinksUpToDate>false</LinksUpToDate>\n"
        "  <SharedDoc>false</SharedDoc>\n"
        "  <HyperlinksChanged>false</HyperlinksChanged>\n"
        "  <AppVersion>1.0</AppVersion>\n"
        "</Properties>\n"
    ),
}


@lru_cache
def _build_docx_bytes() -> bytes:
    """Assemble a minimal DOCX archive from the static parts above."""

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, xml in _DOCX_PARTS.items():
            archive.writestr(name, xml)
    return buffer.getvalue()


@lru_cache
def _build_txt_bytes() -> bytes:
    return b"Demo notes for the knowledge base acceptance scenario.\n"


ASSET_BUILDERS: Dict[str, Callable[[], bytes]] = {
    "demo_contract.pdf": _build_pdf_bytes,
    "demo_overview.docx": _build_docx_bytes,
    "demo_notes.txt": _build_txt_bytes,
}


def _write_bytes(path: Path, data: bytes) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            handle.write(data)
        temporary_path.replace(path)
    except OSError as exc:  # pragma: no cover - surfaced in acceptance tests
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to materialise demo asset {path.name}: {exc.strerror}") from exc


def ensure_demo_assets(target_dir: Path) -> None:
    """Materialise demo documents inside *target_dir* if they are missing."""

    target_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in ASSET_BUILDERS.items():
        path = target_dir / name
        data = builder()

        if path.exists():
            try:
                if path.read_bytes() == data:
                    continue
            except OSError:
                # If we cannot read the file, fall back to re-writing it.
                pass

        _write_bytes(path, data)
