"""Tests for the lightweight DOCX reader and ingestion pipeline."""

from __future__ import annotations

import io
import zipfile
from typing import List

import pytest
from xml.sax.saxutils import escape

from docx import Document

from app.ingest.chunking import parse_and_chunk


def _build_docx(paragraphs: List[str]) -> bytes:
    body = "".join(
        """
        <w:p>
            <w:r><w:t>{text}</w:t></w:r>
        </w:p>
        """.format(
            text=escape(paragraph)
        )
        for paragraph in paragraphs
    )
    document_xml = (
        """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                {body}
                <w:sectPr>
                    <w:pgSz w:w="12240" w:h="15840"/>
                    <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"
                        w:header="720" w:footer="720" w:gutter="0"/>
                </w:sectPr>
            </w:body>
        </w:document>
        """
    ).format(body=body)

    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "[Content_Types].xml",
                """<?xml version='1.0' encoding='UTF-8'?>
                <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
                    <Default Extension="rels"
                        ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
                    <Default Extension="xml" ContentType="application/xml"/>
                    <Override PartName="/word/document.xml"
                        ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
                </Types>
                """,
            )
            archive.writestr(
                "_rels/.rels",
                """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                    <Relationship Id="rId1"
                        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
                        Target="word/document.xml"/>
                </Relationships>
                """,
            )
            archive.writestr("word/document.xml", document_xml)
        return buffer.getvalue()


def test_parse_and_chunk_extracts_text_from_real_docx() -> None:
    docx_bytes = _build_docx(["Alpha beta", "Gamma delta"])

    chunks = parse_and_chunk("demo.docx", docx_bytes)

    assert chunks
    combined = " ".join(chunk["text"] for chunk in chunks)
    assert "Alpha" in combined
    assert "Gamma" in combined


def test_document_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError):
        Document(b"not-a-zip")
