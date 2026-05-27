"""Very small stub of the python-docx API used in tests."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Iterable, List
from xml.etree import ElementTree as ET


XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


@dataclass
class Paragraph:
    text: str


class Document:
    def __init__(self, stream: Iterable[bytes]) -> None:
        if hasattr(stream, "read"):
            data = stream.read()
        elif isinstance(stream, (bytes, bytearray, memoryview)):
            data = bytes(stream)
        else:
            chunks = []
            for part in stream:
                if isinstance(part, (bytes, bytearray, memoryview)):
                    chunks.append(bytes(part))
                else:
                    chunks.append(bytes([int(part)]))
            data = b"".join(chunks)
        if not data:
            raise ValueError("Empty DOCX payload")

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                try:
                    xml_bytes = archive.read("word/document.xml")
                except KeyError as exc:
                    raise ValueError("DOCX file missing word/document.xml") from exc
        except zipfile.BadZipFile as exc:  # pragma: no cover - defensive guard
            raise ValueError("Invalid DOCX file: not a ZIP archive") from exc

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise ValueError("DOCX file contains malformed XML") from exc

        namespace = _namespace(root.tag)
        body = root.find(_qualify("body", namespace)) if namespace else root.find("body")
        if body is None:
            raise ValueError("DOCX document is missing a <w:body> element")

        paragraphs: List[Paragraph] = []
        for element in body.findall(_qualify("p", namespace)) if namespace else body.findall("p"):
            text = _extract_paragraph_text(element, namespace)
            if text:
                paragraphs.append(Paragraph(text))
        self.paragraphs = paragraphs


def _namespace(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return None


def _qualify(local: str, namespace: str | None) -> str:
    return f"{{{namespace}}}{local}" if namespace else local


def _extract_paragraph_text(element: ET.Element, namespace: str | None) -> str:
    text_nodes = (
        element.findall(".//" + _qualify("t", namespace)) if namespace else element.findall(".//t")
    )
    parts: List[str] = []
    for node in text_nodes:
        content = node.text
        if not content:
            continue
        if node.get(XML_SPACE) == "preserve":
            parts.append(content)
        else:
            parts.append(content)
    return "".join(parts).strip()
