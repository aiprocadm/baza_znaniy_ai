"""Golden-set schema + JSONL I/O. Back-compatible with synthetic_qa QAPair lines.

A golden line is a superset of the synthetic_qa QAPair layout: top-level
instruction/input/output is preserved (so scripts/validate_dataset.py stays
happy) and eval-specific fields live under ``meta``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GoldenItem:
    question: str
    relevant_chunk_ids: tuple[int, ...]
    reference_answer: str = ""
    expect_refusal: bool = False
    source: str = "auto"  # "auto" | "curated"

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.question,
            "input": "",
            "output": self.reference_answer,
            "meta": {
                "relevant_chunk_ids": [int(c) for c in self.relevant_chunk_ids],
                "source_chunk_id": int(self.relevant_chunk_ids[0]) if self.relevant_chunk_ids else 0,
                "expect_refusal": bool(self.expect_refusal),
                "source": self.source,
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "GoldenItem":
        data = json.loads(line)
        meta = data.get("meta") or {}
        if "relevant_chunk_ids" in meta:
            ids = [int(c) for c in meta["relevant_chunk_ids"]]
        else:
            sid = meta.get("source_chunk_id")
            ids = [int(sid)] if sid is not None else []
        return cls(
            question=str(data["instruction"]),
            relevant_chunk_ids=tuple(int(c) for c in ids),
            reference_answer=str(data.get("output", "")),
            expect_refusal=bool(meta.get("expect_refusal", False)),
            source=str(meta.get("source", "auto")),
        )


@dataclass(frozen=True, slots=True)
class CorpusSignature:
    doc_count: int
    max_chunk_id: int
    embedder_name: str
    dim: int

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_count": self.doc_count,
            "max_chunk_id": self.max_chunk_id,
            "embedder_name": self.embedder_name,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusSignature":
        return cls(
            doc_count=int(data["doc_count"]),
            max_chunk_id=int(data["max_chunk_id"]),
            embedder_name=str(data["embedder_name"]),
            dim=int(data["dim"]),
        )


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(GoldenItem.from_jsonl_line(line))
    return items


def save_golden(path: Path, items: Iterable[GoldenItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_jsonl_line())


def _sig_path(path: Path) -> Path:
    return Path(path).with_suffix(".sig.json")


def write_signature(path: Path, sig: CorpusSignature) -> None:
    _sig_path(path).write_text(
        json.dumps(sig.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_signature(path: Path) -> CorpusSignature | None:
    sp = _sig_path(path)
    if not sp.exists():
        return None
    return CorpusSignature.from_dict(json.loads(sp.read_text(encoding="utf-8")))
