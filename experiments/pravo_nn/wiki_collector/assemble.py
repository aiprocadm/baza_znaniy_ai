"""Accumulate cleaned Wikipedia articles up to a byte budget, dedupe by title,
and write wiki.txt + a provenance manifest (deterministic — no clock reads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def accumulate(
    articles: Iterable[tuple[str, str]],
    *,
    target_bytes: int,
) -> tuple[list[tuple[str, str]], int]:
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    for title, text in articles:
        if title in seen:
            continue
        seen.add(title)
        kept.append((title, text))
        total += len(text.encode("utf-8"))
        if total >= target_bytes:
            break
    return kept, total


def write_wiki(kept: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "wiki.txt").open("w", encoding="utf-8") as tf:
        for _title, text in kept:
            tf.write(text + "\n\n")


def build_manifest(kept: list[tuple[str, str]], *, collected_at: str, source: str) -> dict:
    return {
        "collected_at": collected_at,
        "source": source,
        "articles": len(kept),
        "total_bytes": sum(len(x.encode("utf-8")) for _, x in kept),
        "titles": [t for t, _ in kept],
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
