"""Mix the legal corpus and the Wikipedia sample ~50/50 by bytes into one
training file. The larger source is truncated (at a newline boundary, so no
article is cut mid-sentence) to the smaller source's byte budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    enc = text.encode("utf-8")
    if len(enc) <= max_bytes:
        return text
    cut = enc[:max_bytes].decode("utf-8", errors="ignore")
    nl = cut.rfind("\n")
    return cut[:nl] if nl > 0 else cut


def mix_corpora(law_text: str, wiki_text: str) -> tuple[str, dict]:
    budget = min(len(law_text.encode("utf-8")), len(wiki_text.encode("utf-8")))
    law_keep = _truncate_to_bytes(law_text, budget)
    wiki_keep = _truncate_to_bytes(wiki_text, budget)
    mixed = law_keep + "\n\n" + wiki_keep
    manifest = {
        "law_bytes": len(law_keep.encode("utf-8")),
        "wiki_bytes": len(wiki_keep.encode("utf-8")),
        "law_md5": hashlib.md5(law_keep.encode("utf-8")).hexdigest(),
        "wiki_md5": hashlib.md5(wiki_keep.encode("utf-8")).hexdigest(),
        "ratio": "50/50",
    }
    return mixed, manifest


def write_mixed(
    mixed: str,
    manifest: dict,
    *,
    out_path: Path,
    manifest_path: Path,
    collected_at: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mixed, encoding="utf-8")
    manifest_path.write_text(
        json.dumps({**manifest, "collected_at": collected_at}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
