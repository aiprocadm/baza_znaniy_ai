"""Write the assembled corpus (`corpus.jsonl` + `corpus.txt`) and the provenance
`manifest.json`. The output files are deterministic (no timestamps) so re-runs
are byte-identical — the manifest's `collected_at` is the only date-varying field
and is passed in, not read from the clock here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.pravo_nn.corpus_collector.extract import Article

# A code whose total body is below this many chars almost certainly failed to
# extract (e.g. an image-PDF that needs OCR). Flagged in the manifest, not silently shipped.
MIN_CODE_CHARS = 500


def write_corpus(articles: Sequence[Article], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "corpus.jsonl"
    txt = out_dir / "corpus.txt"
    with jsonl.open("w", encoding="utf-8") as jf, txt.open("w", encoding="utf-8") as tf:
        for a in articles:
            jf.write(
                json.dumps(
                    {
                        "code": a.code,
                        "article": a.article,
                        "text": a.text,
                        "source_url": a.source_url,
                        "date": a.date,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            tf.write(f"{a.article}\n{a.text}\n\n")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_manifest(
    per_code: Mapping[str, Sequence[Article]], *, collected_at: str, source: str
) -> dict:
    documents = []
    for code, arts in per_code.items():
        body = "".join(a.text for a in arts)
        documents.append(
            {
                "code": code,
                "source_url": arts[0].source_url if arts else "",
                "date": arts[0].date if arts else "",
                "articles": len(arts),
                "bytes": len(body.encode("utf-8")),
                "md5": _md5(body),
                "suspiciously_small": len(body) < MIN_CODE_CHARS,
            }
        )
    return {
        "collected_at": collected_at,
        "source": source,
        "documents": documents,
        "total_documents": len(documents),
        "total_bytes": sum(d["bytes"] for d in documents),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
