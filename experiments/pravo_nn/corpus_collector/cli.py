"""`collect` command: fetch → extract → assemble for every configured code.

A failed fetch is logged and that code is skipped (partial corpus is still
useful); the gap is visible because the code is simply absent from the manifest.
No silent success."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

from experiments.pravo_nn.corpus_collector import assemble, config, extract, fetch
from experiments.pravo_nn.corpus_collector.config import CodeSpec
from experiments.pravo_nn.corpus_collector.fetch import FetchError

LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def collect(
    *,
    codes: Sequence[CodeSpec],
    source_base: str,
    source_label: str,
    data_dir: Path,
    collected_at: str,
    encoding: str = "utf-8",
    opener=None,
) -> None:
    raw_dir = data_dir / "raw"
    corpus_dir = data_dir / "corpus"
    all_articles: list[extract.Article] = []
    per_code: dict[str, list[extract.Article]] = {}
    for spec in codes:
        if not spec.nd:
            LOGGER.warning("no nd id for %s yet — skipping", spec.name)
            continue
        fetch_kwargs: dict = {
            "source_base": source_base,
            "cache_dir": raw_dir,
            "encoding": encoding,
        }
        if opener is not None:
            fetch_kwargs["opener"] = opener
        try:
            raw = fetch.fetch_raw(spec, **fetch_kwargs)
        except FetchError as exc:
            LOGGER.error("skipping %s: %s", spec.name, exc)
            continue
        url = fetch.url_for(spec, source_base=source_base)
        arts = extract.extract_articles(raw, code=spec.name, source_url=url, date="")
        if not arts:
            LOGGER.warning("%s yielded 0 articles — check the extractor / source", spec.name)
        per_code[spec.name] = arts
        all_articles.extend(arts)

    assemble.write_corpus(all_articles, corpus_dir)
    manifest = assemble.build_manifest(per_code, collected_at=collected_at, source=source_label)
    assemble.write_manifest(manifest, data_dir / "manifest.json")
    small = [d["code"] for d in manifest["documents"] if d["suspiciously_small"]]
    if small:
        LOGGER.warning("suspiciously small (possible image-PDF): %s", ", ".join(small))
    LOGGER.info(
        "wrote %d articles from %d codes to %s",
        len(all_articles),
        len(per_code),
        corpus_dir,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="corpus_collector")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect", help="fetch + assemble the RF-codes corpus")
    args = parser.parse_args(argv)
    if args.command == "collect":
        if not config.SOURCE_BASE:
            parser.error("config.SOURCE_BASE is empty — run the Task 3 spike and set it")
        collect(
            codes=config.CODES,
            source_base=config.SOURCE_BASE,
            source_label=config.SOURCE_BASE,
            data_dir=_DEFAULT_DATA,
            collected_at=date.today().isoformat(),
            encoding="cp1251",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
