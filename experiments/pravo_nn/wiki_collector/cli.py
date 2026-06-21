"""`collect_wiki`: loop fetch -> clean -> accumulate until the byte budget is met,
caching each raw batch so a re-run is free and offline. The fetch and sleep are
injectable so tests drive the loop without network."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Callable

from experiments.pravo_nn.wiki_collector import assemble, clean, fetch
from experiments.pravo_nn.wiki_collector.config import WikiConfig

LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def collect_wiki(
    *,
    cfg: WikiConfig,
    data_dir: Path,
    collected_at: str,
    fetch: Callable = fetch.fetch_batch,
    sleep: Callable[[float], None] = time.sleep,
    max_batches: int = 10000,
) -> Path:
    wiki_dir = data_dir / "wiki"
    raw_dir = wiki_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    batch_no = 0
    while total < cfg.target_bytes and batch_no < max_batches:
        batch_no += 1
        cache = raw_dir / f"batch-{batch_no:04d}.json"
        if cache.exists():
            raw_pairs = [tuple(p) for p in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            raw_pairs = fetch(api_url=cfg.api_url, limit=cfg.batch_limit, user_agent=cfg.user_agent)
            cache.write_text(json.dumps(raw_pairs, ensure_ascii=False), encoding="utf-8")
            sleep(0.5)  # be polite to the API
        for title, extract in raw_pairs:
            if title in seen:
                continue
            cleaned = clean.clean_extract(extract)
            if not clean.is_substantial(cleaned):
                continue
            seen.add(title)
            kept.append((title, cleaned))
            total += len(cleaned.encode("utf-8"))
            if total >= cfg.target_bytes:
                break

    assemble.write_wiki(kept, wiki_dir)
    manifest = assemble.build_manifest(kept, collected_at=collected_at, source=cfg.api_url)
    assemble.write_manifest(manifest, wiki_dir / "manifest.json")
    LOGGER.info("wrote %d wiki articles (%d bytes) to %s", len(kept), total, wiki_dir)
    return wiki_dir / "wiki.txt"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="wiki_collector")
    parser.add_argument("--target-bytes", type=int, default=WikiConfig.target_bytes)
    args = parser.parse_args(argv)
    collect_wiki(
        cfg=WikiConfig(target_bytes=args.target_bytes),
        data_dir=_DEFAULT_DATA,
        collected_at=date.today().isoformat(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
