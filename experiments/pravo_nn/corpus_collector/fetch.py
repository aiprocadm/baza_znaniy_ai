"""Network layer: build the source URL for a code, fetch it (once), cache the raw
bytes on disk, retry transient failures with linear backoff. The opener and sleep
are injectable so tests run offline."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from experiments.pravo_nn.corpus_collector.config import CodeSpec


class FetchError(Exception):
    """Raised when a code cannot be fetched after all retries."""


def url_for(spec: CodeSpec, *, source_base: str) -> str:
    """Map a code to its URL at the chosen source. ADAPT to the spike outcome
    (this default assumes `<base>/<slug>`)."""
    return f"{source_base.rstrip('/')}/{spec.slug}"


def fetch_raw(
    spec: CodeSpec,
    *,
    source_base: str,
    cache_dir: Path,
    opener: Callable = urllib.request.urlopen,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    cache = cache_dir / f"{spec.slug}.raw"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = url_for(spec, source_base=source_base)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with opener(url) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(data, encoding="utf-8")
            return data
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep(backoff * (attempt + 1))
    raise FetchError(f"failed to fetch {spec.name} from {url}: {last_exc}")
