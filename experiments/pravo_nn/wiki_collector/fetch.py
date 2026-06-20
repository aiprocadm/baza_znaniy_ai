"""Network layer for the Wikipedia sample: build the API URL and fetch ONE batch
of random-article plaintext extracts. The opener and sleep are injectable so
tests run fully offline (mirrors corpus_collector/fetch.py)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

API_URL = "https://ru.wikipedia.org/w/api.php"
USER_AGENT = "pravo-nn-research/1.0 (mini-GPT corpus; aiproc.adm@gmail.com)"


class WikiFetchError(Exception):
    """Raised when a batch cannot be fetched after all retries."""


def batch_url(*, api_url: str, limit: int) -> str:
    """One request that yields `limit` random article plaintext extracts."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "wiki",  # keep "== Heading ==" markers for clean.py to strip
        "exlimit": "max",
        "generator": "random",
        "grnnamespace": "0",  # article namespace only
        "grnlimit": str(limit),
    }
    return api_url + "?" + urllib.parse.urlencode(params)


def parse_batch(payload: str) -> list[tuple[str, str]]:
    """(title, plaintext) pairs from one API JSON response; drops empty extracts."""
    data = json.loads(payload)
    pages = data.get("query", {}).get("pages", {})
    out: list[tuple[str, str]] = []
    for page in pages.values():
        title = page.get("title", "")
        extract = page.get("extract", "")
        if title and extract:
            out.append((title, extract))
    return out


def fetch_batch(
    *,
    api_url: str = API_URL,
    limit: int = 20,
    opener: Callable = urllib.request.urlopen,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str = USER_AGENT,
) -> list[tuple[str, str]]:
    url = batch_url(api_url=api_url, limit=limit)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with opener(req) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
            return parse_batch(payload)
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep(backoff * (attempt + 1))
    raise WikiFetchError(f"failed to fetch wiki batch from {url}: {last_exc}")
