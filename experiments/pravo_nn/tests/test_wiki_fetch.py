import json
import urllib.error

import pytest

from experiments.pravo_nn.wiki_collector.fetch import (
    WikiFetchError,
    batch_url,
    fetch_batch,
    parse_batch,
)

_PAYLOAD = json.dumps(
    {
        "batchcomplete": "",
        "query": {
            "pages": {
                "12": {"pageid": 12, "ns": 0, "title": "Пушкин", "extract": "Поэт.\n== Жизнь ==\nРодился."},
                "34": {"pageid": 34, "ns": 0, "title": "Стуб", "extract": ""},
            }
        },
    }
)


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_batch_url_has_random_plaintext_params():
    url = batch_url(api_url="https://ru.wikipedia.org/w/api.php", limit=20)
    assert "generator=random" in url
    assert "explaintext=1" in url
    assert "grnlimit=20" in url
    assert "prop=extracts" in url


def test_parse_batch_keeps_titled_nonempty_extracts():
    pairs = parse_batch(_PAYLOAD)
    assert ("Пушкин", "Поэт.\n== Жизнь ==\nРодился.") in pairs
    assert all(title != "Стуб" for title, _ in pairs)  # empty extract dropped


def test_fetch_batch_offline_via_injected_opener():
    calls = []

    def opener(req):
        calls.append(req)
        return _Resp(_PAYLOAD.encode("utf-8"))

    pairs = fetch_batch(opener=opener, limit=20)
    assert ("Пушкин", "Поэт.\n== Жизнь ==\nРодился.") in pairs
    assert len(calls) == 1


def test_fetch_batch_retries_then_raises():
    attempts = []

    def opener(req):
        attempts.append(req)
        raise urllib.error.URLError("boom")

    with pytest.raises(WikiFetchError):
        fetch_batch(opener=opener, retries=3, sleep=lambda _s: None)
    assert len(attempts) == 3
