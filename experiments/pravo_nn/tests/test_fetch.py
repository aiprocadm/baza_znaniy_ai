import urllib.error

import pytest

from experiments.pravo_nn.corpus_collector.config import CodeSpec
from experiments.pravo_nn.corpus_collector.fetch import FetchError, fetch_raw

SPEC = CodeSpec("ГК РФ", "gk-rf", "102033239")


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_writes_and_returns_text(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        return _Resp("Статья 1 текст".encode("utf-8"))

    out = fetch_raw(SPEC, source_base="http://src", cache_dir=tmp_path, opener=opener)
    assert "Статья 1" in out
    assert (tmp_path / "gk-rf.raw").exists()
    assert len(calls) == 1


def test_fetch_uses_cache_without_network(tmp_path):
    (tmp_path / "gk-rf.raw").write_text("cached", encoding="utf-8")

    def opener(url):  # must not be called
        raise AssertionError("network hit despite cache")

    assert fetch_raw(SPEC, source_base="http://src", cache_dir=tmp_path, opener=opener) == "cached"


def test_fetch_retries_then_raises(tmp_path):
    attempts = []

    def opener(url):
        attempts.append(url)
        raise urllib.error.URLError("boom")

    with pytest.raises(FetchError):
        fetch_raw(
            SPEC,
            source_base="http://src",
            cache_dir=tmp_path,
            opener=opener,
            retries=3,
            sleep=lambda _s: None,  # no real waiting in tests
        )
    assert len(attempts) == 3  # all retries exhausted


def test_url_for_builds_ips_document_url():
    from experiments.pravo_nn.corpus_collector.fetch import url_for

    url = url_for(SPEC, source_base="http://pravo.gov.ru/proxy/ips/")
    assert url == "http://pravo.gov.ru/proxy/ips/?doc_itself=&nd=102033239&page=1&rdk=0"


def test_fetch_decodes_with_given_encoding(tmp_path):
    body = "Статья 1 текст".encode("cp1251")

    def opener(url):
        return _Resp(body)

    out = fetch_raw(
        SPEC, source_base="http://x", cache_dir=tmp_path, opener=opener, encoding="cp1251"
    )
    assert "Статья 1 текст" in out  # correct cp1251 decode, not mojibake
