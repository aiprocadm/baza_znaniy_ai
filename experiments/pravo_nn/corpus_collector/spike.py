"""Throwaway source-probe used by the Task 3 SPIKE to pick a text source.

NOT imported by the collector. It documents *how* the decision was made: given
one or more candidate URLs on argv, it fetches each and prints content-type,
is_pdf, decoded char count, and how many times "Стать" (the article marker stem)
appears. Run it to reproduce the evidence behind the source choice.

    py -3.13 -m experiments.pravo_nn.corpus_collector.spike \
        "http://pravo.gov.ru/proxy/ips/?doc_itself=&nd=102033239&page=1&rdk=155"

Findings recorded in README.md "Text source":
  A. publication.pravo.gov.ru  -> official acts are image-PDF scans (OCR risk).
  B. pravo.gov.ru/proxy/ips/    -> clean selectable HTML, "Статья N" structure. CHOSEN.
  C. data.apicrafter.ru/...     -> metadata-only (200 rows, 2021), needs registration.

The portal serves windows-1251 and uses an outdated TLS chain, so we disable
cert verification and decode as cp1251 with a utf-8 fallback. This is a probe,
not production code -- the real collector lives in fetch.py.
"""

from __future__ import annotations

import sys
import urllib.request
from urllib.error import URLError

try:  # ssl is stdlib but guard anyway for odd builds
    import ssl

    _NO_VERIFY = ssl.create_default_context()
    _NO_VERIFY.check_hostname = False
    _NO_VERIFY.verify_mode = ssl.CERT_NONE
except Exception:  # pragma: no cover - probe only
    _NO_VERIFY = None


def _looks_like_pdf(head: bytes) -> bool:
    return head[:5] == b"%PDF-"


def probe(url: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=_NO_VERIFY, timeout=60) as resp:
            ctype = resp.headers.get("Content-Type", "?")
            body = resp.read()
    except URLError as exc:  # pragma: no cover - network
        print(f"{url}\n  ERROR: {exc}")
        return

    is_pdf = _looks_like_pdf(body) or "application/pdf" in ctype.lower()
    if "1251" in ctype:
        text = body.decode("cp1251", errors="replace")
    else:
        text = body.decode("utf-8", errors="replace")

    stat_hits = text.count("Стать")
    print(url)
    print(f"  content-type : {ctype}")
    print(f"  is_pdf       : {is_pdf}")
    print(f"  bytes        : {len(body)}")
    print(f"  decoded chars: {len(text)}")
    print(f"  'Стать' hits : {stat_hits}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for url in argv:
        probe(url)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
