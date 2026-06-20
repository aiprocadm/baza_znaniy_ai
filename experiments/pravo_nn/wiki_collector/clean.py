"""Turn a raw Wikipedia plaintext extract into training text: remove section
heading lines ("== История =="), collapse blank runs, and judge whether an
article carries enough prose to be worth keeping (stubs add noise, not grammar)."""

from __future__ import annotations

import re

# A whole line that is just a "== ... ==" / "=== ... ===" heading.
_HEADING_RE = re.compile(r"^\s*={2,}.*?={2,}\s*$", re.MULTILINE)
_BLANKS_RE = re.compile(r"\n{3,}")
MIN_ARTICLE_CHARS = 200


def clean_extract(text: str) -> str:
    text = _HEADING_RE.sub("", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def is_substantial(text: str, *, min_chars: int = MIN_ARTICLE_CHARS) -> bool:
    return len(text) >= min_chars
