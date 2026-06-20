"""Knobs for the Wikipedia sample. target_bytes ~12 MB matches the legal corpus
(~12.1 MB on disk) so a 50/50 mix keeps all the law."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WikiConfig:
    target_bytes: int = 12_000_000
    batch_limit: int = 20  # articles per API call (exlimit cap for full extracts)
    api_url: str = "https://ru.wikipedia.org/w/api.php"
    user_agent: str = "pravo-nn-research/1.0 (mini-GPT corpus; aiproc.adm@gmail.com)"
