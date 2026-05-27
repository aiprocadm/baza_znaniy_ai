"""Structural checks on the rewritten README.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_starts_with_h1_and_tagline():
    """First lines should declare what + for whom — a clear hook."""
    lines = README.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), f"first line should be H1, got: {lines[0]!r}"


def test_readme_contains_quickstart_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(
        r"^## .*Quickstart|## .*Быстрый старт", text, re.MULTILINE
    ), "expected an H2 like '## Quickstart' or '## Быстрый старт'"


def test_readme_contains_what_inside_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"^## .*Что внутри|## .*What.s inside", text, re.MULTILINE)


def test_readme_contains_not_for_you_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(
        r"^## .*Не для вас|## .*Not for you", text, re.MULTILINE
    ), "expected an H2 like '## Не для вас если' (anti-positioning)"


def test_readme_references_screenshots():
    text = README.read_text(encoding="utf-8")
    for shot in (
        "chat-with-citations.png",
        "pdf-viewer-modal.png",
        "upload-flow.png",
    ):
        assert shot in text, f"README should reference {shot}"


def test_readme_links_to_legacy():
    text = README.read_text(encoding="utf-8")
    assert (
        "docs/legacy_README.md" in text
    ), "README should link to docs/legacy_README.md for developer details"


def test_readme_mentions_apache_license():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"Apache.?2\.0", text), "license section should mention Apache-2.0"


def test_readme_has_ci_badge():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"!\[CI\]\(.*workflows.*\)", text), "expected a CI badge image link"


def test_readme_quickstart_is_short():
    """The quickstart block (≤30 lines) keeps the 5-min promise honest."""
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"## .*(Quickstart|Быстрый старт)\n(.*?)(?=^## )",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "Quickstart section missing"
    body = match.group(2)
    lines = [line for line in body.splitlines() if line.strip()]
    assert len(lines) <= 40, f"quickstart too long ({len(lines)} non-blank lines)"
