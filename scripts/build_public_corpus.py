"""Build / maintain the synthetic public eval corpus.

The committed artifact is the *text* under ``data/eval/corpus_public/*.md`` —
never a generator run (spec 2026-06-06 §6: determinism). Two modes:

``ingest``
    Load every ``*.md`` fixture into the store selected by the environment
    (``KB_MVP_DB_PATH`` — point it at the public store). Idempotent: a document
    whose ``filename`` already exists is deleted and re-added, so re-running
    after editing a fixture cannot duplicate chunks.

``author``
    Draft ONE new document with the configured LLM (``select_provider`` — the
    keyless local GGUF by default) into the corpus directory for human review.
    Extension path only; the initial nine documents were teacher-authored and
    committed as reviewed text.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.services.kb_store import KnowledgeBaseStore, get_store

LOGGER = logging.getLogger(__name__)

CORPUS_DIR = Path("data/eval/corpus_public")

_AUTHOR_PROMPT = (
    "Ты — опытный корпоративный юрист и методолог. Напиши реалистичный "
    "внутренний документ российской компании в формате Markdown.\n"
    "Тип документа: {doc_type}\nТема: {topic}\n"
    "Требования: 3000-5000 слов; нумерованные разделы; конкретные выдуманные "
    "реквизиты, суммы, сроки и должности (внутренне непротиворечивые); без "
    "реальных персональных данных и реальных компаний. Начни с заголовка "
    "первого уровня '# '."
)


def derive_title(text: str, filename: str) -> str:
    """First markdown H1 wins; otherwise the filename stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return Path(filename).stem


def ingest_corpus(store: KnowledgeBaseStore, corpus_dir: Path) -> int:
    """Load every ``*.md`` in *corpus_dir* into *store*. Returns files ingested."""
    existing = {d.filename: d.id for d in store.list_documents() if d.filename}
    count = 0
    for md in sorted(corpus_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        if md.name in existing:
            store.delete_document(existing[md.name])
        store.add_document(derive_title(text, md.name), text=text, source="text", filename=md.name)
        count += 1
        LOGGER.info("ingested %s", md.name)
    return count


def author_document(name: str, doc_type: str, topic: str, *, provider=None) -> Path:
    """Draft one new corpus document for human review (extension path)."""
    if provider is None:
        from app.services.kb_llm import select_provider

        provider = select_provider()
    if provider is None:
        raise SystemExit("No LLM provider available (need the keyless GGUF or a key)")
    prompt = _AUTHOR_PROMPT.format(doc_type=doc_type, topic=topic)
    response = provider.generate(prompt, max_tokens=8192)
    out = CORPUS_DIR / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(response.text.strip() + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="load corpus_public/*.md into the env-selected store")
    ing.add_argument("--corpus-dir", default=str(CORPUS_DIR))

    auth = sub.add_parser("author", help="draft ONE new document via the configured LLM")
    auth.add_argument("name", help="output filename stem, e.g. policy_travel")
    auth.add_argument("--doc-type", required=True)
    auth.add_argument("--topic", required=True)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        n = ingest_corpus(get_store(), Path(args.corpus_dir))
        print(f"OK: ingested {n} document(s)")
    else:
        out = author_document(args.name, args.doc_type, args.topic)
        print(f"Draft written to {out} — review before committing")


if __name__ == "__main__":
    sys.exit(main())
