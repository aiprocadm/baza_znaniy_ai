"""Unit tests for the public-corpus build script (DI — no real models)."""

from __future__ import annotations

from pathlib import Path

from app.services.kb_store import KnowledgeBaseStore
from scripts.build_public_corpus import derive_title, ingest_corpus


class _FakeEmbedder:
    name = "fake"
    dimension = 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def test_derive_title_prefers_first_heading() -> None:
    assert derive_title("# Договор оказания услуг\n\nТекст", "x.md") == "Договор оказания услуг"
    assert derive_title("без заголовка", "contract_services.md") == "contract_services"


def _write(corpus: Path, name: str, body: str) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / name).write_text(body, encoding="utf-8")


def test_ingest_adds_every_md_with_filename(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.md", "# Док А\n\n" + "текст про оплату. " * 30)
    _write(corpus, "b.md", "# Док Б\n\n" + "текст про сроки. " * 30)
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())

    count = ingest_corpus(store, corpus)

    docs = store.list_documents()
    assert count == 2 and len(docs) == 2
    assert sorted(d.filename for d in docs) == ["a.md", "b.md"]
    assert sorted(d.title for d in docs) == ["Док А", "Док Б"]


def test_ingest_is_idempotent_replaces_same_filename(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.md", "# Версия 1\n\n" + "старый текст. " * 30)
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())
    ingest_corpus(store, corpus)

    _write(corpus, "a.md", "# Версия 2\n\n" + "новый текст. " * 30)
    count = ingest_corpus(store, corpus)

    docs = store.list_documents()
    assert count == 1 and len(docs) == 1
    assert docs[0].title == "Версия 2"
