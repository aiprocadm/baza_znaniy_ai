import pytest
from app.eval.dataset import CorpusSignature
from app.eval.guards import ensure_real_embedder


def test_ensure_real_embedder_refuses_hash():
    sig = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="hash", dim=256)
    with pytest.raises(SystemExit, match="hashing"):
        ensure_real_embedder(sig, allow_hashing=False)


def test_ensure_real_embedder_allows_real_or_flagged():
    real = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="ollama", dim=1024)
    ensure_real_embedder(real, allow_hashing=False)  # no raise
    hashed = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="hash", dim=256)
    ensure_real_embedder(hashed, allow_hashing=True)  # no raise
