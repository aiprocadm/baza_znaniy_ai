import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ingest_module = importlib.import_module("app.rag.ingest")
_chunk = ingest_module._chunk
_get_tokenizer = ingest_module._get_tokenizer

for name in ["app.rag.ingest", "app.rag", "app"]:
    sys.modules.pop(name, None)


def _make_text_with_tokens(count: int, token_id: int = 100):
    encoder = _get_tokenizer()
    token_ids = [token_id] * count
    text = encoder.decode(token_ids)
    return text, encoder, token_ids


def test_chunk_respects_token_window_size():
    text, encoder, original_tokens = _make_text_with_tokens(1800)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

    assert len(chunks) == 3
    assert encoded_chunks[0] == original_tokens[:900]
    assert encoded_chunks[1] == original_tokens[760:1660]
    assert encoded_chunks[2] == original_tokens[1520:1800]
    assert all(len(tokens) <= 900 for tokens in encoded_chunks)


def test_chunk_overlap_consistency():
    text, encoder, _ = _make_text_with_tokens(1500, token_id=101)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        overlap = min(140, len(current), len(nxt))
        assert current[-overlap:] == nxt[:overlap]
