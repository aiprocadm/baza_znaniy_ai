from pathlib import Path

from app.eval.dataset import load_golden, read_signature

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "data" / "eval" / "golden_curated.jsonl"
MAX_CHUNK_ID = 48  # kb_mvp corpus snapshot: 1 doc / 48 chunks


def test_curated_golden_is_real_not_stub():
    items = load_golden(GOLDEN)
    assert len(items) >= 18
    assert all("ЗАМЕНИ" not in it.reference_answer for it in items)


def test_curated_golden_has_refusal_probes():
    items = load_golden(GOLDEN)
    refusals = [it for it in items if it.expect_refusal]
    assert len(refusals) >= 3
    assert all(it.relevant_chunks == () for it in refusals)


def test_curated_answerable_items_reference_real_chunks():
    items = load_golden(GOLDEN)
    answerable = [it for it in items if not it.expect_refusal]
    assert len(answerable) >= 12
    for it in answerable:
        assert it.relevant_chunks, it.question
        # Legacy curated file uses int-derived string keys; verify they parse as
        # valid ints within the known corpus range.
        assert all(
            1 <= int(cid) <= MAX_CHUNK_ID for cid in it.relevant_chunks if cid.isdigit()
        ), it.question
        assert it.reference_answer.strip(), it.question


def test_curated_golden_has_signature_sidecar():
    sig = read_signature(GOLDEN)
    assert sig is not None
    assert sig.doc_count == 1
    assert sig.max_chunk_id == MAX_CHUNK_ID
