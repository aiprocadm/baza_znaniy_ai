from app.services.vectorstore import clear_fallback, index_chunks, search


def test_golden_npa_retrieval_current_revision_priority() -> None:
    clear_fallback()
    index_chunks([
        {"file":"fzl.txt","page":1,"text":"Федеральный закон 123 о данных","meta":{"reg_number":"123","is_active":True,"revision":"2025"}},
        {"file":"fzl_old.txt","page":1,"text":"Федеральный закон 123 старая редакция","meta":{"reg_number":"123","is_active":False,"revision":"2020"}},
    ])
    hits = search("закон 123", top_k=2, reg_number="123", revision_mode="current")
    assert hits
    assert hits[0]["file"] == "fzl.txt"


def test_golden_npa_retrieval_historical_mode() -> None:
    clear_fallback()
    index_chunks([
        {"file":"hist.txt","page":2,"text":"приказ 77 утратил силу","meta":{"reg_number":"77","is_active":False}},
    ])
    hits = search("приказ 77", top_k=5, revision_mode="historical")
    assert hits and hits[0]["file"] == "hist.txt"
