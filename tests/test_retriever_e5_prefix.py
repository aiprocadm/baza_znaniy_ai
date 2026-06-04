from app.retriever.e5 import e5_prefix


def test_e5_prefix_applies_for_e5_models_when_enabled():
    assert (
        e5_prefix(
            "кто платит налог",
            role="query",
            model="intfloat/multilingual-e5-small",
            enabled=True,
        )
        == "query: кто платит налог"
    )
    assert (
        e5_prefix(
            "текст нормы",
            role="passage",
            model="intfloat/multilingual-e5-small",
            enabled=True,
        )
        == "passage: текст нормы"
    )


def test_e5_prefix_noop_when_disabled_or_non_e5():
    assert (
        e5_prefix("q", role="query", model="intfloat/multilingual-e5-small", enabled=False) == "q"
    )
    assert e5_prefix("q", role="query", model="BAAI/bge-m3", enabled=True) == "q"
