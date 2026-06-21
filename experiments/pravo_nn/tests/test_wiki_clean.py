from experiments.pravo_nn.wiki_collector.clean import clean_extract, is_substantial


def test_clean_strips_section_headings():
    raw = "Первый абзац.\n== История ==\nВторой абзац.\n=== Подраздел ===\nТретий."
    out = clean_extract(raw)
    assert "==" not in out
    assert "История" not in out  # the heading line is removed whole
    assert "Первый абзац." in out
    assert "Второй абзац." in out


def test_clean_collapses_blank_runs():
    out = clean_extract("А.\n\n\n\nБ.")
    assert "\n\n\n" not in out
    assert out == "А.\n\nБ."


def test_is_substantial_rejects_short_stub():
    assert not is_substantial("Слишком коротко.")
    assert is_substantial("длинный текст " * 50)
