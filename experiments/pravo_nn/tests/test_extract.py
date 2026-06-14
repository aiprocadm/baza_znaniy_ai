from experiments.pravo_nn.corpus_collector.extract import (
    Article,
    extract_articles,
    normalize_whitespace,
    split_articles,
)

# A tiny synthetic raw doc in the spike's chosen format (HTML shown here).
RAW_HTML = (
    "<html><body>"
    "<p>ГРАЖДАНСКИЙ КОДЕКС</p>"
    "<p>Статья 1. Основные начала</p>"
    "<p>1. Гражданское законодательство основывается на равенстве.</p>"
    "<p>2</p>"  # standalone page number — must be dropped
    "<p>Статья 2. Регулируемые отношения</p>"
    "<p>Гражданское законодательство определяет правовое положение.</p>"
    "</body></html>"
)


def test_normalize_drops_page_numbers_and_collapses_whitespace():
    out = normalize_whitespace("Статья 1\n\n  много   пробелов \n42\nтекст")
    assert "  " not in out  # runs collapsed
    assert "\n42\n" not in out  # standalone number dropped
    assert "Статья 1" in out and "текст" in out


def test_split_articles_yields_one_article_per_marker():
    text = "преамбула\nСтатья 1\nтело один\nСтатья 2\nтело два"
    arts = split_articles(text, code="ГК РФ", source_url="http://x", date="")
    assert [a.article for a in arts] == ["Статья 1", "Статья 2"]
    assert arts[0].text == "тело один"
    assert all(isinstance(a, Article) and a.code == "ГК РФ" for a in arts)


def test_extract_articles_end_to_end_strips_tags_and_splits():
    arts = extract_articles(RAW_HTML, code="ГК РФ", source_url="http://x", date="1994-11-30")
    assert len(arts) == 2
    assert arts[0].article == "Статья 1. Основные начала"
    assert "равенстве" in arts[0].text
    assert "<p>" not in arts[0].text and "<" not in arts[1].text  # no tags survive
    assert all(a.date == "1994-11-30" for a in arts)


def test_normalize_collapses_non_breaking_spaces():
    # Real HTML/legal text is littered with U+00A0; it must collapse like a space.
    out = normalize_whitespace("Статья  1 текст")
    assert " " not in out
    assert out == "Статья 1 текст"
