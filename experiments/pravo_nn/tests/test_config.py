from experiments.pravo_nn.corpus_collector.config import CODES, CodeSpec


def test_codes_are_codespecs_with_nonempty_fields():
    assert len(CODES) >= 18  # ~20 RF codes
    for spec in CODES:
        assert isinstance(spec, CodeSpec)
        assert spec.name.strip()
        assert spec.slug.strip()


def test_slugs_are_unique_and_filename_safe():
    slugs = [s.slug for s in CODES]
    assert len(slugs) == len(set(slugs))  # no duplicates
    for slug in slugs:
        assert all(c.isalnum() or c == "-" for c in slug), slug
