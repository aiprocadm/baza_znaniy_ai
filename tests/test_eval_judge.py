from app.eval.judge import build_judge_prompt, parse_verdict, Verdict


def test_build_prompt_includes_sections():
    p = build_judge_prompt(question="Q?", answer="A [1]", context="[1] текст", reference="эталон")
    assert "Q?" in p and "A [1]" in p and "[1] текст" in p and "эталон" in p


def test_parse_verdict_plain_json():
    v = parse_verdict('{"faithfulness":5,"relevance":4,"completeness":3,"citation":2,"rationale":"ok"}')
    assert v == Verdict(5, 4, 3, 2, "ok")
    assert v.normalized()["faithfulness"] == 1.0 and v.normalized()["citation"] == 0.25


def test_parse_verdict_tolerates_fence_and_prose():
    raw = "Вот оценка:\n```json\n{\"faithfulness\":1,\"relevance\":1,\"completeness\":1,\"citation\":1}\n```"
    v = parse_verdict(raw)
    assert v is not None and v.faithfulness == 1


def test_parse_verdict_clamps_out_of_range():
    v = parse_verdict('{"faithfulness":9,"relevance":0,"completeness":3,"citation":3}')
    assert v.faithfulness == 5 and v.relevance == 1


def test_parse_verdict_malformed_returns_none():
    assert parse_verdict("no json here") is None
    assert parse_verdict("") is None
