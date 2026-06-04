import pytest

from scripts.eval_sweep import _parse_values, format_sweep_table


def test_parse_values_parses_csv():
    assert _parse_values("5,8,10,12") == [5, 8, 10, 12]
    assert _parse_values(" 5 , 10 ") == [5, 10]


def test_parse_values_rejects_empty():
    with pytest.raises(SystemExit):
        _parse_values("  ")


def test_format_sweep_table_one_row_per_value_with_missing_placeholder():
    rows = [
        {"top_k": 5, "recall@5": 0.4, "recall@10": 0.5, "completeness": 3.2, "faithfulness": 4.1},
        {"top_k": 10, "recall@5": 0.6, "recall@10": 0.7},  # generation metrics absent
    ]
    table = format_sweep_table(rows)
    lines = table.splitlines()
    assert "top_k" in lines[0] and "faithfulness" in lines[0]
    assert len(lines) == 4  # header + separator + 2 data rows
    assert "0.400" in table  # formatted float
    assert "—" in table  # placeholder for missing completeness/faithfulness
