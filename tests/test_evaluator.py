import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_platform.conditions.evaluator import (
    describe,
    evaluate,
    evaluate_row,
    fields,
    parse_condition,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "amt": [1500.0, 200.0, np.nan, 3000.0],
        "hour": [3, 3, 14, 23],
        "country": ["TR", None, "US", "US"],
        "flag": [1, 0, 1, 0],
    })


@pytest.mark.parametrize("op,value,expected", [
    ("eq", 1, [True, False, True, False]),
    ("neq", 1, [False, True, False, True]),
    ("gt", 1000, [True, False, False, True]),
    ("gte", 200, [True, True, False, True]),
    ("lt", 1000, [False, True, False, False]),
    ("lte", 200, [False, True, False, False]),
])
def test_comparison_operators(df, op, value, expected):
    col = "flag" if op in ("eq", "neq") else "amt"
    assert evaluate(parse_condition({"field": col, "op": op, "value": value}), df).tolist() == expected


def test_missing_values_never_match_except_is_null(df):
    assert not evaluate(parse_condition({"field": "amt", "op": "neq", "value": 5}), df)[2]
    assert not evaluate(parse_condition({"field": "country", "op": "not_in", "value": ["TR"]}), df)[1]
    assert evaluate(parse_condition({"field": "country", "op": "is_null"}), df).tolist() == [False, True, False, False]
    assert evaluate(parse_condition({"field": "amt", "op": "is_null", "value": False}), df).tolist() == [True, True, False, True]


def test_list_and_range_operators(df):
    assert evaluate(parse_condition({"field": "country", "op": "in", "value": ["US"]}), df).tolist() == [False, False, True, True]
    assert evaluate(parse_condition({"field": "hour", "op": "between", "value": [0, 5]}), df).tolist() == [True, True, False, False]


def test_nested_all_any_not(df):
    cond = parse_condition({
        "all": [
            {"field": "amt", "op": "gt", "value": 1000},
            {"any": [{"field": "hour", "op": "between", "value": [0, 5]}, {"not": {"field": "country", "op": "eq", "value": "US"}}]},
        ]
    })
    assert evaluate(cond, df).tolist() == [True, False, False, False]
    assert fields(cond) == ["amt", "hour", "country"]
    assert "VE" in describe(cond) and "VEYA" in describe(cond)


def test_unknown_field_is_treated_as_missing(df):
    cond = parse_condition({"field": "yok", "op": "gt", "value": 0})
    assert not evaluate(cond, df).any()


def test_evaluate_row():
    cond = parse_condition({"field": "amt", "op": "gt", "value": 100})
    assert evaluate_row(cond, {"amt": 150}) is True
    assert evaluate_row(cond, {"amt": None}) is False


@pytest.mark.parametrize("bad,message", [
    ({"field": "x", "op": "gtt", "value": 1}, "bilinmeyen operatör"),
    ({"field": "x", "op": "in", "value": 3}, "liste"),
    ({"field": "x", "op": "between", "value": [1]}, "alt, üst"),
    ({"field": "x", "op": "gt"}, "value gerekli"),
    ({"all": []}, "at least 1"),
    ({"all": [{"field": "x", "op": "gt", "value": 1}], "extra": 1}, "Extra inputs"),
])
def test_invalid_conditions_are_rejected(bad, message):
    with pytest.raises(ValidationError, match=message):
        parse_condition(bad)
