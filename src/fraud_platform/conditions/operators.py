"""Koşul operatörleri (Registry). Yeni operatör eklemek = sözlüğe bir satır.

Her operatör bir kolonu (Series) ve YAML'daki değeri alır, satır başına True/False döner.
Boş değer (NaN) hiçbir karşılaştırmayı sağlamaz; sadece is_null ile yakalanır.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

Operator = Callable[[pd.Series, Any], pd.Series]


def _present(s: pd.Series, mask: pd.Series) -> pd.Series:
    return (mask & s.notna()).fillna(False).astype(bool)


OPERATORS: dict[str, Operator] = {
    "eq": lambda s, v: _present(s, s.eq(v)),
    "neq": lambda s, v: _present(s, s.ne(v)),
    "gt": lambda s, v: _present(s, s.gt(v)),
    "gte": lambda s, v: _present(s, s.ge(v)),
    "lt": lambda s, v: _present(s, s.lt(v)),
    "lte": lambda s, v: _present(s, s.le(v)),
    "in": lambda s, v: _present(s, s.isin(v)),
    "not_in": lambda s, v: _present(s, ~s.isin(v)),
    "between": lambda s, v: _present(s, s.between(v[0], v[1])),   # iki uç dahil
    "is_null": lambda s, v: (s.isna() if v else s.notna()).astype(bool),
}

LIST_OPERATORS = {"in", "not_in"}
RANGE_OPERATORS = {"between"}
