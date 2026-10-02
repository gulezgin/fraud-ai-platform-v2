"""İç içe koşulların doğrulanması ve değerlendirilmesi (context motoru ve kural motoru ortak kullanır).

YAML biçimi:
    {field: TransactionAmt, op: gt, value: 1000}
    {all: [koşul, koşul, ...]}     VE
    {any: [koşul, koşul, ...]}     VEYA
    {not: koşul}                   DEĞİL

Koşullar Pydantic ile doğrulanır: bilinmeyen operatör, eksik değer, hatalı yapı YAML yüklenirken hata verir.
Değerlendirme vektöreldir: aynı kod 118 bin işlemi tek seferde ya da API'de tek işlemi değerlendirir.
"""
from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, model_validator

from fraud_platform.conditions.operators import LIST_OPERATORS, OPERATORS, RANGE_OPERATORS


class Leaf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    op: str
    value: Any = None

    @model_validator(mode="after")
    def _check(self) -> Leaf:
        if self.op not in OPERATORS:
            raise ValueError(f"bilinmeyen operatör '{self.op}', geçerli: {sorted(OPERATORS)}")
        if self.op in LIST_OPERATORS and not isinstance(self.value, list):
            raise ValueError(f"'{self.op}' için value bir liste olmalı")
        if self.op in RANGE_OPERATORS and not (isinstance(self.value, list) and len(self.value) == 2):
            raise ValueError("'between' için value [alt, üst] olmalı")
        if self.op == "is_null" and self.value is None:
            self.value = True
        if self.op != "is_null" and self.value is None:
            raise ValueError(f"'{self.op}' için value gerekli")
        return self


class AllOf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    all: list[Condition] = Field(min_length=1)


class AnyOf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    any: list[Condition] = Field(min_length=1)


class NotOf(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    not_: Condition = Field(alias="not")


def _kind(v: Any) -> str | None:
    """Koşul türünü anahtarlardan seçer; böylece hata mesajı ilgili türe özel olur (13 union hatası yerine 1)."""
    if isinstance(v, dict):
        return next((k for k in ("all", "any", "not") if k in v), "leaf")
    return {Leaf: "leaf", AllOf: "all", AnyOf: "any", NotOf: "not"}.get(type(v))


Condition = Annotated[
    Annotated[Leaf, Tag("leaf")] | Annotated[AllOf, Tag("all")] | Annotated[AnyOf, Tag("any")] | Annotated[NotOf, Tag("not")],
    Discriminator(_kind),
]
for _m in (AllOf, AnyOf, NotOf):
    _m.model_rebuild()

condition_adapter = TypeAdapter(Condition)


def parse_condition(raw: dict) -> Condition:
    return condition_adapter.validate_python(raw)


def evaluate(cond: Condition, df: pd.DataFrame) -> pd.Series:
    """Her satır için koşul sağlanıyor mu. Veride olmayan alan boş (NaN) kabul edilir."""
    if isinstance(cond, AllOf):
        return np.logical_and.reduce([evaluate(c, df) for c in cond.all])
    if isinstance(cond, AnyOf):
        return np.logical_or.reduce([evaluate(c, df) for c in cond.any])
    if isinstance(cond, NotOf):
        return ~evaluate(cond.not_, df)
    col = df[cond.field] if cond.field in df else pd.Series(np.nan, index=df.index)
    return OPERATORS[cond.op](col, cond.value)


def evaluate_row(cond: Condition, row: dict[str, Any]) -> bool:
    return bool(evaluate(cond, pd.DataFrame([row])).iloc[0])


def fields(cond: Condition) -> list[str]:
    """Koşulun okuduğu alanlar (açıklamada 'eşleşen değerler' için)."""
    if isinstance(cond, AllOf):
        return [f for c in cond.all for f in fields(c)]
    if isinstance(cond, AnyOf):
        return [f for c in cond.any for f in fields(c)]
    if isinstance(cond, NotOf):
        return fields(cond.not_)
    return [cond.field]


def describe(cond: Condition) -> str:
    """Okunabilir koşul metni: (TransactionAmt > 1000 VE local_hour 0-5 arası)."""
    symbols = {"eq": "=", "neq": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "in": "∈", "not_in": "∉"}
    if isinstance(cond, AllOf):
        return "(" + " VE ".join(describe(c) for c in cond.all) + ")"
    if isinstance(cond, AnyOf):
        return "(" + " VEYA ".join(describe(c) for c in cond.any) + ")"
    if isinstance(cond, NotOf):
        return f"DEĞİL {describe(cond.not_)}"
    if cond.op == "between":
        return f"{cond.field} {cond.value[0]}-{cond.value[1]} arası"
    if cond.op == "is_null":
        return f"{cond.field} {'boş' if cond.value else 'dolu'}"
    return f"{cond.field} {symbols[cond.op]} {cond.value}"
