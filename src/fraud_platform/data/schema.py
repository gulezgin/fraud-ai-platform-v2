"""Kolonların anlamsal tipini otomatik çıkarır ve schema.json olarak saklar.

pandas dtype tek başına yetmez: card1 int saklanır ama bir koddur, C1 de int'tir ama bir sayaçtır.
Bu yüzden sıra şöyle: config'teki roller -> config override'ları -> veriye dayalı kurallar.
Her kolon için hangi kuralın uygulandığı `reason` alanında tutulur.
"""
from __future__ import annotations

import json
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fraud_platform.config import DataConfig, SchemaConfig


class SemanticType(StrEnum):
    IDENTIFIER = "identifier"
    TARGET = "target"
    DATETIME = "datetime"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    CONSTANT = "constant"


class ColumnInfo(BaseModel):
    name: str
    semantic_type: SemanticType
    reason: str
    dtype: str
    n_unique: int
    null_ratio: float
    cardinality: str | None = None  # sadece kategorik kolonlar için: low / medium / high


def _is_integer_valued(s: pd.Series) -> bool:
    v = s.dropna().to_numpy()
    return len(v) > 0 and bool(np.all(np.mod(v, 1) == 0))


def infer_semantic_type(
    s: pd.Series, name: str, data_cfg: DataConfig, schema_cfg: SchemaConfig, n_unique: int
) -> tuple[SemanticType, str]:
    if name == data_cfg.id_column:
        return SemanticType.IDENTIFIER, "config: id_column"
    if name == data_cfg.target_column:
        return SemanticType.TARGET, "config: target_column"
    if name == data_cfg.time_column or pd.api.types.is_datetime64_any_dtype(s):
        return SemanticType.DATETIME, "config: time_column"

    if n_unique <= 1:
        return SemanticType.CONSTANT, "tek değer"
    if n_unique == 2:
        return SemanticType.BINARY, "2 farklı değer"

    for pattern in schema_cfg.categorical_overrides:
        if fnmatch(name, pattern):
            return SemanticType.CATEGORICAL, f"override: {pattern}"

    if not pd.api.types.is_numeric_dtype(s):
        return SemanticType.CATEGORICAL, "metin tipi"
    if n_unique == s.notna().sum() and _is_integer_valued(s):
        return SemanticType.IDENTIFIER, "her değer tekil tam sayı"
    return SemanticType.NUMERIC, "sayısal"


class Schema(BaseModel):
    columns: dict[str, ColumnInfo]

    def of_type(self, *types: SemanticType) -> list[str]:
        return [c.name for c in self.columns.values() if c.semantic_type in types]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in self.columns.values()]).set_index("name")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Schema:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class SchemaInferer:
    def __init__(self, data_cfg: DataConfig, schema_cfg: SchemaConfig):
        self.data_cfg = data_cfg
        self.schema_cfg = schema_cfg

    def _cardinality_level(self, n_unique: int) -> str:
        if n_unique <= self.schema_cfg.low_cardinality_max:
            return "low"
        if n_unique > self.schema_cfg.high_cardinality_min:
            return "high"
        return "medium"

    def infer(self, df: pd.DataFrame) -> Schema:
        n_unique = df.nunique(dropna=True)
        null_ratio = df.isna().mean()

        columns = {}
        for name in df.columns:
            nu = int(n_unique[name])
            stype, reason = infer_semantic_type(df[name], name, self.data_cfg, self.schema_cfg, nu)
            columns[name] = ColumnInfo(
                name=name,
                semantic_type=stype,
                reason=reason,
                dtype=str(df[name].dtype),
                n_unique=nu,
                null_ratio=round(float(null_ratio[name]), 6),
                cardinality=self._cardinality_level(nu) if stype == SemanticType.CATEGORICAL else None,
            )
        return Schema(columns=columns)
