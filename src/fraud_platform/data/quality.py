"""Veri kalite raporu: tekrar, sabit/kopya kolon, eksiklik, geçersiz değer, yazım farkı, cardinality."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_platform.config import DataConfig, QualityConfig
from fraud_platform.data.schema import Schema, SemanticType

NULL_BINS = [-0.01, 0, 0.2, 0.5, 0.9, 1.0]
NULL_LABELS = ["eksiksiz", "az (<%20)", "orta (%20-50)", "çok (%50-90)", "neredeyse boş (>%90)"]

ENCODING_HINT = {
    "low": "one-hot",
    "medium": "frequency encoding",
    "high": "frequency encoding + nadir değerleri 'other' altında toplama",
}


def _digest(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df.isna().mean().sort_values(ascending=False)
    group = pd.cut(ratio, NULL_BINS, labels=NULL_LABELS)
    return pd.DataFrame({"null_ratio": ratio, "group": group})


def null_blocks(df: pd.DataFrame, min_size: int = 2) -> list[dict]:
    """Boşluk maskesi birebir aynı olan kolon grupları (V kolonlarında blok halinde eksiklik)."""
    groups = defaultdict(list)
    for c in df.columns[df.isna().any()]:
        groups[_digest(np.packbits(df[c].isna().to_numpy()))].append(c)

    blocks = [
        {"null_ratio": round(float(df[cols[0]].isna().mean()), 4), "n_columns": len(cols), "columns": cols}
        for cols in groups.values()
        if len(cols) >= min_size
    ]
    return sorted(blocks, key=lambda b: b["n_columns"], reverse=True)


def duplicate_columns(df: pd.DataFrame) -> list[list[str]]:
    """Değerleri birebir aynı olan kolonlar. Önce hash ile aday bulunur, sonra equals ile doğrulanır."""
    candidates = defaultdict(list)
    for c in df.columns:
        h = pd.util.hash_pandas_object(df[c], index=False).to_numpy()
        candidates[_digest(h)].append(c)

    result = []
    for cols in candidates.values():
        if len(cols) > 1 and all(df[cols[0]].equals(df[c]) for c in cols[1:]):
            result.append(cols)
    return result


def negative_value_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = {c: int((df[c] < 0).sum()) for c in columns}
    return {c: n for c, n in sorted(counts.items(), key=lambda x: -x[1]) if n > 0}


def case_variants(s: pd.Series) -> dict[str, list[str]]:
    """Büyük/küçük harf veya boşluk farkıyla yazılmış aynı değerler."""
    values = pd.Series(s.dropna().unique())
    key = values.str.strip().str.lower()
    groups = values.groupby(key).agg(list)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def domain_variants(s: pd.Series) -> dict[str, list[str]]:
    """Kök domain'i aynı olan farklı yazımlar: gmail / gmail.com, yahoo.com / yahoo.co.uk."""
    values = pd.Series(s.dropna().unique())
    root = values.str.lower().str.split(".").str[0]
    groups = values.groupby(root).agg(list)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def numeric_stored_as_text(df: pd.DataFrame, columns: list[str], min_ratio: float = 0.95) -> list[str]:
    result = []
    for c in columns:
        s = df[c].dropna()
        if len(s) and pd.to_numeric(s, errors="coerce").notna().mean() >= min_ratio:
            result.append(c)
    return result


class QualityReport:
    def __init__(self, data_cfg: DataConfig, quality_cfg: QualityConfig, schema: Schema):
        self.data_cfg = data_cfg
        self.cfg = quality_cfg
        self.schema = schema

    def high_cardinality(self) -> list[dict]:
        rows = [
            {
                "column": c.name,
                "n_unique": c.n_unique,
                "null_ratio": c.null_ratio,
                "suggested_encoding": ENCODING_HINT["high"],
            }
            for c in self.schema.columns.values()
            if c.cardinality == "high"
        ]
        return sorted(rows, key=lambda r: -r["n_unique"])

    def build(self, df: pd.DataFrame) -> dict:
        id_col, amt_col = self.data_cfg.id_column, self.data_cfg.amount_column
        numeric_cols = self.schema.of_type(SemanticType.NUMERIC)
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]

        nulls = null_summary(df)
        near_empty = nulls.index[nulls["null_ratio"] > self.cfg.near_empty_null_ratio].tolist()

        return {
            "shape": {"rows": len(df), "columns": df.shape[1]},
            "duplicates": {
                "duplicate_ids": int(df[id_col].duplicated().sum()),
                "duplicate_rows_excluding_id": int(df.drop(columns=id_col).duplicated().sum()),
                "duplicate_columns": duplicate_columns(df),
            },
            "constant_columns": self.schema.of_type(SemanticType.CONSTANT),
            "missing": {
                "groups": nulls["group"].value_counts().reindex(NULL_LABELS).astype(int).to_dict(),
                "near_empty_columns": near_empty,
                "null_blocks": null_blocks(df),
            },
            "invalid_values": {
                "non_positive_amount": int((df[amt_col] <= 0).sum()),
                "negative_value_columns": negative_value_columns(df, numeric_cols),
            },
            "text_issues": {
                "case_variants": {c: v for c in text_cols if (v := case_variants(df[c]))},
                "domain_variants": {c: domain_variants(df[c]) for c in self.cfg.domain_columns if c in df},
                "numeric_stored_as_text": numeric_stored_as_text(df, text_cols),
            },
            "high_cardinality": self.high_cardinality(),
        }

    @staticmethod
    def save(report: dict, path: Path) -> None:
        save_json(report, path)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
