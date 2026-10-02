"""Dedektörlerin ortak arayüzü (Strategy pattern).

Her dedektör aynı işleme farklı açıdan bakar ve iki şey üretir:
- score(): her işlem için ham anomali skoru (büyük = daha anormal). Ölçekler farklı; Adım 5'te normalize edilir.
- explain(): istenen işlemler için en çok katkı veren bileşenler ve okunabilir gerekçe.

Gerekçeler toplu skorlamada üretilmez: column dedektörü 300+ kolona bakıyor, 590 bin satırın tüm kolon
katkılarını saklamak ~1,5 GB tutar. API tek işlem için explain() çağırır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reason:
    detector: str
    feature: str
    value: object
    contribution: float     # 0-1, bileşenin anomali katkısı
    text: str

    def to_dict(self) -> dict:
        d = asdict(self)
        v = d["value"].item() if isinstance(d["value"], np.generic) else d["value"]
        if isinstance(v, float):
            v = None if np.isnan(v) else round(v, 4)
        d["value"] = v
        return d


def saturate(x: pd.Series, cap: float) -> pd.Series:
    """0..cap aralığını log ölçekte 0..1'e taşır; cap üstü 1. Sayaçların ilk artışları daha anlamlı."""
    return (np.log1p(x.clip(lower=0)) / np.log1p(cap)).clip(0, 1).fillna(0.0)


class BaseDetector(ABC):
    name: str = ""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseDetector: ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[Reason]]: ...

    def input_columns(self) -> list[str]:
        """Dedektörün okuduğu ham kolonlar; API'den eksik gelenler boş olarak tamamlanır."""
        return []

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> BaseDetector:
        return joblib.load(path)


class WeightedComponentDetector(BaseDetector):
    """Birkaç yorumlanabilir bileşenin (0-1) ağırlıklı ortalaması. Entity ve temporal katman bunu kullanır."""

    def __init__(self, weights: dict[str, float], caps: dict[str, float], amount_col: str):
        unknown = set(weights) - set(self.TEXTS)
        if unknown:
            raise ValueError(f"{self.name}: bilinmeyen bileşen {sorted(unknown)}")
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items() if v > 0}
        self.caps = caps
        self.amount_col = amount_col

    TEXTS: ClassVar[dict[str, str]] = {}
    VALUE_OF: ClassVar[dict[str, str]] = {}   # bileşen -> gerekçede "value" olarak dönecek ham değerin anahtarı

    @abstractmethod
    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bileşen skorları (0-1), sadece ağırlığı olan bileşenler."""

    @abstractmethod
    def component_values(self, row: pd.Series) -> dict[str, object]:
        """Gerekçe metnine girecek ham değerler."""

    def fit(self, df: pd.DataFrame) -> WeightedComponentDetector:
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        comp = self.components(df)
        w = pd.Series(self.weights)
        return (comp[w.index] * w).sum(axis=1).rename(self.name)

    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[Reason]]:
        comp = self.components(df)
        out = []
        for idx, row in comp.iterrows():
            values = self.component_values(df.loc[idx])
            top = row[row > 0].sort_values(ascending=False).head(top_k)
            out.append([
                Reason(self.name, c, values.get(self.VALUE_OF.get(c, c)), round(float(v), 4), self.TEXTS[c].format(**values))
                for c, v in top.items()
            ])
        return out
