"""Katman skorlarını normalize edip tek bir ham anomali skorunda birleştirir (Adım 5).

Normalizasyon, eğitim verisindeki skor dağılımından öğrenilir ve saklanır. API'ye tek işlem geldiğinde
sıralama (rank) hesaplanamaz; kaydedilmiş yüzdelik ızgarasıyla dönüşüm yapılır.

Çıktı:
- {katman}_pct       : katman skorunun eğitim dağılımındaki yüzdeliği (insan için okunabilir)
- raw_anomaly_score  : normalize skorların ağırlıklı toplamı, 0-1 (henüz context ile düzeltilmemiş)
- raw_percentile     : ham skorun eğitim dağılımındaki yüzdeliği (alarm bütçesi ile doğrudan okunur)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_platform.config import ScoringConfig


def _percentile_grid(tail_resolution: int) -> np.ndarray:
    """0-%99 arası düzenli, %99 üstü log aralıklı yüzdelik noktaları: kuyrukta çözünürlük."""
    body = np.linspace(0, 0.99, 991)
    tail = 1 - np.logspace(-2, -5.5, tail_resolution)
    return np.unique(np.r_[body, tail, 1.0])


class PercentileMap:
    """Skor -> eğitim dağılımındaki yüzdelik, kaydedilmiş ızgara üzerinden doğrusal interpolasyon."""

    def __init__(self, tail_resolution: int):
        self.grid = _percentile_grid(tail_resolution)

    def fit(self, s: pd.Series) -> PercentileMap:
        self.values_ = np.quantile(s.to_numpy(dtype="float64"), self.grid)
        self.n_ = len(s)
        return self

    def __call__(self, s: pd.Series) -> pd.Series:
        return pd.Series(np.interp(s.to_numpy(dtype="float64"), self.values_, self.grid), index=s.index)


class Normalizer(ABC):
    """Katman skorunu 0-1 aralığına taşıyan strateji."""

    def __init__(self, tail_resolution: int):
        self.pct = PercentileMap(tail_resolution)

    def fit(self, s: pd.Series) -> Normalizer:
        self.pct.fit(s)
        return self

    @abstractmethod
    def __call__(self, s: pd.Series) -> pd.Series: ...


class PercentileNormalizer(Normalizer):
    def __call__(self, s: pd.Series) -> pd.Series:
        return self.pct(s)


class MinMaxNormalizer(Normalizer):
    def fit(self, s: pd.Series) -> MinMaxNormalizer:
        super().fit(s)
        self.lo_, self.hi_ = float(s.min()), float(s.max())
        return self

    def __call__(self, s: pd.Series) -> pd.Series:
        return ((s - self.lo_) / (self.hi_ - self.lo_)).clip(0, 1)


class TailLogNormalizer(Normalizer):
    """-log10(1 - yüzdelik) / log10(n): %90 -> 1/5,7, %99 -> 2/5,7, %99,99 -> 4/5,7 (n = 472 bin).

    Kuyruk olasılıklarını log ölçekte toplamak p-değerlerini birleştiren Fisher yöntemine benzer.
    """

    def __call__(self, s: pd.Series) -> pd.Series:
        n = self.pct.n_
        tail = (1 - self.pct(s)).clip(lower=1 / n)
        return -np.log10(tail) / np.log10(n)


NORMALIZERS: dict[str, type[Normalizer]] = {
    "percentile": PercentileNormalizer,
    "minmax": MinMaxNormalizer,
    "tail_log": TailLogNormalizer,
}


class ScoreAggregator:
    def __init__(self, cfg: ScoringConfig):
        total = sum(cfg.weights.values())
        self.weights = {k: v / total for k, v in cfg.weights.items()}
        self.cfg = cfg

    @property
    def layers(self) -> list[str]:
        return list(self.weights)

    def fit(self, scores: pd.DataFrame) -> ScoreAggregator:
        """scores: katman başına ham skor kolonları (eğitim / referans verisi)."""
        cls = NORMALIZERS[self.cfg.normalization]
        self.normalizers_ = {l: cls(self.cfg.tail_resolution).fit(scores[l]) for l in self.layers}
        self.final_pct_ = PercentileMap(self.cfg.tail_resolution).fit(self._raw(scores))
        return self

    def _raw(self, scores: pd.DataFrame) -> pd.Series:
        return sum(w * self.normalizers_[l](scores[l]) for l, w in self.weights.items()).rename("raw_anomaly_score")

    def transform(self, scores: pd.DataFrame) -> pd.DataFrame:
        out = {f"{l}_pct": self.normalizers_[l].pct(scores[l]) for l in self.layers}
        raw = self._raw(scores)
        out["raw_anomaly_score"] = raw
        out["raw_percentile"] = self.final_pct_(raw)
        return pd.DataFrame(out, index=scores.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ScoreAggregator:
        return joblib.load(path)
