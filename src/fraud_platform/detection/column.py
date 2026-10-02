"""Column katmanı: her değer tek başına tuhaf mı?

Sayısal kolon: robust z (medyan / MAD). Kategorik kolon: nadirlik = -log10(frekans).
Satır skoru: tüm kolon skorlarının ortalaması. Doğrulamada max ve top-k'dan belirgin şekilde iyi çıktı
(317 kolonda neredeyse her işlemin en az bir uç kolonu var, max doyuyor). Gerekçe: en uç 3 kolon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.config import ColumnDetectorConfig
from fraud_platform.detection.base import BaseDetector, Reason


class ColumnDetector(BaseDetector):
    name = "column"

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], cfg: ColumnDetectorConfig):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cfg = cfg

    def input_columns(self) -> list[str]:
        return [*self.numeric_cols, *self.categorical_cols]

    def fit(self, df: pd.DataFrame) -> ColumnDetector:
        x = df[self.numeric_cols].astype("float64")
        self.median_ = x.median()
        abs_dev = (x - self.median_).abs()
        mad, mean_ad = abs_dev.median(), abs_dev.mean()
        # MAD = 0 olan kolonlarda ortalama mutlak sapmaya düş (profiler.robust_z ile aynı kural)
        self.scale_ = (1.4826 * mad).where(mad > 0, 1.2533 * mean_ad).replace(0, np.nan)
        self.freq_ = {c: df[c].value_counts(normalize=True) for c in self.categorical_cols}
        self.min_freq_ = 1 / len(df)
        return self

    def column_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        z = ((df[self.numeric_cols].astype("float64") - self.median_) / self.scale_).abs()
        num = (z.clip(upper=self.cfg.z_cap) / self.cfg.z_cap).fillna(0.0)

        cat = {}
        for c in self.categorical_cols:
            # eğitimde görülmemiş değer en nadir kabul edilir; boş değer anomali sayılmaz
            f = df[c].map(self.freq_[c]).astype("float64").fillna(self.min_freq_)
            cat[c] = (-np.log10(f) / self.cfg.rarity_log_cap).clip(0, 1).where(df[c].notna(), 0.0)
        return pd.concat([num, pd.DataFrame(cat, index=df.index)], axis=1)

    def score(self, df: pd.DataFrame) -> pd.Series:
        parts = [
            self.column_scores(df.iloc[i : i + self.cfg.batch_size]).mean(axis=1)
            for i in range(0, len(df), self.cfg.batch_size)
        ]
        return pd.concat(parts).rename(self.name)

    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[Reason]]:
        scores = self.column_scores(df)
        out = []
        for idx, row in scores.iterrows():
            reasons = []
            for col, s in row.sort_values(ascending=False).head(top_k).items():
                if s <= 0:
                    break
                value = df.at[idx, col]
                if col in self.freq_:
                    share = self.freq_[col].get(value, 0.0)
                    text = (f"{col} = {value} eğitimde hiç görülmemiş bir değer" if share == 0
                            else f"{col} = {value} nadir bir değer (eğitimde payı %{share * 100:.3f})")
                else:
                    z = (value - self.median_[col]) / self.scale_[col]
                    text = f"{col} = {value:.4g}, tipik değerden {abs(z):.1f} robust-z uzakta (medyan {self.median_[col]:.4g})"
                reasons.append(Reason(self.name, col, value, round(float(s), 4), text))
            out.append(reasons)
        return out
