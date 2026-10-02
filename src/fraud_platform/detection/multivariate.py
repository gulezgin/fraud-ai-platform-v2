"""Multivariate katman: değerlerin birlikteliği tuhaf mı?

Isolation Forest, kantil dönüşümünden geçmiş feature'lar üzerinde çalışır. Kantil dönüşümü her kolonu 0-1 düzgün
dağılıma çevirir; böylece model tek kolondaki uç değeri değil (o column katmanının işi), kolonların birlikte nadir
görülen kombinasyonlarını yalıtır. Valid PR-AUC: ham + medyan 0,086 -> kantil 0,094 -> kantil + 260 temsilci kolon 0,155.
Boş değerler -1'e (0-1 aralığı dışına) atanır: "boş" ayrı bir bölge olarak öğrenilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import QuantileTransformer

from fraud_platform.config import MultivariateDetectorConfig
from fraud_platform.detection.base import BaseDetector, Reason

MISSING = -1.0


class MultivariateDetector(BaseDetector):
    name = "multivariate"

    def __init__(self, columns: list[str], cfg: MultivariateDetectorConfig):
        self.columns = columns
        self.cfg = cfg

    def input_columns(self) -> list[str]:
        return list(self.columns)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        q = self.quantiles_.transform(df[self.columns].astype("float64"))
        return pd.DataFrame(q, columns=self.columns, index=df.index)

    def fit(self, df: pd.DataFrame) -> MultivariateDetector:
        sample = df.sample(min(self.cfg.fit_sample_size, len(df)), random_state=self.cfg.random_state)
        n_q = min(self.cfg.n_quantiles, len(sample))   # küçük veride kantil sayısı satır sayısını aşamaz
        self.quantiles_ = QuantileTransformer(
            n_quantiles=n_q, subsample=max(self.cfg.fit_sample_size, n_q), random_state=self.cfg.random_state
        ).fit(sample[self.columns].astype("float64"))
        self.forest_ = IsolationForest(
            n_estimators=self.cfg.n_estimators, max_samples=self.cfg.max_samples,
            random_state=self.cfg.random_state, n_jobs=-1,
        ).fit(self._transform(sample).fillna(MISSING))
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        # score_samples: büyük = normal; işaret çevrilir
        return pd.Series(-self.forest_.score_samples(self._transform(df).fillna(MISSING)), index=df.index, name=self.name)

    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[Reason]]:
        """Yaklaşık açıklama: eğitimde en az görülen yönde uç değer alan feature'lar.

        Uçluk kuyruk olasılığıyla ölçülür: tail = min(P(X <= v), P(X >= v)). Kantil değeri kullanılmaz çünkü
        0/1 gibi kesikli kolonlarda en üst değer, payı %11 bile olsa %100'e eşlenir.
        Ağacın yalıtma yolunu birebir açıklamaz (SHAP gerekir); kombinasyonu tuhaflaştıran değerlere işaret eder.
        """
        ref = self.quantiles_.quantiles_   # (n_quantiles, n_features) eğitim dağılımı ızgarası
        values = df[self.columns].astype("float64").to_numpy()
        le = (ref[None, :, :] <= values[:, None, :]).mean(axis=1)
        ge = (ref[None, :, :] >= values[:, None, :]).mean(axis=1)
        tail = np.where(np.isnan(values), 0.5, np.minimum(le, ge))
        extremeness = np.clip(1 - 2 * tail, 0, 1)

        out = []
        for i, idx in enumerate(df.index):
            reasons = []
            for j in np.argsort(-extremeness[i])[:top_k]:
                if extremeness[i, j] <= 0:
                    break
                col, v = self.columns[j], values[i, j]
                side = "yüksek" if ge[i, j] <= le[i, j] else "düşük"
                text = f"{col} = {v:.4g}; eğitimde işlemlerin sadece %{tail[i, j] * 100:.1f}'i bu kadar {side}"
                reasons.append(Reason(self.name, col, df.at[idx, col], round(float(extremeness[i, j]), 4), text))
            out.append(reasons)
        return out
