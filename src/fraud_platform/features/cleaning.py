"""Adım 1 kalite bulgularına göre feature öncesi düzeltmeler (yeni feature üretmez)."""
from __future__ import annotations

import pandas as pd

from fraud_platform.config import FeaturesConfig
from fraud_platform.features.base import FeatureBuilder, FeatureSpec


class DataCleaner(FeatureBuilder):
    group = "cleaning"

    def __init__(self, cfg: FeaturesConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.cfg.email_columns:
            if c in df:
                # gmail / gmail.com aynı domain (quality_report.json -> domain_variants)
                df[c] = df[c].replace(self.cfg.email_aliases)
        for c in self.cfg.lowercase_columns:
            if c in df:
                # JSON'dan tek işlem null ile gelirse kolon float tiplenir; önce metne çevir (NaN korunur)
                df[c] = df[c].astype("str").str.lower()   # ALCATEL / Alcatel
        for c in self.cfg.non_negative_columns:
            if c in df:
                df[c] = df[c].clip(lower=0)  # gün farkı negatif olamaz
        return df

    def specs(self) -> list[FeatureSpec]:
        return []

    def required_columns(self) -> list[str]:
        return [*self.cfg.email_columns, *self.cfg.lowercase_columns, *self.cfg.non_negative_columns]
