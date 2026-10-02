"""Context motorunun (Adım 6) ve kuralların (Adım 7) kullanacağı bayraklar + boşluk bayrakları."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.config import DataConfig, FeaturesConfig, TimeConfig
from fraud_platform.features.base import FeatureBuilder, FeatureSpec, add_columns


def _in_window(hour: pd.Series, window: tuple[int, int]) -> pd.Series:
    start, end = window
    if start <= end:
        return hour.between(start, end, inclusive="left")
    return (hour >= start) | (hour < end)   # gece yarısını geçen pencere, ör. (22, 6)


class ContextFeatures(FeatureBuilder):
    group = "context"

    def __init__(self, cfg: FeaturesConfig, time_cfg: TimeConfig, data_cfg: DataConfig):
        self.cfg = cfg
        self.time_cfg = time_cfg
        self.amt = data_cfg.amount_column
        self.high_value_threshold_: float | None = None

    def fit(self, df: pd.DataFrame) -> ContextFeatures:
        avg = df["user_avg_amount"].dropna()
        self.high_value_threshold_ = float(avg.quantile(self.cfg.high_value_quantile))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        hour, amt = df["local_hour"], df[self.amt]
        addr2 = df["addr2"]
        new = {
            "is_night": _in_window(hour, self.time_cfg.night_hours).astype("int8"),
            "is_business_hours": _in_window(hour, self.time_cfg.business_hours).astype("int8"),
            "is_weekend": df["day_of_week"].isin(self.time_cfg.weekend_days).astype("int8"),
            "card_age_days": df["D1"],
            "is_new_card": (df["D1"] <= self.cfg.new_card_max_days).astype("int8"),
            "is_foreign_address": (addr2.notna() & (addr2 != self.cfg.domestic_country_code)).astype("int8"),
            # 3 ondalık haneli tutar: muhtemelen kur çevrimi. float32'de 47.95 = 47.950001 olduğu için
            # binde bire yuvarlayıp tam sayıyla kontrol ediliyor
            "amount_is_converted": (np.round(amt.astype("float64") * 1000) % 10 != 0).astype("int8"),
            "amount_log": np.log1p(amt),
            "is_mobile": (df["DeviceType"] == "mobile").astype("int8"),
            "is_high_value_customer": (df["user_avg_amount"] >= self.high_value_threshold_).astype("int8"),
        }
        return add_columns(df, new)

    def required_columns(self) -> list[str]:
        return [self.amt, "addr2", "D1", "DeviceType"]

    def specs(self) -> list[FeatureSpec]:
        c, cr = ("context",), ("context", "rules")
        return [
            FeatureSpec("is_night", "Yerel saat gece penceresinde (night_hours)", ("context", "temporal", "rules")),
            FeatureSpec("is_business_hours", "Yerel saat iş saatinde (business_hours)", c),
            FeatureSpec("is_weekend", "Cumartesi/Pazar (referans tarih varsayımı)", c),
            FeatureSpec("card_age_days", "D1: kartın ilk kullanımından beri gün (C/H/R'de çoğunlukla 0)", ("context", "multivariate", "rules")),
            FeatureSpec("is_new_card", "card_age_days <= new_card_max_days", cr),
            FeatureSpec("is_foreign_address", "Fatura ülkesi (addr2) yurt içi kodundan farklı", cr),
            FeatureSpec("amount_is_converted", "Tutar 3 ondalık haneli (kur çevrimi izi)", ("context", "multivariate")),
            FeatureSpec("amount_log", "log1p(tutar)", ("column", "multivariate")),
            FeatureSpec("is_mobile", "DeviceType = mobile", c),
            FeatureSpec("is_high_value_customer", "Geçmiş ortalama tutarı high_value_quantile üstünde", cr),
        ]


class MissingnessFeatures(FeatureBuilder):
    """Boşluğun kendisi sinyal (Adım 2): seçili kolonlar için _isna bayrakları + satırdaki boş kolon sayısı."""

    group = "context"

    def __init__(self, cfg: FeaturesConfig):
        self.cfg = cfg
        self.raw_columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> MissingnessFeatures:
        self.raw_columns_ = list(df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = {f"{c}_isna": df[c].isna().astype("int8") for c in self.cfg.null_flag_columns}
        cols = [c for c in self.raw_columns_ if c in df]
        new["null_count"] = df[cols].isna().sum(axis=1).astype("int16")
        return add_columns(df, new)

    def required_columns(self) -> list[str]:
        return list(self.cfg.null_flag_columns)

    def specs(self) -> list[FeatureSpec]:
        specs = [FeatureSpec(f"{c}_isna", f"{c} boş (boşluk fraud ile ilişkili)", ("multivariate", "context"))
                 for c in self.cfg.null_flag_columns]
        specs.append(FeatureSpec("null_count", "Satırdaki boş ham kolon sayısı (eksiklik profili)", ("multivariate",)))
        return specs
