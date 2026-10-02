"""Zaman feature'ları: yerel saat, gün, döngüsel saat kodlaması ve uid bazında velocity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.config import DataConfig, TimeConfig
from fraud_platform.features.base import FeatureBuilder, FeatureSpec, add_columns
from fraud_platform.features.history import past_window


class TemporalFeatures(FeatureBuilder):
    group = "temporal"

    def __init__(self, cfg: TimeConfig, data_cfg: DataConfig):
        self.cfg = cfg
        self.time_col = data_cfg.time_column
        self.amt = data_cfg.amount_column
        self.reference = pd.Timestamp(data_cfg.reference_date)

    def local_datetime(self, seconds: pd.Series) -> pd.Series:
        # saat dilimi varsayımı: hour_offset (settings.yaml -> time)
        return self.reference + pd.to_timedelta(seconds + self.cfg.hour_offset * 3600, unit="s")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df[self.time_col]
        local = self.local_datetime(t)
        hour = local.dt.hour + local.dt.minute / 60
        angle = 2 * np.pi * hour / 24

        new = {
            "local_hour": local.dt.hour.astype("int8"),
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
            "day_of_week": local.dt.dayofweek.astype("int8"),
        }

        last_gap = t.groupby(df["uid"], sort=False).diff()
        new["time_since_last_tx"] = last_gap
        new["is_rapid_repeat"] = (last_gap < self.cfg.rapid_repeat_seconds).astype("int8")

        for label, seconds in self.cfg.velocity_windows.items():
            if label == "24h":
                count, total = past_window(t, df["uid"], seconds, df[self.amt])
                new["amount_sum_24h"] = total
            else:
                count = past_window(t, df["uid"], seconds)
            new[f"tx_count_{label}"] = count

        return add_columns(df, new)

    def required_columns(self) -> list[str]:
        return [self.time_col, self.amt]

    def specs(self) -> list[FeatureSpec]:
        specs = [
            FeatureSpec("local_hour", "Yerel saat (TransactionDT + hour_offset)", ("temporal", "context", "rules")),
            FeatureSpec("hour_sin", "Saatin döngüsel kodlaması (sin)", ("multivariate",)),
            FeatureSpec("hour_cos", "Saatin döngüsel kodlaması (cos)", ("multivariate",)),
            FeatureSpec("day_of_week", "Haftanın günü (0=Pzt, referans tarih varsayımı)", ("temporal", "context")),
            FeatureSpec("time_since_last_tx", "Aynı kullanıcının önceki işleminden beri geçen saniye", ("temporal", "multivariate", "rules")),
            FeatureSpec("is_rapid_repeat", "Önceki işlemden bu yana rapid_repeat_seconds'tan az geçti", ("temporal", "rules")),
            FeatureSpec("amount_sum_24h", "Kullanıcının son 24 saatteki önceki işlemlerinin toplam tutarı", ("temporal", "rules")),
        ]
        specs += [
            FeatureSpec(f"tx_count_{label}", f"Kullanıcının son {label} içindeki önceki işlem sayısı (velocity)",
                        ("temporal", "multivariate", "rules"))
            for label in self.cfg.velocity_windows
        ]
        return specs
