"""Temporal katman: zamanlaması tuhaf mı?

Velocity (son 1 saat / 24 saat / 7 gün işlem sayısı), önceki işlemle arasındaki kısa süre, 1 dakikadan kısa tekrar,
nadir saat (eğitimdeki saat dağılımından) ve 24 saatlik tutar patlaması.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from fraud_platform.detection.base import WeightedComponentDetector, saturate


class TemporalDetector(WeightedComponentDetector):
    name = "temporal"

    TEXTS: ClassVar[dict[str, str]] = {
        "velocity_1h": "Kullanıcının son 1 saatte {v1h:.0f} işlemi daha var",
        "velocity_24h": "Kullanıcının son 24 saatte {v24h:.0f} işlemi daha var",
        "velocity_7d": "Kullanıcının son 7 günde {v7d:.0f} işlemi daha var",
        "gap": "Önceki işlemden {gap_min:.1f} dakika sonra",
        "rapid_repeat": "Önceki işlemden 1 dakikadan kısa süre sonra tekrar",
        "rare_hour": "Yerel saat {hour}:00, işlem hacminin düşük olduğu saat (payı %{hour_share:.1f})",
        "amount_burst": "Son 24 saatte toplam {sum24:.2f} USD harcama (kullanıcı ortalamasının {burst:.1f} katı)",
    }
    VALUE_OF: ClassVar[dict[str, str]] = {
        "velocity_1h": "v1h", "velocity_24h": "v24h", "velocity_7d": "v7d", "gap": "gap_min",
        "rapid_repeat": "gap_min", "rare_hour": "hour", "amount_burst": "sum24",
    }

    def fit(self, df: pd.DataFrame) -> TemporalDetector:
        self.hour_freq_ = df["local_hour"].value_counts(normalize=True)
        return self

    def _burst_ratio(self, df: pd.DataFrame) -> pd.Series:
        base = df["user_avg_amount"].fillna(df[self.amount_col])
        return df["amount_sum_24h"] / (base + 1)

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.caps
        hour_share = df["local_hour"].map(self.hour_freq_).astype("float64").fillna(0.0)
        all_components = {
            "velocity_1h": saturate(df["tx_count_1h"], c["velocity_1h"]),
            "velocity_24h": saturate(df["tx_count_24h"], c["velocity_24h"]),
            "velocity_7d": saturate(df["tx_count_7d"], c["velocity_7d"]),
            # 0 sn -> 1, 1 saat -> 0,37, geçmiş yoksa 0
            "gap": np.exp(-df["time_since_last_tx"] / c["gap_seconds"]).fillna(0.0),
            "rapid_repeat": df["is_rapid_repeat"].astype("float64"),
            "rare_hour": 1 - hour_share / self.hour_freq_.max(),
            "amount_burst": saturate(self._burst_ratio(df), c["amount_burst"]),
        }
        return pd.DataFrame({k: all_components[k] for k in self.weights}, index=df.index)

    def component_values(self, row: pd.Series) -> dict[str, object]:
        base = row["user_avg_amount"] if pd.notna(row["user_avg_amount"]) else row[self.amount_col]
        return {
            "v1h": row["tx_count_1h"], "v24h": row["tx_count_24h"], "v7d": row["tx_count_7d"],
            "gap_min": row["time_since_last_tx"] / 60 if pd.notna(row["time_since_last_tx"]) else np.nan,
            "hour": int(row["local_hour"]), "hour_share": self.hour_freq_.get(row["local_hour"], 0.0) * 100,
            "sum24": row["amount_sum_24h"], "burst": row["amount_sum_24h"] / (base + 1),
        }
