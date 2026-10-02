"""Entity anahtarları (uid, cihaz) ve kullanıcının kendi geçmişine göre sapma feature'ları.

Tüm istatistikler sadece geçmişten hesaplanır (bkz. history.py). Kullanıcının ilk işleminde geçmiş olmadığı için
ortalama/sapma feature'ları NaN kalır; "ilk işlem" bilgisi is_first_tx ile ayrıca verilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.config import DataConfig, EntityConfig
from fraud_platform.features.base import FeatureBuilder, FeatureSpec, add_columns
from fraud_platform.features.history import (
    circular_hour_distance,
    first_seen,
    past_count,
    past_distinct_count,
    past_mean_std,
)

SECONDS_PER_DAY = 86400


def _as_key_part(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s):
        s = s.round().astype("Int64")   # 315.0 -> "315"
    return s.astype("string").fillna("NA")


class EntityKeys(FeatureBuilder):
    """uid = card1 + addr1 + (işlem günü - D1), device_key = cihaz parmak izi."""

    group = "entity"

    def __init__(self, cfg: EntityConfig, data_cfg: DataConfig):
        self.cfg = cfg
        self.time_col = data_cfg.time_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [_as_key_part(df[c]) for c in self.cfg.uid_columns]
        if self.cfg.anchor_days_column:
            start_day = df[self.time_col] // SECONDS_PER_DAY - df[self.cfg.anchor_days_column]
            parts.append(_as_key_part(start_day))
        uid = parts[0].str.cat(parts[1:], sep="_")

        dev = df[self.cfg.device_columns]
        device_key = _as_key_part(dev.iloc[:, 0]).str.cat([_as_key_part(dev[c]) for c in dev.columns[1:]], sep="|")
        device_key = device_key.where(dev.notna().any(axis=1))   # hiç cihaz bilgisi yoksa anahtar da yok

        return add_columns(df, {"uid": uid, "device_key": device_key})

    def required_columns(self) -> list[str]:
        anchor = [self.cfg.anchor_days_column] if self.cfg.anchor_days_column else []
        return [*self.cfg.uid_columns, *anchor, *self.cfg.device_columns, self.time_col]

    def specs(self) -> list[FeatureSpec]:
        return [
            FeatureSpec("uid", "Sözde kullanıcı: card1 + addr1 + (işlem günü - D1)", ("entity",)),
            FeatureSpec("device_key", "Cihaz parmak izi: DeviceInfo|id_31|id_33|id_19|id_20", ("entity",)),
        ]


class EntityFeatures(FeatureBuilder):
    """Kullanıcının bu işlemden önceki davranışına göre sapma."""

    group = "entity"

    def __init__(self, data_cfg: DataConfig):
        self.amt = data_cfg.amount_column
        self.time_col = data_cfg.time_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        uid, amt = df["uid"], df[self.amt]
        n = past_count(uid)
        mean, std = past_mean_std(amt, uid)

        # saatin dairesel ortalaması: 23 ve 1'in ortalaması 12 değil 0 olmalı
        sin_mean, _ = past_mean_std(df["hour_sin"], uid)
        cos_mean, _ = past_mean_std(df["hour_cos"], uid)
        typical_hour = (np.degrees(np.arctan2(sin_mean, cos_mean)) / 15) % 24

        first_t = df.groupby(uid, sort=False)[self.time_col].transform("min")

        new = {
            "user_tx_count": n,
            "is_first_tx": (n == 0).astype("int8"),
            "user_avg_amount": mean,
            "user_std_amount": std,
            "amount_to_user_avg": amt / mean,
            "amount_user_zscore": (amt - mean) / (std + 1),
            "user_n_devices": past_distinct_count(uid, df["device_key"]),
            "is_new_device_for_user": (first_seen(uid, df["device_key"]) & (n > 0)).astype("int8"),
            "user_n_emails": past_distinct_count(uid, df["P_emaildomain"]),
            "is_new_email_for_user": (first_seen(uid, df["P_emaildomain"]) & (n > 0)).astype("int8"),
            "hour_dev_from_user": circular_hour_distance(df["local_hour"], typical_hour),
            "days_since_first_tx": (df[self.time_col] - first_t) / SECONDS_PER_DAY,
        }
        return add_columns(df, new)

    def required_columns(self) -> list[str]:
        return [self.amt, self.time_col, "P_emaildomain"]

    def specs(self) -> list[FeatureSpec]:
        e, em = ("entity",), ("entity", "multivariate")
        return [
            FeatureSpec("user_tx_count", "Kullanıcının bu işlemden önceki işlem sayısı", em),
            FeatureSpec("is_first_tx", "Kullanıcının ilk işlemi (geçmiş yok)", ("entity", "context")),
            FeatureSpec("user_avg_amount", "Önceki işlemlerin ortalama tutarı", e),
            FeatureSpec("user_std_amount", "Önceki işlemlerin tutar std'si (en az 2 işlem)", e),
            FeatureSpec("amount_to_user_avg", "Tutar / kullanıcının geçmiş ortalaması", ("entity", "multivariate", "rules")),
            FeatureSpec("amount_user_zscore", "(tutar - geçmiş ort.) / (geçmiş std + 1)", em),
            FeatureSpec("user_n_devices", "Kullanıcının daha önce kullandığı farklı cihaz sayısı", em),
            FeatureSpec("is_new_device_for_user", "Geçmişi olan kullanıcı ilk kez bu cihazı kullanıyor", ("entity", "rules")),
            FeatureSpec("user_n_emails", "Kullanıcının daha önce kullandığı farklı e-posta domaini sayısı", em),
            FeatureSpec("is_new_email_for_user", "Geçmişi olan kullanıcı ilk kez bu e-posta domainini kullanıyor", e),
            FeatureSpec("hour_dev_from_user", "İşlem saatinin kullanıcının alışık saatinden dairesel farkı (0-12)", e),
            FeatureSpec("days_since_first_tx", "Kullanıcının veride ilk görüldüğü andan bu yana gün", em),
        ]
