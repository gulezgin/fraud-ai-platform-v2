"""Entity'ler arası ilişki feature'ları: cihaz paylaşımı, e-posta eşleşmesi, frekans ve kombinasyon nadirliği.

Cihaz paylaşımı sayıları sadece geçmişten; frekans tabloları ise eğitim verisinden fit edilip saklanır.
"""
from __future__ import annotations

import pandas as pd

from fraud_platform.config import FeaturesConfig
from fraud_platform.features.base import FeatureBuilder, FeatureSpec, add_columns
from fraud_platform.features.history import past_distinct_count


def _combo_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[c].astype("string").fillna("NA") for c in columns]
    return parts[0].str.cat(parts[1:], sep="|")


class RelationalFeatures(FeatureBuilder):
    group = "relational"

    def __init__(self, cfg: FeaturesConfig):
        self.cfg = cfg
        self.freq_: dict[str, pd.Series] = {}
        self.combo_freq_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> RelationalFeatures:
        self.freq_ = {c: df[c].value_counts(normalize=True) for c in self.cfg.frequency_columns}
        self.combo_freq_ = _combo_key(df, self.cfg.combo_columns).value_counts(normalize=True)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        device = df["device_key"]
        p_mail, r_mail = df["P_emaildomain"], df["R_emaildomain"]

        new = {
            "device_n_uids": past_distinct_count(device, df["uid"]).where(device.notna()),
            "device_n_cards": past_distinct_count(device, df["card1"]).where(device.notna()),
            "email_match": (p_mail.notna() & (p_mail == r_mail)).astype("int8"),
        }
        # eğitimde hiç görülmemiş değer -> frekans 0 (en nadir)
        for c, freq in self.freq_.items():
            new[f"{c}_freq"] = df[c].map(freq).astype("float64").fillna(0.0)
        new["combo_rarity"] = 1 - _combo_key(df, self.cfg.combo_columns).map(self.combo_freq_).astype("float64").fillna(0.0)

        return add_columns(df, new)

    def required_columns(self) -> list[str]:
        return ["card1", "P_emaildomain", "R_emaildomain", *self.cfg.frequency_columns, *self.cfg.combo_columns]

    def specs(self) -> list[FeatureSpec]:
        specs = [
            FeatureSpec("device_n_uids", "Bu cihazda daha önce görülen farklı kullanıcı sayısı", ("multivariate", "rules")),
            FeatureSpec("device_n_cards", "Bu cihazda daha önce görülen farklı kart (card1) sayısı", ("multivariate", "rules")),
            FeatureSpec("email_match", "Alıcı ve karşı taraf e-posta domaini aynı", ("multivariate", "rules")),
            FeatureSpec("combo_rarity", "ProductCD/card4/card6/e-posta/cihaz tipi kombinasyonunun nadirliği (1 - frekans)",
                        ("column", "multivariate")),
        ]
        specs += [
            FeatureSpec(f"{c}_freq", f"{c} frekans kodlaması (eğitim verisindeki payı)", ("column", "multivariate"))
            for c in self.cfg.frequency_columns
        ]
        return specs
