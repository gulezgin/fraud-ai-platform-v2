"""Feature builder'ları sırayla çalıştıran pipeline ve feature sözlüğü.

Sıra önemli: anahtarlar (uid) -> zaman (saat, velocity uid'ye ihtiyaç duyar) -> entity (saat sapması saate ihtiyaç duyar)
-> ilişkisel -> context. Yeni bir feature grubu = yeni bir FeatureBuilder sınıfı + build_pipeline'a bir satır.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fraud_platform.config import Settings
from fraud_platform.features.base import FeatureBuilder
from fraud_platform.features.cleaning import DataCleaner
from fraud_platform.features.context import ContextFeatures, MissingnessFeatures
from fraud_platform.features.entity import EntityFeatures, EntityKeys
from fraud_platform.features.relational import RelationalFeatures
from fraud_platform.features.temporal import TemporalFeatures


class FeaturePipeline:
    def __init__(self, builders: list[FeatureBuilder], time_column: str, id_column: str):
        self.builders = builders
        self.sort_by = [time_column, id_column]

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        # past-only hesaplar zamana göre sıralı veri ister
        return df.sort_values(self.sort_by, kind="stable").reset_index(drop=True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sorted(df)
        for b in self.builders:
            df = b.fit(df).transform(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sorted(df)
        for b in self.builders:
            df = b.transform(df)
        return df

    def dictionary(self) -> pd.DataFrame:
        rows = [
            {"feature": s.name, "group": b.group, "builder": type(b).__name__,
             "description": s.description, "layers": ", ".join(s.layers)}
            for b in self.builders for s in b.specs()
        ]
        return pd.DataFrame(rows)

    @property
    def feature_names(self) -> list[str]:
        return [n for b in self.builders for n in b.feature_names]

    def required_columns(self) -> list[str]:
        cols = [*self.sort_by, *(c for b in self.builders for c in b.required_columns())]
        return list(dict.fromkeys(cols))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> FeaturePipeline:
        return joblib.load(path)


def build_pipeline(settings: Settings) -> FeaturePipeline:
    d, f, t = settings.data, settings.features, settings.time
    builders = [
        DataCleaner(f),
        MissingnessFeatures(f),       # ham kolonları görmesi için feature'lardan önce
        EntityKeys(settings.entity, d),
        TemporalFeatures(t, d),
        EntityFeatures(d),
        RelationalFeatures(f),
        ContextFeatures(f, t, d),
    ]
    return FeaturePipeline(builders, time_column=d.time_column, id_column=d.id_column)
