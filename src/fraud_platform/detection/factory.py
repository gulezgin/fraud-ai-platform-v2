"""Dedektörleri config + şema + profil çıktılarından oluşturur (Factory).

Kolon listeleri elle yazılmaz: column katmanı şemadaki sayısal temsilciler ve kategorik kolonları,
multivariate katman üretilen feature'lar + sayısal temsilcileri kullanır. Başka veri setinde de aynı kod çalışır.
"""
from __future__ import annotations

import json
from pathlib import Path

from fraud_platform.config import Settings
from fraud_platform.data.schema import Schema, SemanticType
from fraud_platform.detection.base import BaseDetector
from fraud_platform.detection.column import ColumnDetector
from fraud_platform.detection.entity import EntityDetector
from fraud_platform.detection.multivariate import MultivariateDetector
from fraud_platform.detection.temporal import TemporalDetector

LAYERS = ("column", "multivariate", "entity", "temporal")
KEY_FEATURES = {"uid", "device_key"}


def build_detectors(settings: Settings, schema: Schema, profile: dict, feature_names: list[str]) -> list[BaseDetector]:
    cfg, amt = settings.detectors, settings.data.amount_column
    numeric = profile["numeric_representatives"]
    categorical = schema.of_type(SemanticType.CATEGORICAL, SemanticType.BINARY)
    engineered = [f for f in feature_names if f not in KEY_FEATURES]

    return [
        ColumnDetector(numeric, categorical, cfg.column),
        MultivariateDetector(list(dict.fromkeys(engineered + numeric)), cfg.multivariate),
        EntityDetector(cfg.entity.weights, cfg.entity.caps, amt),
        TemporalDetector(cfg.temporal.weights, cfg.temporal.caps, amt),
    ]


def build_detectors_from_artifacts(settings: Settings, feature_names: list[str]) -> list[BaseDetector]:
    schema = Schema.load(settings.schema_path)
    profile = json.loads(settings.profile_path.read_text(encoding="utf-8"))
    return build_detectors(settings, schema, profile, feature_names)


def load_detectors(directory: Path) -> list[BaseDetector]:
    return [BaseDetector.load(directory / f"{name}.joblib") for name in LAYERS]
