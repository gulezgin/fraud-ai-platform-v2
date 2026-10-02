"""Tek çağrıda skorlama (Facade): ham işlem -> online feature -> 4 dedektör -> birleşik skor -> context düzeltmesi.

API, agent'lar ve script'ler alt modülleri tek tek bilmek yerine bu servisi kullanır.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_platform.config import Settings
from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.detection.aggregator import ScoreAggregator
from fraud_platform.detection.base import BaseDetector
from fraud_platform.detection.factory import load_detectors
from fraud_platform.features.online import OnlineFeatureBuilder
from fraud_platform.features.pipeline import FeaturePipeline
from fraud_platform.store.profile_store import EntityProfileStore


class ScoringService:
    def __init__(self, online: OnlineFeatureBuilder, detectors: list[BaseDetector], aggregator: ScoreAggregator,
                 context: ContextAdjuster, id_column: str):
        self.online = online
        self.detectors = detectors
        self.aggregator = aggregator
        self.context = context
        self.id_column = id_column

    @classmethod
    def from_artifacts(cls, settings: Settings) -> ScoringService:
        pipeline = FeaturePipeline.load(settings.feature_pipeline_path)
        store = EntityProfileStore.load(settings.profile_store_path)
        return cls(
            OnlineFeatureBuilder(pipeline, store),
            load_detectors(settings.detectors_path),
            ScoreAggregator.load(settings.aggregator_path),
            ContextAdjuster(settings.context.rules_file),
            settings.data.id_column,
        )

    def features(self, transactions: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        tx = pd.DataFrame(transactions) if not isinstance(transactions, pd.DataFrame) else transactions
        feats = self.online.build(tx)
        # API'den kısmi işlem gelebilir (435 alanın hepsi değil): dedektörlerin okuduğu eksik ham kolonlar boş sayılır
        needed = dict.fromkeys(c for d in self.detectors for c in d.input_columns())
        missing = [c for c in needed if c not in feats]
        if not missing:
            return feats
        # tek concat: yüzlerce kolonu assign ile tek tek eklemek DataFrame'i parçalıyor (PerformanceWarning)
        return pd.concat([feats, pd.DataFrame(np.nan, index=feats.index, columns=missing)], axis=1)

    def score_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        layer = pd.DataFrame({d.name: d.score(features) for d in self.detectors}, index=features.index)
        scores = layer.join(self.aggregator.transform(layer))
        context = self.context.adjust(features, scores["raw_anomaly_score"])
        # düzeltilmiş skorun, eğitimdeki ham skor dağılımındaki yeri: 0,97 = "en riskli %3 kadar riskli"
        context["risk_percentile"] = self.aggregator.final_pct_(context["adjusted_score"])
        return scores.join(context)

    def assessment_frame(self, transactions: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Feature'lar + tüm skorlar tek tabloda: kural motoru koşullarını bu alanlar üzerinden değerlendirir."""
        feats = self.features(transactions)
        return feats.join(self.score_frame(feats))

    def score(self, transactions: pd.DataFrame | list[dict[str, Any]], explain: bool = True,
              top_k: int = 3) -> list[dict[str, Any]]:
        return self.score_features(self.features(transactions), explain, top_k)[0]

    def score_features(self, feats: pd.DataFrame, explain: bool = True,
                       top_k: int = 3) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        """Hazır feature'ları skorlar: (okunabilir sonuçlar, feature + skor tablosu). Feature bir kez hesaplanır."""
        scores = self.score_frame(feats)
        reasons = {d.name: d.explain(feats, top_k) for d in self.detectors} if explain else {}

        results = []
        for i, idx in enumerate(feats.index):
            s = scores.loc[idx]
            result = {
                self.id_column: int(feats.at[idx, self.id_column]),
                "uid": feats.at[idx, "uid"],
                "layer_scores": {l: round(float(s[l]), 6) for l in self.aggregator.layers},
                "layer_percentiles": {l: round(float(s[f"{l}_pct"]), 6) for l in self.aggregator.layers},
                "raw_anomaly_score": round(float(s["raw_anomaly_score"]), 6),
                "raw_percentile": round(float(s["raw_percentile"]), 6),
                "context_factor": round(float(s["context_factor"]), 4),
                "adjusted_score": round(float(s["adjusted_score"]), 6),
                "risk_percentile": round(float(s["risk_percentile"]), 6),
                "context_adjustments": self.context.explain(feats.loc[idx]),
            }
            if explain:
                result["reasons"] = {name: [r.to_dict() for r in rs[i]] for name, rs in reasons.items()}
            results.append(result)
        return results, feats.join(scores)
