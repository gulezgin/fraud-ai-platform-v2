import pandas as pd
import pytest
from helpers import make_transactions

from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.detection.aggregator import ScoreAggregator
from fraud_platform.detection.column import ColumnDetector
from fraud_platform.detection.entity import EntityDetector
from fraud_platform.detection.multivariate import MultivariateDetector
from fraud_platform.detection.temporal import TemporalDetector
from fraud_platform.features.online import OnlineFeatureBuilder
from fraud_platform.features.pipeline import build_pipeline
from fraud_platform.services.scoring_service import ScoringService
from fraud_platform.store.profile_store import EntityProfileStore


def build_service(settings) -> tuple[ScoringService, pd.DataFrame]:
    raw = make_transactions(n=120)
    pipe = build_pipeline(settings)
    feats = pipe.fit_transform(raw)

    cfg, amt = settings.detectors, "TransactionAmt"
    engineered = [f for f in pipe.feature_names if f not in ("uid", "device_key")]
    detectors = [
        ColumnDetector(["C1", "TransactionAmt"], ["ProductCD", "P_emaildomain"], cfg.column),
        MultivariateDetector(engineered, cfg.multivariate.model_copy(update={"fit_sample_size": 120, "max_samples": 64})),
        EntityDetector(cfg.entity.weights, cfg.entity.caps, amt),
        TemporalDetector(cfg.temporal.weights, cfg.temporal.caps, amt),
    ]
    layer = pd.DataFrame({d.name: d.fit(feats).score(feats) for d in detectors})
    aggregator = ScoreAggregator(settings.scoring).fit(layer)

    store = EntityProfileStore.from_features(feats, pipe.required_columns(), "TransactionDT", "TransactionID", amt)
    context = ContextAdjuster(settings.context.rules_file)
    return ScoringService(OnlineFeatureBuilder(pipe, store), detectors, aggregator, context, "TransactionID"), raw


def test_score_end_to_end(settings):
    service, raw = build_service(settings)
    tx = raw.drop(columns="isFraud").iloc[[100]].to_dict(orient="records")

    result = service.score(tx)[0]

    assert result["TransactionID"] == raw["TransactionID"].iloc[100]
    assert set(result["layer_scores"]) == {"column", "multivariate", "entity", "temporal"}
    assert 0 <= result["raw_anomaly_score"] <= 1 and 0 <= result["raw_percentile"] <= 1
    assert result["adjusted_score"] == pytest.approx(min(result["raw_anomaly_score"] * result["context_factor"], 1), abs=1e-4)
    assert all({"id", "factor", "reason"} <= set(a) for a in result["context_adjustments"])
    assert set(result["reasons"]) == {"column", "multivariate", "entity", "temporal"}
    assert all(isinstance(r["text"], str) for rs in result["reasons"].values() for r in rs)


def test_score_without_explain(settings):
    service, raw = build_service(settings)
    result = service.score(raw.drop(columns="isFraud").iloc[[5, 6]], explain=False)
    assert len(result) == 2 and "reasons" not in result[0]
