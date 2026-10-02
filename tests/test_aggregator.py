import numpy as np
import pandas as pd
import pytest

from fraud_platform.config import ScoringConfig
from fraud_platform.detection.aggregator import PercentileMap, ScoreAggregator, TailLogNormalizer


@pytest.fixture
def layer_scores():
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame({
        "column": rng.exponential(1, n),          # farklı ölçekler
        "multivariate": rng.normal(0.5, 0.05, n),
        "entity": rng.uniform(0, 1, n),
        "temporal": rng.exponential(10, n),
    })


def test_percentile_map_is_monotonic_and_bounded(layer_scores):
    pm = PercentileMap(tail_resolution=100).fit(layer_scores["column"])
    p = pm(pd.Series([-5.0, 0.5, 2.0, 1e9]))
    assert p.is_monotonic_increasing
    assert p.iloc[0] == 0 and p.iloc[-1] == 1


def test_tail_log_expands_the_tail(layer_scores):
    norm = TailLogNormalizer(tail_resolution=400).fit(layer_scores["column"])
    q = layer_scores["column"].quantile([0.5, 0.99, 0.999])
    v = norm(pd.Series(q.to_numpy()))
    log_n = np.log10(len(layer_scores))
    assert v.iloc[1] == pytest.approx(2 / log_n, rel=0.05)   # %99 -> -log10(0.01) = 2
    assert v.iloc[2] == pytest.approx(3 / log_n, rel=0.05)   # %99,9 -> 3
    # düz yüzdelikte %99 ile %99,9 arası 0,009; kuyruk-logda ~1/log_n
    assert v.iloc[2] - v.iloc[1] > 0.1


@pytest.mark.parametrize("method", ["percentile", "minmax", "tail_log"])
def test_aggregator_outputs(layer_scores, method):
    cfg = ScoringConfig(normalization=method, weights={"column": 2, "multivariate": 1, "entity": 1, "temporal": 0})
    agg = ScoreAggregator(cfg).fit(layer_scores)
    assert agg.weights == {"column": 0.5, "multivariate": 0.25, "entity": 0.25, "temporal": 0.0}

    out = agg.transform(layer_scores.head(100))
    assert {"column_pct", "raw_anomaly_score", "raw_percentile"} <= set(out.columns)
    assert out["raw_anomaly_score"].between(0, 1).all()
    assert out["raw_percentile"].between(0, 1).all()


def test_aggregator_ranks_extreme_row_highest(layer_scores):
    agg = ScoreAggregator(ScoringConfig(weights={"column": 1, "multivariate": 1, "entity": 1, "temporal": 1})).fit(layer_scores)
    rows = pd.DataFrame({"column": [0.5, 9.0], "multivariate": [0.5, 0.8], "entity": [0.5, 0.99], "temporal": [5.0, 90.0]})
    out = agg.transform(rows)
    assert out["raw_anomaly_score"].iloc[1] > out["raw_anomaly_score"].iloc[0]
    assert out["raw_percentile"].iloc[1] > 0.999


def test_scoring_config_rejects_negative_weights():
    with pytest.raises(ValueError):
        ScoringConfig(weights={"column": -1, "entity": 2})
