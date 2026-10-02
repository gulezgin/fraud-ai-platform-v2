import numpy as np
import pandas as pd
import pytest

from fraud_platform.detection.column import ColumnDetector
from fraud_platform.detection.entity import EntityDetector
from fraud_platform.detection.multivariate import MultivariateDetector
from fraud_platform.detection.temporal import TemporalDetector
from fraud_platform.evaluation import alert_metrics, time_split


@pytest.fixture
def numeric_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "amt": rng.normal(100, 10, 1000),
        "flat": np.r_[np.ones(900), rng.integers(2, 5, 100)],   # MAD = 0
        "cat": rng.choice(["a", "b"], 1000, p=[0.9, 0.1]),
    })


def test_column_detector_flags_extreme_and_unseen(settings, numeric_frame):
    det = ColumnDetector(["amt", "flat"], ["cat"], settings.detectors.column).fit(numeric_frame)
    new = pd.DataFrame({"amt": [100.0, 400.0], "flat": [1.0, 1.0], "cat": ["a", "zzz"]})
    s = det.score(new)
    assert s.iloc[1] > s.iloc[0]
    assert np.isfinite(det.scale_["flat"])          # MAD=0 kolonu sonsuz skor üretmemeli

    reasons = det.explain(new.iloc[[1]])[0]
    assert {r.feature for r in reasons} == {"amt", "cat"}
    assert any("hiç görülmemiş" in r.text for r in reasons)


def test_column_detector_ignores_missing(settings, numeric_frame):
    det = ColumnDetector(["amt"], ["cat"], settings.detectors.column).fit(numeric_frame)
    s = det.score(pd.DataFrame({"amt": [np.nan], "cat": [None]}))
    assert s.iloc[0] == 0


def test_multivariate_explain_uses_tail_probability(settings):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"flag": (rng.random(5000) < 0.11).astype(float), "x": rng.normal(size=5000)})
    det = MultivariateDetector(["flag", "x"], settings.detectors.multivariate).fit(df)

    row = pd.DataFrame({"flag": [1.0], "x": [0.0]})
    reasons = det.explain(row)[0]
    flag_reason = next(r for r in reasons if r.feature == "flag")
    # %11 açık olan bayrak "%100 uç" sayılmamalı: kuyruk ~%11 -> uçluk ~0,78
    assert 0.7 < flag_reason.contribution < 0.85
    assert len(det.score(df.head(10))) == 10


def test_weighted_detector_rejects_unknown_component(settings):
    with pytest.raises(ValueError, match="bilinmeyen"):
        EntityDetector({"yok_boyle_bilesen": 1}, {}, "TransactionAmt")


def test_temporal_detector_scores_in_unit_interval(settings):
    cfg = settings.detectors.temporal
    df = pd.DataFrame({
        "local_hour": [3, 14, 14], "tx_count_1h": [9, 0, 0], "tx_count_24h": [30, 0, 1], "tx_count_7d": [60, 0, 2],
        "time_since_last_tx": [10.0, np.nan, 86400.0], "is_rapid_repeat": [1, 0, 0],
        "amount_sum_24h": [5000.0, 0.0, 20.0], "user_avg_amount": [50.0, np.nan, 20.0], "TransactionAmt": [80.0, 30.0, 25.0],
    })
    det = TemporalDetector(cfg.weights, cfg.caps, "TransactionAmt").fit(df)
    s = det.score(df)
    assert s.between(0, 1).all()
    assert s.iloc[0] > s.iloc[2] > s.iloc[1] - 1e-9
    assert det.explain(df.iloc[[0]])[0][0].contribution == 1.0


def test_time_split_is_ordered(settings):
    t = pd.Series(np.arange(100))
    split = time_split(t, settings.evaluation)
    assert split.iloc[0] == "train" and split.iloc[-1] == "test"
    assert t[split == "valid"].min() > t[split == "train"].max()


def test_alert_metrics():
    y = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = pd.Series([0.9, 0.8, 0.1, 0.2, 0, 0, 0, 0, 0, 0])
    m = alert_metrics(y, score, alert_rate=0.2)
    assert m == {"alerts": 2, "tp": 1, "fp": 1, "precision": 0.5, "recall": 0.5}
