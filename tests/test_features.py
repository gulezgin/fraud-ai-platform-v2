import numpy as np
import pandas as pd
import pytest

from fraud_platform.features.context import ContextFeatures, _in_window
from fraud_platform.features.entity import EntityKeys
from fraud_platform.features.history import (
    circular_hour_distance,
    past_count,
    past_distinct_count,
    past_mean_std,
    past_window,
)
from fraud_platform.features.pipeline import build_pipeline


def test_past_mean_std_uses_only_previous_rows():
    g = pd.Series(["a", "a", "a", "b"])
    v = pd.Series([10.0, 20.0, 60.0, 5.0])
    mean, std = past_mean_std(v, g)
    assert np.isnan(mean[0]) and np.isnan(mean[3])   # ilk işlemlerin geçmişi yok
    assert mean[1] == 10 and mean[2] == 15
    assert np.isnan(std[1]) and std[2] == pytest.approx(5.0)


def test_past_count_and_distinct():
    g = pd.Series(["a", "a", "a", "a"])
    dev = pd.Series(["x", "x", "y", None])
    assert past_count(g).tolist() == [0, 1, 2, 3]
    assert past_distinct_count(g, dev).tolist() == [0, 1, 1, 2]


def test_past_window_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    t = pd.Series(np.sort(rng.integers(0, 20_000, n)))
    g = pd.Series(rng.choice(["a", "b", "c"], n))
    v = pd.Series(rng.uniform(1, 100, n))
    window = 3600

    count, total = past_window(t, g, window, v)

    for i in range(n):
        prev = (g[:i] == g[i]) & (t[:i] >= t[i] - window)
        assert count[i] == prev.sum()
        assert total[i] == pytest.approx(v[:i][prev].sum())


def test_circular_hour_distance():
    a, b = pd.Series([23, 1, 12]), pd.Series([1, 23, 0])
    assert circular_hour_distance(a, b).tolist() == [2, 2, 12]


def test_in_window_wraps_midnight():
    hours = pd.Series([21, 23, 0, 5, 6, 12])
    assert _in_window(hours, (22, 6)).tolist() == [False, True, True, True, False, False]
    assert _in_window(hours, (0, 6)).tolist() == [False, False, True, True, False, False]


def test_amount_is_converted_survives_float32(settings):
    # float32'de 47.95 = 47.950001; yanlışlıkla 3 haneli sayılmamalı
    df = pd.DataFrame({
        "TransactionAmt": np.array([47.95, 159.95, 31937.391, 0.251, 100.0], dtype="float32"),
        "local_hour": 12, "day_of_week": 1, "addr2": 87.0, "D1": 10.0, "DeviceType": None,
        "user_avg_amount": 50.0,
    })
    out = ContextFeatures(settings.features, settings.time, settings.data).fit_transform(df)
    assert out["amount_is_converted"].tolist() == [0, 0, 1, 1, 0]


def test_uid_key_format(settings):
    df = pd.DataFrame({
        "TransactionDT": [86400 * 10, 86400 * 12],
        "card1": [1001, 1001], "addr1": [315.0, None], "D1": [3.0, 5.0],
        "DeviceInfo": [None, "windows"], "id_31": [None, None], "id_33": [None, None],
        "id_19": [None, None], "id_20": [None, None],
    })
    out = EntityKeys(settings.entity, settings.data).transform(df)
    assert out["uid"].tolist() == ["1001_315_7", "1001_NA_7"]
    assert pd.isna(out["device_key"][0]) and out["device_key"][1].startswith("windows|")


def test_pipeline_has_no_future_leakage(settings):
    """Son satırları değiştirmek önceki satırların feature'larını değiştirmemeli."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "TransactionID": np.arange(n), "isFraud": 0,
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 5, n)),
        "TransactionAmt": rng.uniform(5, 500, n).astype("float32"),
        "ProductCD": "W", "card1": rng.choice([1, 2], n), "card2": 100.0, "card4": "visa", "card5": 226.0,
        "card6": "debit", "addr1": 315.0, "addr2": 87.0, "dist1": np.nan, "D1": 0.0, "D4": 0.0, "D11": 0.0,
        "D15": 0.0, "P_emaildomain": "gmail.com", "R_emaildomain": None, "M1": "T", "M4": "M0", "M6": "F",
        "M7": None, "DeviceType": None, "DeviceInfo": None, "id_19": np.nan, "id_20": np.nan,
        "id_31": None, "id_33": None,
    })
    pipe = build_pipeline(settings)
    base = pipe.fit_transform(df)

    changed = df.copy()
    changed.loc[n - 10:, "TransactionAmt"] = 99999.0
    after = pipe.transform(changed)

    past_cols = ["user_avg_amount", "amount_to_user_avg", "tx_count_24h", "amount_sum_24h", "user_tx_count"]
    pd.testing.assert_frame_equal(base.loc[: n - 11, past_cols], after.loc[: n - 11, past_cols])
