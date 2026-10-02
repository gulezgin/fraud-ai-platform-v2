import numpy as np
import pandas as pd

from fraud_platform.data.profiler import (
    combination_key,
    compare_entity_keys,
    correlated_groups,
    null_target_profile,
    robust_z,
)


def test_robust_z_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = robust_z(s)
    assert z.iloc[2] == 0           # medyan
    assert z.iloc[-1] > 3.5         # uç değer


def test_robust_z_falls_back_when_mad_is_zero():
    # değerlerin çoğu aynı -> MAD = 0; sıfıra bölme yerine ortalama mutlak sapma kullanılmalı
    s = pd.Series([1.0] * 8 + [5.0, 50.0])
    z = robust_z(s)
    assert np.isfinite(z).all()
    assert z.iloc[-1] > z.iloc[-2] > 0


def test_robust_z_constant_series():
    z = robust_z(pd.Series([3.0, 3.0, 3.0]))
    assert (z == 0).all()


def test_combination_key_treats_nan_as_value():
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", None]})
    key = combination_key(df, ["a", "b"])
    assert key[0] == key[1]
    assert key.nunique() == 3


def test_null_target_profile():
    df = pd.DataFrame({"addr1": [None, None, 1.0, 2.0], "y": [1, 1, 0, 0]})
    out = null_target_profile(df, ["addr1"], "y")
    assert out.loc["addr1", "fraud_rate_null"] == 1.0
    assert out.loc["addr1", "fraud_rate_present"] == 0.0


def test_correlated_groups():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": x * 2 + 0.001, "c": rng.normal(size=200)})
    groups = correlated_groups(df, ["a", "b", "c"], threshold=0.95)
    assert sorted(map(sorted, groups)) == [["a", "b"], ["c"]]


def test_compare_entity_keys_purity():
    # card=1 iki farklı kişi; addr ile ayrılınca fraud'lar tek entity'de toplanıyor
    df = pd.DataFrame({
        "card": [1, 1, 1, 1],
        "addr": [10, 10, 20, 20],
        "y": [1, 1, 0, 0],
    })
    out = compare_entity_keys(df, {"card": ["card"], "card+addr": ["card", "addr"]}, "y")
    assert out.loc["card", "purity_fraud"] == 0.0
    assert out.loc["card+addr", "purity_fraud"] == 1.0
