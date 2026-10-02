"""Sadece geçmişe bakan (past-only) grup hesapları.

Satırların zamana göre sıralı olduğu varsayılır. Her satır için yalnızca kendisinden ÖNCEKİ satırlar kullanılır;
böylece bir işlemin feature'ı gelecekteki işlemlerden bilgi taşımaz ve API'deki profil deposuyla aynı mantıkta olur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def past_count(group: pd.Series) -> pd.Series:
    return group.groupby(group, sort=False).cumcount()


def past_sum(values: pd.Series, group: pd.Series) -> pd.Series:
    return values.groupby(group, sort=False).cumsum() - values


def past_mean_std(values: pd.Series, group: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Önceki değerlerin ortalaması ve std'si (popülasyon). Geçmiş yoksa NaN, std için en az 2 değer gerekir."""
    v = values.astype("float64")
    n = past_count(group)
    s1 = past_sum(v, group)
    s2 = past_sum(v**2, group)
    mean = (s1 / n).where(n > 0)
    var = (s2 / n - mean**2).clip(lower=0)
    return mean, np.sqrt(var).where(n > 1)


def first_seen(group: pd.Series, value: pd.Series) -> pd.Series:
    """(grup, değer) çifti ilk kez mi görülüyor? Değer boşsa False."""
    key = pd.DataFrame({"g": group, "v": value})
    return ~key.duplicated() & value.notna()


def past_distinct_count(group: pd.Series, value: pd.Series) -> pd.Series:
    """Grubun bu satırdan önce gördüğü farklı değer sayısı."""
    new = first_seen(group, value).astype("int32")
    return new.groupby(group, sort=False).cumsum() - new


def past_window(time: pd.Series, group: pd.Series, window: float, values: pd.Series | None = None):
    """Aynı grupta, [t - window, t) aralığındaki önceki işlem sayısı (ve istenirse değer toplamı).

    Grup + zaman sıralı tek bir anahtar üzerinde searchsorted ile O(n log n); groupby+rolling'den çok daha hızlı.
    Aynı saniyedeki işlemlerden sadece satır sırasında öncekiler sayılır.
    """
    codes, _ = pd.factorize(group, sort=False)
    t = time.to_numpy(dtype="int64")
    order = np.lexsort((np.arange(len(t)), t, codes))   # grup, zaman, orijinal sıra

    span = int(t.max() - t.min() + window + 1)
    key = codes[order].astype("int64") * span + (t[order] - t.min())
    pos = np.arange(len(key))
    left = np.searchsorted(key, key - int(window), side="left")

    count = np.empty(len(t), dtype="int32")
    count[order] = pos - left
    if values is None:
        return pd.Series(count, index=time.index)

    v = values.to_numpy(dtype="float64")[order]
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
    total = np.empty(len(t))
    total[order] = csum[pos] - csum[left]
    return pd.Series(count, index=time.index), pd.Series(total, index=time.index)


def circular_hour_distance(a: pd.Series, b: pd.Series) -> pd.Series:
    """İki saat arasındaki en kısa fark (0-12). 23 ile 1 arası 2 saattir, 22 değil."""
    d = (a - b).abs() % 24
    return np.minimum(d, 24 - d)
