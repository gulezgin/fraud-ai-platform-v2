"""Dağılım, aykırı değer, eksiklik sinyali, nadir kombinasyon, kolon ilişkisi ve entity analizleri.

Fonksiyonlar tek başına da kullanılabilir; DataProfiler bunları settings ile bağlar.
Hedef kolonu (isFraud) burada sadece analiz/ölçüm için kullanılır, hiçbir çıktı feature olarak modele girmez.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from fraud_platform.config import DataConfig, ProfilingConfig
from fraud_platform.data.schema import Schema, SemanticType

# ---------------------------------------------------------------- dağılım


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """describe + çarpıklık/basıklık. log_candidate: sağa çok çarpık ve negatif değer içermiyor."""
    desc = df[columns].describe().T
    desc["skew"] = df[columns].skew()
    desc["kurtosis"] = df[columns].kurt()
    desc["log_candidate"] = (desc["skew"] > 2) & (desc["min"] >= 0)
    return desc


def top_categories(df: pd.DataFrame, columns: list[str], k: int = 5) -> pd.DataFrame:
    rows = []
    for c in columns:
        vc = df[c].value_counts(normalize=True, dropna=False).head(k)
        rows += [{"column": c, "value": v, "share": round(float(p), 4)} for v, p in vc.items()]
    return pd.DataFrame(rows)


def target_rate_by(df: pd.DataFrame, column: str | pd.Series, target: str, min_count: int = 1) -> pd.DataFrame:
    """Bir kolonun her değeri için işlem sayısı, payı, hedef oranı ve genel orana göre lift."""
    overall = df[target].mean()
    g = df.groupby(column, dropna=False, observed=True)[target].agg(count="size", rate="mean")
    g = g[g["count"] >= min_count]
    g["share"] = g["count"] / len(df)
    g["lift"] = g["rate"] / overall
    return g.sort_values("rate", ascending=False)


def to_datetime(df: pd.DataFrame, time_col: str, reference_date) -> pd.Series:
    """TransactionDT saniye offset'ini varsayılan referans tarihe ekleyerek datetime üretir."""
    return pd.Timestamp(reference_date) + pd.to_timedelta(df[time_col], unit="s")


def temporal_profile(df: pd.DataFrame, dt: pd.Series, target: str) -> dict[str, pd.DataFrame]:
    """Saat ve haftanın gününe göre işlem hacmi ve fraud oranı. Saat dilimi bilinmediği için saat görelidir."""
    return {
        "hour": target_rate_by(df, dt.dt.hour.rename("hour"), target).sort_index(),
        "day_of_week": target_rate_by(df, dt.dt.dayofweek.rename("day_of_week"), target).sort_index(),
        "daily_volume": df.groupby(dt.dt.date.rename("date"))[target].agg(count="size", rate="mean"),
    }


# ---------------------------------------------------------------- aykırı değer


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def robust_z(s: pd.Series) -> pd.Series:
    """Medyan/MAD tabanlı z-skoru; çarpık veride ortalama/std'den daha dayanıklı.

    MAD = 0 ise (değerlerin yarısından fazlası medyanda) ortalama mutlak sapmaya düşülür.
    """
    med = s.median()
    abs_dev = (s - med).abs()
    mad = abs_dev.median()
    if mad > 0:
        scale = 1.4826 * mad
    else:
        mean_ad = abs_dev.mean()
        if not mean_ad > 0:
            return pd.Series(0.0, index=s.index)
        scale = 1.2533 * mean_ad
    return (s - med) / scale


def outlier_table(
    df: pd.DataFrame, columns: list[str], target: str, iqr_k: float = 1.5, z_threshold: float = 3.5
) -> pd.DataFrame:
    """Her sayısal kolon için IQR ve robust-z ile aykırı oranı, aykırılarda fraud oranı ve lift."""
    overall = df[target].mean()
    rows = []
    for c in columns:
        s = df[c].dropna()
        if s.nunique() < 3:
            continue
        lo, hi = iqr_bounds(s, iqr_k)
        iqr_mask = (s < lo) | (s > hi)
        z_mask = robust_z(s).abs() > z_threshold
        fraud_in_z = df.loc[z_mask[z_mask].index, target].mean() if z_mask.any() else np.nan
        rows.append({
            "column": c,
            "iqr_ratio": iqr_mask.mean(),
            "robust_z_ratio": z_mask.mean(),
            "fraud_rate_outlier": fraud_in_z,
            "lift_outlier": fraud_in_z / overall,
        })
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------- eksiklik sinyali


def null_target_profile(df: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    """Kolon boşken ve doluyken fraud oranı. lift_null >> 1 ise 'boş olması' başlı başına sinyaldir."""
    m = df[columns].isna().to_numpy()
    y = df[target].to_numpy(dtype=float)
    n_null = m.sum(axis=0)
    n_present = len(df) - n_null
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_null = (m.T @ y) / n_null
        rate_present = ((~m).T @ y) / n_present
    out = pd.DataFrame(
        {"null_ratio": n_null / len(df), "fraud_rate_null": rate_null, "fraud_rate_present": rate_present},
        index=columns,
    )
    out["lift_null"] = out["fraud_rate_null"] / out["fraud_rate_present"]
    return out[out["null_ratio"] > 0]


# ---------------------------------------------------------------- nadir kombinasyon


def combination_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Kolon değer kombinasyonuna tamsayı kimlik verir (NaN de bir değer sayılır).

    Satır satır string birleştirme (agg('|'.join, axis=1)) 590 bin satırda dakikalar sürüyordu.
    """
    return df.groupby(columns, dropna=False, sort=False, observed=True).ngroup()


def rare_combinations(df: pd.DataFrame, columns: list[str], target: str, threshold: float) -> dict:
    """Kolonların birlikte nadir görüldüğü işlemler ve bu işlemlerdeki fraud oranı."""
    key = combination_key(df, columns)
    freq = key.map(key.value_counts(normalize=True))
    rare = freq < threshold
    return {
        "columns": columns,
        "n_combinations": int(key.nunique()),
        "rare_tx_ratio": float(rare.mean()),
        "fraud_rate_rare": float(df.loc[rare, target].mean()) if rare.any() else float("nan"),
        "fraud_rate_common": float(df.loc[~rare, target].mean()),
    }


def pairwise_rare_combinations(df: pd.DataFrame, columns: list[str], target: str, threshold: float) -> pd.DataFrame:
    rows = [rare_combinations(df, list(pair), target, threshold) for pair in combinations(columns, 2)]
    out = pd.DataFrame(rows)
    out["pair"] = out.pop("columns").map(" + ".join)
    out["lift_rare"] = out["fraud_rate_rare"] / df[target].mean()
    return out.set_index("pair").sort_values("lift_rare", ascending=False)


# ---------------------------------------------------------------- kolon ilişkileri


def correlated_groups(df: pd.DataFrame, columns: list[str], threshold: float) -> list[list[str]]:
    """|r| > threshold olan kolonları gruplar (açgözlü). Her grubun ilk kolonu temsilcidir.

    Temsilci önceliği: en az boş, sonra en çok farklı değer.
    """
    order = sorted(columns, key=lambda c: (df[c].isna().mean(), -df[c].nunique()))
    corr = df[order].corr().abs()

    assigned, groups = set(), []
    for c in order:
        if c in assigned:
            continue
        members = [m for m in order if m not in assigned and (m == c or corr.at[c, m] > threshold)]
        assigned.update(members)
        groups.append(members)
    return groups


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """İki kategorik kolon arasındaki ilişki (0-1), Bergsma yanlılık düzeltmesiyle."""
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr, k_corr = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def cramers_v_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    filled = df[columns].astype("string").fillna("NA")
    m = pd.DataFrame(1.0, index=columns, columns=columns)
    for a, b in combinations(columns, 2):
        m.loc[a, b] = m.loc[b, a] = cramers_v(filled[a], filled[b])
    return m


def univariate_auc(df: pd.DataFrame, columns: list[str], target: str) -> pd.Series:
    """Her sayısal kolonun tek başına fraud'u ayırma gücü: max(AUC, 1-AUC). 0.5 = bilgi yok."""
    out = {}
    for c in columns:
        sub = df[[c, target]].dropna()
        if sub[target].nunique() < 2 or sub[c].nunique() < 2:
            continue
        auc = roc_auc_score(sub[target], sub[c])
        out[c] = max(auc, 1 - auc)
    return pd.Series(out, name="auc").sort_values(ascending=False)


# ---------------------------------------------------------------- entity


def compare_entity_keys(df: pd.DataFrame, candidates: dict[str, list[str]], target: str) -> pd.DataFrame:
    """Aday entity tanımlarını karşılaştırır.

    purity_fraud: fraud işlemlerin, fraud oranı > 0.5 olan bir entity'de bulunma oranı.
    Etiket burada sadece tanımın 'gerçek kişiyi' ne kadar yakaladığını ölçmek için kullanılır.
    """
    rows = []
    for name, cols in candidates.items():
        key = combination_key(df, cols)
        g = df.groupby(key)[target]
        size, rate = g.transform("size"), g.transform("mean")
        ent = g.agg(["size", "mean"])
        is_fraud = df[target] == 1
        rows.append({
            "uid": name,
            "n_entity": len(ent),
            "median_tx_per_entity": float(ent["size"].median()),
            "single_tx_entity_ratio": float((ent["size"] == 1).mean()),
            "tx_with_history_ratio": float((size > 1).mean()),
            "purity_fraud": float((rate[is_fraud] > 0.5).mean()),
            "mixed_entity_ratio": float(((ent["mean"] > 0) & (ent["mean"] < 1)).mean()),
        })
    return pd.DataFrame(rows).set_index("uid")


def entity_profile(df: pd.DataFrame, key: pd.Series, target: str, amount_col: str, extra: dict | None = None) -> pd.DataFrame:
    """Entity başına davranış özeti: işlem sayısı, tutar, kaç farklı cihaz/e-posta, fraud oranı."""
    agg = {
        "n_tx": (target, "size"),
        "avg_amt": (amount_col, "mean"),
        "std_amt": (amount_col, "std"),
        "fraud_rate": (target, "mean"),
    }
    agg.update(extra or {})
    return df.groupby(key).agg(**agg)


# ---------------------------------------------------------------- birleştirici


class DataProfiler:
    """Profiling fonksiyonlarını settings'teki eşik ve kolon rolleriyle çalıştırır."""

    def __init__(self, data_cfg: DataConfig, cfg: ProfilingConfig):
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.target = data_cfg.target_column

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        n = min(self.cfg.sample_size, len(df))
        return df.sample(n, random_state=self.data_cfg.random_state)

    def datetime(self, df: pd.DataFrame) -> pd.Series:
        return to_datetime(df, self.data_cfg.time_column, self.data_cfg.reference_date)

    def outliers(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return outlier_table(df, columns, self.target, self.cfg.iqr_k, self.cfg.robust_z_threshold)

    def null_signal(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return null_target_profile(df, columns, self.target)

    def rare_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        return pairwise_rare_combinations(df, self.cfg.rare_combo_columns, self.target, self.cfg.rare_combo_threshold)

    def rare_all(self, df: pd.DataFrame) -> dict:
        return rare_combinations(df, self.cfg.rare_combo_columns, self.target, self.cfg.rare_combo_threshold)

    def correlated(self, df: pd.DataFrame, columns: list[str]) -> list[list[str]]:
        return correlated_groups(self.sample(df), columns, self.cfg.corr_threshold)

    def signal(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        return univariate_auc(self.sample(df), columns, self.target)

    def summary(self, df: pd.DataFrame, schema: Schema) -> dict:
        """Sonraki adımların kullanacağı profil çıktıları (artifacts/profile.json)."""
        numeric = schema.of_type(SemanticType.NUMERIC)
        groups = self.correlated(df, numeric)

        features = [c for c in df.columns if c != self.target]
        nulls = self.null_signal(df, features)
        strong = nulls[(nulls["lift_null"] >= 2) | (nulls["lift_null"] <= 0.5)]

        hourly = temporal_profile(df, self.datetime(df), self.target)["hour"]
        return {
            "correlated_groups": [g for g in groups if len(g) > 1],
            "numeric_representatives": [g[0] for g in groups],
            "null_signal_columns": strong["lift_null"].round(3).to_dict(),
            "log_candidates": numeric_summary(df, numeric).query("log_candidate").index.tolist(),
            "hourly": hourly[["count", "rate"]].round(5).to_dict(orient="index"),
        }
