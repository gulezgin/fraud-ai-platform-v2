"""Zaman bazlı bölme ve skor değerlendirme metrikleri (Adım 4-6 ortak).

Etiket (isFraud) sadece burada, ölçüm için kullanılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_platform.config import EvaluationConfig

SPLITS = ("train", "valid", "test")


def time_split(time: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    """Her satıra train / valid / test etiketi verir (zamana göre, karıştırmadan)."""
    q_valid, q_test = time.quantile([cfg.valid_start_quantile, cfg.test_start_quantile])
    split = np.where(time <= q_valid, "train", np.where(time <= q_test, "valid", "test"))
    return pd.Series(pd.Categorical(split, categories=SPLITS), index=time.index, name="split")


def alert_metrics(y: pd.Series, score: pd.Series, alert_rate: float) -> dict:
    """En riskli alert_rate kadar işlem alarm alırsa: TP, FP, precision, recall."""
    n_alert = max(1, round(len(score) * alert_rate))
    top = score.rank(method="first", ascending=False) <= n_alert
    tp = int(y[top].sum())
    return {
        "alerts": n_alert,
        "tp": tp,
        "fp": n_alert - tp,
        "precision": tp / n_alert,
        "recall": tp / max(1, int(y.sum())),
    }


def score_metrics(y: pd.Series, score: pd.Series, alert_rate: float) -> dict:
    return {
        "n": len(y),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        **{f"at_{alert_rate:.0%}_{k}": v for k, v in alert_metrics(y, score, alert_rate).items()},
    }


def metrics_table(df: pd.DataFrame, score_cols: list[str], target: str, alert_rate: float) -> pd.DataFrame:
    return pd.DataFrame({c: score_metrics(df[target], df[c], alert_rate) for c in score_cols}).T
