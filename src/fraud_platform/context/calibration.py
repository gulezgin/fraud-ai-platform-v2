"""Context çarpanlarının veriden türetilmesi (etiket sadece burada, kalibrasyon için kullanılır).

Ham lift yanıltıcıdır: ProductCD=C işlemleri 3,3 kat riskli ama anomali skoru bunu zaten görüyor (koşullu lift 1,04).
Çarpan, skorun GÖREMEDİĞİ riski düzeltmeli. Bu yüzden koşullu lift kullanılır:
aynı skor seviyesindeki işlemler arasında, kuralın kapsadığı işlemlerde gözlenen fraud / beklenen fraud.
Alarm kararları en riskli kısımda verildiği için ölçüm alarm bölgesinde yapılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.conditions.evaluator import describe
from fraud_platform.config import ContextCalibrationConfig
from fraud_platform.context.adjuster import ContextRule


def conditional_lift(mask: pd.Series, score: pd.Series, y: pd.Series, region: float, bins: int) -> float:
    """Alarm bölgesinde (skorun en üst `region` kısmı), skor dilimlerine göre beklenen fraud'a karşı gözlenen fraud."""
    top = score >= score.quantile(1 - region)
    s, m, t = score[top], mask[top], y[top]
    bin_id = pd.qcut(s.rank(method="first"), bins, labels=False)
    expected = bin_id.map(t.groupby(bin_id).mean())
    exp_sum = expected[m].sum()
    return float(t[m].sum() / exp_sum) if exp_sum > 0 else float("nan")


def suggest_factor(lift: float, cfg: ContextCalibrationConfig) -> float:
    if not np.isfinite(lift) or lift <= 0:
        return 1.0
    lo, hi = cfg.factor_bounds
    return float(np.clip(lift ** cfg.damping, lo, hi))


def calibrate(rules: list[ContextRule], hits: pd.DataFrame, score: pd.Series, y: pd.Series,
              cfg: ContextCalibrationConfig) -> pd.DataFrame:
    base = y.mean()
    top = score >= score.quantile(1 - cfg.alarm_region)
    rows = []
    for r in rules:
        m = hits[r.id]
        lift = conditional_lift(m, score, y, cfg.alarm_region, cfg.bins)
        rows.append({
            "id": r.id, "name": r.name, "enabled": r.enabled, "condition": describe(r.when),
            "coverage": m.mean(), "coverage_alarm_region": m[top].mean(),
            "raw_lift": y[m].mean() / base if m.any() else np.nan,
            "conditional_lift": lift,
            "suggested_factor": suggest_factor(lift, cfg),
            "yaml_factor": r.factor,
        })
    return pd.DataFrame(rows).set_index("id")
