"""Kural seti istatistikleri: kural başına tetiklenme, isabet, ek isabet, kazanma/ezilme ve karar dağılımı.

    python scripts/rule_report.py                       # valid ve test
    python scripts/rule_report.py --strategy most_severe

Önce scripts/train.py çalıştırılmalı (risk_percentile kolonlarına ihtiyaç var). Valid dönemi train'de fit edilmiş
modellerin skorlarıyla, test dönemi final modellerle değerlendirilir.
"""
import argparse

import pandas as pd

from fraud_platform.config import get_settings
from fraud_platform.rules.engine import RuleEngine


def load_period(settings, split: str) -> pd.DataFrame:
    d = settings.data
    suffix = "_valfit" if split == "valid" else ""
    cols = [d.id_column, "split", f"risk_percentile{suffix}", f"adjusted_score{suffix}"]
    scores = pd.read_parquet(settings.layer_scores_path, columns=cols)
    scores = scores.rename(columns={f"risk_percentile{suffix}": "risk_percentile", f"adjusted_score{suffix}": "adjusted_score"})
    df = pd.read_parquet(settings.features_path).merge(scores, on=d.id_column)
    return df[df["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None)
    args = parser.parse_args()

    s = get_settings()
    engine = RuleEngine(s.rules.rules_file, s.rules.base_risk_field)
    pd.set_option("display.width", 220)
    for split in ("valid", "test"):
        df = load_period(s, split)
        target = s.data.target_column
        print(f"\n===== {split}: {len(df)} işlem, fraud {df[target].mean():.4f}, strateji "
              f"{engine.resolver(args.strategy).name} =====")
        print(engine.stats(df, target, args.strategy, s.rules.review_budget).round(4).to_string())
        print()
        print(engine.decision_summary(df, target, args.strategy).round(4).to_string())


if __name__ == "__main__":
    main()
