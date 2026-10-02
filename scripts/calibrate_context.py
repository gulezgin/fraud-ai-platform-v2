"""Context kurallarının çarpanlarını valid-A döneminde kalibre eder ve kanıt tablosu basar.

    python scripts/calibrate_context.py

YAML'ı otomatik değiştirmez: çarpanlar bir karar, bu script o kararın kanıtı.
Tablodaki suggested_factor ile yaml_factor karşılaştırılıp config/context_rules.yaml elle güncellenir.
Önce scripts/train.py çalıştırılmalı (raw_anomaly_score_valfit kolonuna ihtiyaç var).
"""
import pandas as pd

from fraud_platform.config import get_settings
from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.context.calibration import calibrate


def main() -> None:
    s = get_settings()
    d = s.data
    scores = pd.read_parquet(s.layer_scores_path, columns=[d.id_column, "split", "raw_anomaly_score_valfit"])
    features = pd.read_parquet(s.features_path)
    valid = features.merge(scores, on=d.id_column).query("split == 'valid'").sort_values(d.time_column)
    valid_a = valid.iloc[: len(valid) // 2]   # valid-B ölçüm için saklı

    adjuster = ContextAdjuster(s.context.rules_file)
    rules = adjuster.ruleset.rules
    hits = adjuster.hits(valid_a, rules)
    table = calibrate(rules, hits, valid_a["raw_anomaly_score_valfit"], valid_a[d.target_column], s.context.calibration)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(f"valid-A: {len(valid_a)} işlem, fraud oranı {valid_a[d.target_column].mean():.4f}\n")
    print(table.drop(columns="condition").round(3).to_string())


if __name__ == "__main__":
    main()
