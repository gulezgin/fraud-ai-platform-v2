"""Bir işlemi uçtan uca skorlar, kuralları değerlendirir ve JSON basar (API öncesi ara kontrol).

    python scripts/score_transaction.py 3554906
    python scripts/score_transaction.py 3554906 --no-explain --strategy most_severe

İşlem ham veriden (merged.parquet) okunur; geçmişi profil deposundan gelir, sadece kendinden önceki işlemler kullanılır.
"""
import argparse
import json
import time

import pandas as pd

from fraud_platform.config import get_settings
from fraud_platform.rules.engine import RuleEngine
from fraud_platform.services.scoring_service import ScoringService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transaction_id", type=int)
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--strategy", default=None, help="çakışma stratejisi (varsayılan: rules.yaml)")
    args = parser.parse_args()

    s = get_settings()
    raw = pd.read_parquet(s.merged_path, filters=[(s.data.id_column, "==", args.transaction_id)])
    if raw.empty:
        raise SystemExit(f"{args.transaction_id} bulunamadı")
    raw = raw.drop(columns=[s.data.target_column])   # etiket skorlamaya girmez

    service = ScoringService.from_artifacts(s)
    rules = RuleEngine(s.rules.rules_file, s.rules.base_risk_field)

    start = time.time()
    result = service.score(raw, explain=not args.no_explain)[0]
    frame = service.assessment_frame(raw)
    result["rules"] = rules.evaluate(frame.iloc[0], strategy=args.strategy)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
