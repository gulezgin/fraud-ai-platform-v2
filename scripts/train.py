"""Dört anomali dedektörünü ve skor birleştiriciyi eğitir; context düzeltmesini uygular, skorlar ve değerlendirir.

    python scripts/train.py

İki tur:
  1. doğrulama turu: dedektörler + birleştirici train'de fit edilir, train+valid skorlanır.
     Adım 5-6 kararları (ağırlık, context çarpanları) valid skorlarıyla verilir (*_valfit kolonları).
  2. final tur: dedektörler + birleştirici train+valid'de fit edilir, tüm işlemler skorlanır.
     Test metrikleri ve API modelleri bu turdan.

Çıktılar (artifacts/): detectors/*.joblib, aggregator.joblib, layer_scores.parquet, layer_metrics.json,
profile_store.parquet. Önce scripts/prepare_data.py çalıştırılmalı.
"""
import logging
import time

import pandas as pd

from fraud_platform.config import get_settings
from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.data.quality import save_json
from fraud_platform.detection.aggregator import ScoreAggregator
from fraud_platform.detection.factory import LAYERS, build_detectors_from_artifacts
from fraud_platform.evaluation import metrics_table, time_split
from fraud_platform.features.pipeline import FeaturePipeline
from fraud_platform.store.profile_store import EntityProfileStore

logger = logging.getLogger("train")
SCORE_COLS = [*LAYERS, "raw_anomaly_score", "adjusted_score"]


def fit_and_score(detectors, fit_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for d in detectors:
        start = time.time()
        d.fit(fit_df)
        scores[d.name] = d.score(score_df)
        logger.info("%-13s fit+score %.1f sn", d.name, time.time() - start)
    return pd.DataFrame(scores, index=score_df.index)


def run_round(settings, feature_names, df: pd.DataFrame, fit_mask: pd.Series, score_mask: pd.Series):
    detectors = build_detectors_from_artifacts(settings, feature_names)
    layer = fit_and_score(detectors, df[fit_mask], df[score_mask])
    aggregator = ScoreAggregator(settings.scoring).fit(layer[fit_mask[score_mask]])
    scored = layer.join(aggregator.transform(layer))

    context = ContextAdjuster(settings.context.rules_file).adjust(df[score_mask], scored["raw_anomaly_score"])
    context["risk_percentile"] = aggregator.final_pct_(context["adjusted_score"])
    return detectors, aggregator, scored.join(context)


def main() -> None:
    s = get_settings()
    d, ev = s.data, s.evaluation
    pipeline = FeaturePipeline.load(s.feature_pipeline_path)
    df = pd.read_parquet(s.features_path)
    df["split"] = time_split(df[d.time_column], ev)
    logger.info("bölme: %s", df["split"].value_counts().to_dict())

    base = df[[d.id_column, d.target_column, d.time_column, "split"]].copy()
    is_train, not_test = df["split"] == "train", df["split"] != "test"

    # 1) doğrulama turu: train'de fit; train skorları normalizasyonun referansı, valid skorları karar verisi
    _, _, valid_round = run_round(s, pipeline.feature_names, df, is_train, not_test)
    base = base.join(valid_round.add_suffix("_valfit"))

    # 2) final tur
    detectors, aggregator, final_round = run_round(s, pipeline.feature_names, df, not_test, pd.Series(True, index=df.index))
    base = base.join(final_round)

    for det in detectors:
        det.save(s.detectors_path)
    aggregator.save(s.aggregator_path)
    base.to_parquet(s.layer_scores_path, index=False)

    metrics = {
        "valid (train'de fit)": metrics_table(base[base["split"] == "valid"], [f"{x}_valfit" for x in SCORE_COLS],
                                              d.target_column, ev.alert_rate),
        "test (train+valid'de fit)": metrics_table(base[base["split"] == "test"], SCORE_COLS,
                                                   d.target_column, ev.alert_rate),
    }
    save_json({k: v.to_dict(orient="index") for k, v in metrics.items()}, s.layer_metrics_path)
    for name, table in metrics.items():
        logger.info("%s\n%s", name, table[["roc_auc", "pr_auc", "at_3%_precision", "at_3%_recall"]].round(4))

    store = EntityProfileStore.from_features(
        df, pipeline.required_columns(), d.time_column, d.id_column, d.amount_column
    )
    store.save(s.profile_store_path)
    logger.info("profil deposu: %d işlem, %s", len(store), s.profile_store_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
