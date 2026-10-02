"""Ham veriden işlenmiş veri setlerini üretir.

    python scripts/prepare_data.py            # merged.parquet varsa onu kullanır
    python scripts/prepare_data.py --rebuild  # CSV'lerden baştan oluşturur

Çıktılar:
    data/processed/merged.parquet, data/processed/features.parquet
    artifacts/schema.json, quality_report.json, profile.json, feature_pipeline.joblib, feature_dictionary.csv
"""
import argparse
import logging

from fraud_platform.config import get_settings
from fraud_platform.data.loader import DataLoader
from fraud_platform.data.profiler import DataProfiler
from fraud_platform.data.quality import QualityReport, save_json
from fraud_platform.data.schema import SchemaInferer
from fraud_platform.features.pipeline import build_pipeline

logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="merged.parquet'i CSV'lerden yeniden oluştur")
    args = parser.parse_args()

    settings = get_settings()
    loader = DataLoader(settings)

    if args.rebuild or not settings.merged_path.exists():
        df = loader.build()
    else:
        logger.info("mevcut %s kullanılıyor (--rebuild ile yeniden oluşturulur)", settings.merged_path.name)
        df = loader.load()

    schema = SchemaInferer(settings.data, settings.schema_).infer(df)
    schema.save(settings.schema_path)
    logger.info("şema kaydedildi: %s", settings.schema_path)

    report = QualityReport(settings.data, settings.quality, schema).build(df)
    QualityReport.save(report, settings.quality_report_path)
    logger.info("kalite raporu kaydedildi: %s", settings.quality_report_path)

    profile = DataProfiler(settings.data, settings.profiling).summary(df, schema)
    save_json(profile, settings.profile_path)
    logger.info("profil kaydedildi: %s", settings.profile_path)

    pipeline = build_pipeline(settings)
    features = pipeline.fit_transform(df)
    features.to_parquet(settings.features_path, index=False)
    pipeline.save(settings.feature_pipeline_path)
    pipeline.dictionary().to_csv(settings.feature_dictionary_path, index=False, encoding="utf-8")
    logger.info("%d feature üretildi: %s", len(pipeline.feature_names), settings.features_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
