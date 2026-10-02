"""settings.yaml dosyasını okuyup doğrulanmış bir Settings nesnesine çevirir.

Hatalı veya eksik bir alan varsa uygulama daha açılırken ValidationError ile durur.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    @field_validator("*")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        # göreli yollar nereden çalıştırılırsa çalıştırılsın proje köküne göre çözülsün
        return p if p.is_absolute() else PROJECT_ROOT / p


class DataConfig(BaseModel):
    transaction_file: str
    identity_file: str
    merged_file: str
    schema_file: str
    quality_report_file: str
    profile_file: str
    features_file: str
    feature_pipeline_file: str
    feature_dictionary_file: str
    detectors_dir: str
    profile_store_file: str
    layer_scores_file: str
    layer_metrics_file: str
    aggregator_file: str

    id_column: str
    target_column: str
    time_column: str
    amount_column: str
    reference_date: date

    dev_sample_size: PositiveInt | None = None
    random_state: int = 42


class EntityConfig(BaseModel):
    uid_columns: list[str] = Field(min_length=1)
    anchor_days_column: str | None = None
    device_columns: list[str] = Field(min_length=1)


class TimeConfig(BaseModel):
    hour_offset: int = Field(0, ge=-12, le=14)
    night_hours: tuple[int, int] = (0, 6)
    business_hours: tuple[int, int] = (9, 18)
    weekend_days: list[int] = [5, 6]
    velocity_windows: dict[str, PositiveInt]
    rapid_repeat_seconds: PositiveInt = 60

    @field_validator("night_hours", "business_hours")
    @classmethod
    def _valid_hours(cls, v: tuple[int, int]) -> tuple[int, int]:
        if not all(0 <= h <= 24 for h in v):
            raise ValueError("saatler 0-24 arasında olmalı")
        return v


class FeaturesConfig(BaseModel):
    domestic_country_code: float
    email_columns: list[str] = []
    email_aliases: dict[str, str] = {}
    lowercase_columns: list[str] = []
    non_negative_columns: list[str] = []
    null_flag_columns: list[str] = []
    frequency_columns: list[str] = []
    combo_columns: list[str] = []
    new_card_max_days: float = Field(7, ge=0)
    high_value_quantile: float = Field(0.9, gt=0, lt=1)


class EvaluationConfig(BaseModel):
    valid_start_quantile: float = Field(0.6, gt=0, lt=1)
    test_start_quantile: float = Field(0.8, gt=0, lt=1)
    alert_rate: float = Field(0.03, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_order(self) -> EvaluationConfig:
        if self.valid_start_quantile >= self.test_start_quantile:
            raise ValueError("valid_start_quantile, test_start_quantile'dan küçük olmalı")
        return self


class ColumnDetectorConfig(BaseModel):
    z_cap: float = Field(10, gt=0)
    rarity_log_cap: float = Field(5, gt=0)
    batch_size: PositiveInt = 100_000


class MultivariateDetectorConfig(BaseModel):
    n_estimators: PositiveInt = 200
    max_samples: PositiveInt = 1024
    fit_sample_size: PositiveInt = 200_000
    n_quantiles: PositiveInt = 1000
    random_state: int = 42


class WeightedDetectorConfig(BaseModel):
    weights: dict[str, float] = Field(min_length=1)
    caps: dict[str, float] = {}

    @field_validator("weights")
    @classmethod
    def _positive(cls, w: dict[str, float]) -> dict[str, float]:
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError("ağırlıklar negatif olamaz ve toplamı 0'dan büyük olmalı")
        return w


class DetectorsConfig(BaseModel):
    column: ColumnDetectorConfig
    multivariate: MultivariateDetectorConfig
    entity: WeightedDetectorConfig
    temporal: WeightedDetectorConfig


class ScoringConfig(BaseModel):
    normalization: Literal["percentile", "minmax", "tail_log"] = "tail_log"
    weights: dict[str, float] = Field(min_length=1)
    tail_resolution: PositiveInt = 400

    @field_validator("weights")
    @classmethod
    def _positive(cls, w: dict[str, float]) -> dict[str, float]:
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError("ağırlıklar negatif olamaz ve toplamı 0'dan büyük olmalı")
        return w


class ContextCalibrationConfig(BaseModel):
    alarm_region: float = Field(0.10, gt=0, le=1)
    bins: PositiveInt = 5
    damping: float = Field(0.5, ge=0, le=1)
    factor_bounds: tuple[float, float] = (0.5, 1.5)


class ContextConfig(BaseModel):
    rules_file: Path
    calibration: ContextCalibrationConfig

    @field_validator("rules_file")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


class RulesConfig(BaseModel):
    rules_file: Path
    base_risk_field: str = "risk_percentile"
    review_budget: float = Field(0.97, gt=0, lt=1)

    @field_validator("rules_file")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str
    host: str = "http://localhost:11434"
    temperature: float = Field(0.1, ge=0, le=2)
    num_ctx: PositiveInt = 4096
    max_tokens: PositiveInt = 400
    timeout: float = Field(180, gt=0)


class RAGConfig(BaseModel):
    knowledge_base_dir: Path
    index_dir: Path
    embedder: Literal["ollama", "tfidf"] = "ollama"
    embedding_model: str = "bge-m3"
    top_k: PositiveInt = 3
    max_context_chunks: PositiveInt = 5
    max_chunk_words: PositiveInt = 220
    eval_file: Path

    @field_validator("knowledge_base_dir", "index_dir", "eval_file")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


class AgentsConfig(BaseModel):
    planner_mode: Literal["hybrid", "llm", "rules"] = "hybrid"
    investigate_on: list[Literal["ALLOW", "FLAG", "REVIEW", "BLOCK"]] = ["BLOCK", "REVIEW", "FLAG"]


class SchemaConfig(BaseModel):
    categorical_overrides: list[str] = []
    low_cardinality_max: PositiveInt = 10
    high_cardinality_min: PositiveInt = 100

    @model_validator(mode="after")
    def _check_levels(self) -> SchemaConfig:
        if self.low_cardinality_max >= self.high_cardinality_min:
            raise ValueError("low_cardinality_max, high_cardinality_min'den küçük olmalı")
        return self


class ProfilingConfig(BaseModel):
    iqr_k: float = Field(1.5, gt=0)
    robust_z_threshold: float = Field(3.5, gt=0)
    rare_combo_threshold: float = Field(0.0005, gt=0, lt=1)
    rare_combo_columns: list[str] = []
    corr_threshold: float = Field(0.95, gt=0, le=1)
    sample_size: PositiveInt = 100_000


class QualityConfig(BaseModel):
    near_empty_null_ratio: float = Field(0.9, gt=0, le=1)
    domain_columns: list[str] = []


class Settings(BaseModel):
    paths: PathsConfig
    data: DataConfig
    entity: EntityConfig
    time: TimeConfig
    features: FeaturesConfig
    evaluation: EvaluationConfig
    detectors: DetectorsConfig
    scoring: ScoringConfig
    context: ContextConfig
    rules: RulesConfig
    llm: LLMConfig
    rag: RAGConfig
    agents: AgentsConfig
    schema_: SchemaConfig = Field(alias="schema")
    profiling: ProfilingConfig
    quality: QualityConfig

    @property
    def transaction_path(self) -> Path:
        return self.paths.raw_dir / self.data.transaction_file

    @property
    def identity_path(self) -> Path:
        return self.paths.raw_dir / self.data.identity_file

    @property
    def merged_path(self) -> Path:
        return self.paths.processed_dir / self.data.merged_file

    @property
    def schema_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.schema_file

    @property
    def quality_report_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.quality_report_file

    @property
    def profile_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.profile_file

    @property
    def features_path(self) -> Path:
        return self.paths.processed_dir / self.data.features_file

    @property
    def feature_pipeline_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.feature_pipeline_file

    @property
    def feature_dictionary_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.feature_dictionary_file

    @property
    def detectors_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.detectors_dir

    @property
    def profile_store_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.profile_store_file

    @property
    def layer_scores_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.layer_scores_file

    @property
    def layer_metrics_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.layer_metrics_file

    @property
    def aggregator_path(self) -> Path:
        return self.paths.artifacts_dir / self.data.aggregator_file


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path) if path else DEFAULT_SETTINGS_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Varsayılan settings.yaml'i bir kez okuyup önbellekte tutar."""
    return load_settings()
