"""API istek ve cevap modelleri. Girdi Pydantic ile doğrulanır; hatalı istek 422 döner."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["ALLOW", "FLAG", "REVIEW", "BLOCK"]

EXAMPLE_TRANSACTION = {
    "TransactionDT": 15055092, "TransactionAmt": 119.21, "ProductCD": "C", "card1": 5812, "card2": 408.0,
    "card3": 185.0, "card4": "mastercard", "card5": 224.0, "card6": "debit", "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com", "D1": 0.0, "M4": "M2", "DeviceType": "mobile",
    "DeviceInfo": "SAMSUNG SM-G892A Build/NRD90M", "id_13": 27.0, "id_19": 153.0, "id_20": 340.0,
    "id_31": "chrome 66.0 for android", "C1": 3.0, "C13": 2.0,
}


class TransactionIn(BaseModel):
    """IEEE-CIS işlem alanları. Belirtilmeyen alanlar (V1-V339, C, D, M, id_...) da kabul edilir; eksik alan boş sayılır."""

    model_config = ConfigDict(extra="allow", json_schema_extra={"example": EXAMPLE_TRANSACTION})

    TransactionID: int | None = Field(None, description="Yoksa sistem yeni bir kimlik verir")
    TransactionDT: int = Field(ge=0, description="Referans andan itibaren saniye")
    TransactionAmt: float = Field(gt=0, description="USD")
    ProductCD: str
    card1: int
    card2: float | None = None
    card4: str | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    D1: float | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None

    def to_record(self) -> dict[str, Any]:
        return self.model_dump()


class TransactionRequest(BaseModel):
    """İşlem ya JSON olarak ya da kayıtlı bir TransactionID ile verilir (ID ile tüm 435 alan veri setinden okunur)."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"transaction_id": 3554906},
                                                              {"transaction": EXAMPLE_TRANSACTION}]})

    transaction: TransactionIn | None = None
    transaction_id: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> TransactionRequest:
        if (self.transaction is None) == (self.transaction_id is None):
            raise ValueError("transaction veya transaction_id alanlarından tam olarak biri verilmeli")
        return self


class ScoreRequest(TransactionRequest):
    explain: bool = Field(True, description="Katman gerekçeleri (en etkili feature'lar) dönsün mü")


class Reason(BaseModel):
    detector: str
    feature: str
    value: Any = None
    contribution: float
    text: str


class ContextAdjustment(BaseModel):
    id: str
    name: str
    category: str
    factor: float
    condition: str
    reason: str


class ScoreResponse(BaseModel):
    TransactionID: int
    uid: str
    layer_scores: dict[str, float]
    layer_percentiles: dict[str, float]
    raw_anomaly_score: float
    raw_percentile: float
    context_factor: float
    adjusted_score: float
    risk_percentile: float = Field(description="Düzeltilmiş skorun eğitimdeki yeri; 0,97 = en riskli %3")
    context_adjustments: list[ContextAdjustment]
    reasons: dict[str, list[Reason]] | None = None
    latency_ms: float


class RulesEvaluateRequest(BaseModel):
    """Üç yoldan biri: transaction (skorlanır), transaction_id (skorlanır) veya fields (hazır alanlarla, skorlamadan)."""

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"transaction_id": 3554906},
        {"fields": {"risk_percentile": 0.995, "user_tx_count": 4, "tx_count_1h": 12, "TransactionAmt": 820.0,
                    "is_night": 1, "is_foreign_address": 0, "local_hour": 3}, "strategy": "most_severe"},
    ]})

    transaction: TransactionIn | None = None
    transaction_id: int | None = None
    fields: dict[str, Any] | None = Field(None, description="Kural alanları (ör. risk_percentile, tx_count_1h), what-if analizi için")
    strategy: Literal["priority_then_severity", "most_severe", "first_match"] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> RulesEvaluateRequest:
        given = sum(x is not None for x in (self.transaction, self.transaction_id, self.fields))
        if given != 1:
            raise ValueError("transaction, transaction_id veya fields alanlarından tam olarak biri verilmeli")
        return self


class FiredRule(BaseModel):
    id: str
    name: str
    category: str
    priority: int
    action: Decision
    risk_delta: float
    policy_ref: str | None
    condition: str
    matched_values: dict[str, Any]
    explanation: str
    overridden_by: str | None = None


class RulesEvaluateResponse(BaseModel):
    final_decision: Decision
    final_risk: float
    base_risk: float | None
    winning_rule: str | None
    decision_reason: str
    conflict_strategy: str
    fired_rules: list[FiredRule]
    rules_evaluated: int


class RulesReloadResponse(BaseModel):
    rules: int
    context_rules: int


class ExplainRequest(TransactionRequest):
    strategy: Literal["priority_then_severity", "most_severe", "first_match"] | None = None


class ExplainResponse(BaseModel):
    status: str
    transaction_id: int | None = None
    decision: Decision | None = None
    final_risk: float | None = None
    scores: dict[str, Any] | None = None
    context_adjustments: list[ContextAdjustment] | None = None
    top_features: list[dict[str, Any]] | None = None
    rules: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = Field(None, description="RAG açıklaması: metin, atıflar, doğrulama, kaynaklar")
    plan: dict[str, Any] | None = None
    agent_trace: list[dict[str, Any]] | None = None
    error: str | None = None
    latency_ms: float | None = None


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"question": "Kart test saldırısında cihaz ne kadar süre gözetimde kalır?"}})

    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = Field(None, ge=1, le=10)


class RAGSource(BaseModel):
    policy_code: str | None
    title: str
    source: str
    score: float
    via: str
    text: str


class RAGQueryResponse(BaseModel):
    question: str
    answer: str | None
    sources: list[RAGSource]
    citations: list[str]
    unsupported_citations: list[str]
    citations_verified: bool
    llm_error: str | None = None
    latency_ms: float


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"goal": "Bu işlemi değerlendir ve kararın nedenini açıkla", "transaction_id": 3554906},
        {"goal": "Yeni kart politikasında günlük limit ne kadar?"},
    ]})

    goal: str = Field(min_length=3, max_length=500, description="Doğal dilde istek; orchestrator planı buna göre çıkarır")
    transaction: TransactionIn | None = None
    transaction_id: int | None = None
    strategy: Literal["priority_then_severity", "most_severe", "first_match"] | None = None
    force_investigation: bool = False

    @model_validator(mode="after")
    def _at_most_one(self) -> AgentRunRequest:
        if self.transaction is not None and self.transaction_id is not None:
            raise ValueError("transaction ve transaction_id birlikte verilemez")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
    llm_reachable: bool
    rules: int | None = None
    knowledge_base_chunks: int | None = None
