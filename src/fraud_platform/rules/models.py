"""Kural ve kural seti modelleri. YAML/JSON yüklenirken Pydantic ile doğrulanır; hatalı kural sessizce geçmez."""
from __future__ import annotations

from string import Formatter
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from fraud_platform.conditions.evaluator import Condition

Action = Literal["ALLOW", "FLAG", "REVIEW", "BLOCK"]
SEVERITY: dict[str, int] = {"ALLOW": 0, "FLAG": 1, "REVIEW": 2, "BLOCK": 3}


class Rule(BaseModel):
    id: str
    name: str
    category: str = ""
    priority: int = Field(ge=0, le=1000)
    condition: Condition
    action: Action
    risk_delta: float = Field(0.0, ge=-1, le=1)
    explanation: str                      # {alan} veya {alan:.2f} yer tutucuları işlemin değerleriyle doldurulur
    policy_ref: str | None = None          # bilgi tabanındaki politika kodu (Adım 8 RAG bu kodla eşleşir)
    enabled: bool = True
    evidence: dict[str, float | None] = {}   # valid dönemi ölçümü, dokümantasyon için (scripts/rule_report.py)

    @field_validator("explanation")
    @classmethod
    def _valid_template(cls, text: str) -> str:
        try:
            list(Formatter().parse(text))
        except ValueError as e:
            raise ValueError(f"açıklama şablonu hatalı: {e}") from e
        return text

    @property
    def severity(self) -> int:
        return SEVERITY[self.action]


class RuleSet(BaseModel):
    conflict_strategy: str
    default_action: Action = "ALLOW"
    rules: list[Rule] = Field(min_length=1)

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, rules: list[Rule]) -> list[Rule]:
        ids = [r.id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError(f"tekrarlanan kural id: {sorted({i for i in ids if ids.count(i) > 1})}")
        return rules
