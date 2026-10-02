"""Context adjust engine (Adım 6): ham anomali skorunu iş ve zaman bağlamına göre düzeltir.

Her kural koşulu sağlayan işlemin skorunu bir çarpanla değiştirir. Çarpanlar çarpılır, toplam çarpan
[min_total_factor, max_total_factor] ile sınırlanır (tek bir bağlam skoru silemesin / şişiremesin), sonuç 1 ile kırpılır.
Kurallar config/context_rules.yaml'dan okunur; reload() ile kod değişmeden yeniden yüklenir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from fraud_platform.conditions.evaluator import Condition, describe, evaluate


class ContextRule(BaseModel):
    id: str
    name: str
    category: str
    when: Condition
    factor: float = Field(gt=0)
    reason: str
    enabled: bool = True
    evidence: dict[str, float] = {}   # kalibrasyon çıktısı, dokümantasyon için


class ContextRuleSet(BaseModel):
    min_total_factor: float = Field(gt=0)
    max_total_factor: float = Field(gt=0)
    rules: list[ContextRule]

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, rules: list[ContextRule]) -> list[ContextRule]:
        ids = [r.id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError(f"tekrarlanan kural id: {sorted({i for i in ids if ids.count(i) > 1})}")
        return rules

    @model_validator(mode="after")
    def _bounds(self) -> ContextRuleSet:
        if self.min_total_factor > 1 or self.max_total_factor < 1:
            raise ValueError("toplam çarpan sınırları 1'i kapsamalı (min <= 1 <= max)")
        return self


class ContextAdjuster:
    def __init__(self, rules_path: Path):
        self.rules_path = Path(rules_path)
        self.reload()

    def reload(self) -> int:
        raw = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        self.ruleset = ContextRuleSet.model_validate(raw)
        return len(self.active_rules)

    @property
    def active_rules(self) -> list[ContextRule]:
        return [r for r in self.ruleset.rules if r.enabled]

    def hits(self, df: pd.DataFrame, rules: list[ContextRule] | None = None) -> pd.DataFrame:
        """Kural başına koşul sağlandı mı (satır x kural)."""
        rules = self.active_rules if rules is None else rules
        return pd.DataFrame({r.id: evaluate(r.when, df) for r in rules}, index=df.index)

    def factors(self, hits: pd.DataFrame, rules: list[ContextRule] | None = None) -> pd.Series:
        rules = self.active_rules if rules is None else rules
        f = pd.Series({r.id: r.factor for r in rules})
        log_total = (hits[f.index].astype(float) * np.log(f)).sum(axis=1)
        rs = self.ruleset
        return np.exp(log_total).clip(rs.min_total_factor, rs.max_total_factor).rename("context_factor")

    def adjust(self, df: pd.DataFrame, score: pd.Series, rules: list[ContextRule] | None = None) -> pd.DataFrame:
        """adjusted_score = min(ham skor x sınırlanmış toplam çarpan, 1)."""
        h = self.hits(df, rules)
        factor = self.factors(h, rules)
        return pd.DataFrame({"context_factor": factor, "adjusted_score": (score * factor).clip(upper=1.0)}, index=df.index)

    def explain(self, row: pd.Series | dict[str, Any]) -> list[dict[str, Any]]:
        """Tek işlem için uygulanan kurallar."""
        one = pd.DataFrame([row]) if isinstance(row, dict) else row.to_frame().T
        h = self.hits(one).iloc[0]
        return [
            {"id": r.id, "name": r.name, "category": r.category, "factor": r.factor,
             "condition": describe(r.when), "reason": r.reason}
            for r in self.active_rules if h[r.id]
        ]
