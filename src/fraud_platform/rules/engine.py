"""Configurable rule engine (Adım 7): iş kuralları + anomali skoru -> karar ve açıklama.

Anomali skoru kurallara ayrı bir mekanizmayla değil, alan olarak girer (ör. risk_percentile >= 0,97 -> REVIEW, öncelik 70).
Böylece skor da çakışma çözümüne öncelikli bir kural gibi katılır: yüksek öncelikli bir BLOCK kuralı skoru geçersiz kılabilir,
düşük öncelikli bir beyaz liste (ALLOW) kılamaz.

final_risk = min(1, max(0, risk_percentile + tetiklenen kuralların risk_delta toplamı))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform.conditions.evaluator import describe, evaluate, fields
from fraud_platform.rules.loader import RuleLoader
from fraud_platform.rules.models import Rule
from fraud_platform.rules.resolver import RESOLVERS, ConflictResolver

_SPEC = re.compile(r"\{(\w+)(?::[^}]*)?\}")


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _clean(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    return round(v, 4) if isinstance(v, float) else v


def render(template: str, values: dict[str, Any]) -> str:
    """Açıklama şablonunu doldurur; boş değer '?' olur, biçim hatasında biçimsiz yazar."""
    shown = _Missing({k: ("?" if v is None else v) for k, v in values.items()})
    try:
        return template.format_map(shown)
    except (ValueError, TypeError):
        return _SPEC.sub(r"{\1}", template).format_map(shown)


class RuleEngine:
    def __init__(self, rules_path: Path, base_risk_field: str = "risk_percentile"):
        self.loader = RuleLoader(rules_path)
        self.base_risk_field = base_risk_field
        self.reload()

    def reload(self) -> int:
        """Kural dosyasını yeniden okur (kod değişmeden yeni / değişen kurallar devreye girer)."""
        self.ruleset = self.loader.load()
        return len(self.rules)

    @property
    def rules(self) -> list[Rule]:
        return [r for r in self.ruleset.rules if r.enabled]

    def resolver(self, strategy: str | None = None) -> ConflictResolver:
        name = strategy or self.ruleset.conflict_strategy
        if name not in RESOLVERS:
            raise ValueError(f"bilinmeyen çakışma stratejisi '{name}', geçerli: {sorted(RESOLVERS)}")
        return RESOLVERS[name]()

    def hits(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({r.id: evaluate(r.condition, df) for r in self.rules}, index=df.index)

    def _final_risk(self, df: pd.DataFrame, hits: pd.DataFrame) -> pd.Series:
        deltas = pd.Series({r.id: r.risk_delta for r in self.rules})
        base = df[self.base_risk_field] if self.base_risk_field in df else pd.Series(0.0, index=df.index)
        return (base + hits[deltas.index].astype(float) @ deltas).clip(0, 1)

    # ------------------------------------------------------------ toplu değerlendirme (istatistik)

    def evaluate_batch(self, df: pd.DataFrame, strategy: str | None = None) -> pd.DataFrame:
        rules, hits = self.rules, self.hits(df)
        winner = self.resolver(strategy).winner_index(hits, rules)
        ids = np.array([r.id for r in rules] + [""])
        actions = np.array([r.action for r in rules] + [self.ruleset.default_action])
        return pd.DataFrame({
            "decision": actions[winner],          # -1 -> son eleman: varsayılan karar
            "winning_rule": ids[winner],
            "n_fired": hits.sum(axis=1),
            "final_risk": self._final_risk(df, hits),
        }, index=df.index).join(hits)

    # ------------------------------------------------------------ tek işlem (açıklama)

    def evaluate(self, row: pd.Series | dict[str, Any], strategy: str | None = None) -> dict[str, Any]:
        one = pd.DataFrame([row]) if isinstance(row, dict) else row.to_frame().T
        rules, resolver = self.rules, self.resolver(strategy)
        h = self.hits(one).iloc[0]
        values = {k: _clean(v) for k, v in one.iloc[0].items()}

        fired = [(i, r) for i, r in enumerate(rules) if h[r.id]]
        win = resolver.resolve(fired)
        winner = win[1] if win else None

        fired_out = []
        for _, r in sorted(fired, key=lambda x: resolver.key(x[1], x[0]), reverse=True):
            item = {
                "id": r.id, "name": r.name, "category": r.category, "priority": r.priority,
                "action": r.action, "risk_delta": r.risk_delta, "policy_ref": r.policy_ref,
                "condition": describe(r.condition),
                "matched_values": {f: values.get(f) for f in dict.fromkeys(fields(r.condition))},
                "explanation": render(r.explanation, values),
            }
            if r is not winner:
                item["overridden_by"] = winner.id
            fired_out.append(item)

        if winner:
            reason = (f"{resolver.name}: kazanan kural {winner.id} ({winner.name}, öncelik {winner.priority}, "
                      f"{winner.action}); tetiklenen {len(fired)} kural")
        else:
            reason = f"Hiçbir kural tetiklenmedi; varsayılan karar {self.ruleset.default_action}"

        return {
            "final_decision": winner.action if winner else self.ruleset.default_action,
            "final_risk": round(float(self._final_risk(one, self.hits(one)).iloc[0]), 4),
            "base_risk": values.get(self.base_risk_field),
            "winning_rule": winner.id if winner else None,
            "decision_reason": reason,
            "conflict_strategy": resolver.name,
            "fired_rules": fired_out,
            "rules_evaluated": len(rules),
        }

    # ------------------------------------------------------------ kural istatistikleri

    def stats(self, df: pd.DataFrame, target: str, strategy: str | None = None,
              budget_threshold: float = 0.97) -> pd.DataFrame:
        """Kural başına tetiklenme istatistikleri. Kapalı kurallar da ölçülür (kanıt için), kazanan hesabına girmez.

        incremental_precision: skorun alarm bütçesi (base_risk_field >= budget_threshold) DIŞINDA kalan tetiklenmelerde
        fraud oranı, yani kuralın skora kattığı ek isabet (REVIEW / FLAG ayrımının ölçütü).
        """
        out = self.evaluate_batch(df, strategy)
        y = df[target]
        in_budget = df[self.base_risk_field] >= budget_threshold if self.base_risk_field in df else False
        rows = []
        for r in self.ruleset.rules:
            fired = out[r.id] if r.enabled else evaluate(r.condition, df)
            won = (out["winning_rule"] == r.id) if r.enabled else pd.Series(False, index=df.index)
            extra = fired & ~in_budget
            rows.append({
                "id": r.id, "name": r.name, "enabled": r.enabled, "action": r.action, "priority": r.priority,
                "fired": int(fired.sum()), "coverage": fired.mean(),
                "precision": y[fired].mean() if fired.any() else np.nan,
                "recall": y[fired].sum() / max(1, y.sum()),
                "incremental_precision": y[extra].mean() if extra.any() else np.nan,
                "won": int(won.sum()), "overridden": int((fired & ~won).sum()) if r.enabled else 0,
            })
        return pd.DataFrame(rows).set_index("id")

    def decision_summary(self, df: pd.DataFrame, target: str, strategy: str | None = None) -> pd.DataFrame:
        out = self.evaluate_batch(df, strategy)
        y = df[target]
        g = out.groupby("decision")
        return pd.DataFrame({
            "n": g.size(), "share": g.size() / len(df),
            "fraud_rate": y.groupby(out["decision"]).mean(),
            "fraud_share": y.groupby(out["decision"]).sum() / max(1, y.sum()),
        }).reindex(["BLOCK", "REVIEW", "FLAG", "ALLOW"])
