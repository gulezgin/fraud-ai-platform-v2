"""Açıklama servisi (Facade): skor + kurallar + RAG açıklaması + agent izi tek cevapta.

Açıklama agent sistemi üzerinden üretilir; böylece cevap hangi agent'ın ne yaptığını (iz) da içerir.
/explain açıkça açıklama istediği için ALLOW kararlarında da soruşturma yapılır (force_investigation).
"""
from __future__ import annotations

from typing import Any

from fraud_platform.agents.orchestrator import Orchestrator

GOAL = "Bu işlemi değerlendir ve kararın nedenini açıkla"


class ExplainService:
    def __init__(self, orchestrator: Orchestrator, top_reasons: int = 3):
        self.orchestrator = orchestrator
        self.top_reasons = top_reasons

    def explain(self, transaction: dict[str, Any], strategy: str | None = None,
                force_investigation: bool = True) -> dict[str, Any]:
        r = self.orchestrator.run(GOAL, transaction=transaction, strategy=strategy,
                                  force_investigation=force_investigation)
        if r["status"] != "ok":
            return {"status": r["status"], "error": r.get("error"), "plan": r.get("plan"), "trace": r["trace"]}

        score, rules, inv = r["score"], r["rules"], r.get("investigation")
        top_features = sorted(
            ({"layer": layer, **reason} for layer, rs in score.get("reasons", {}).items() for reason in rs),
            key=lambda x: x["contribution"], reverse=True,
        )[: self.top_reasons * 2]
        return {
            "status": "ok",
            "transaction_id": score.get("TransactionID"),
            "decision": rules["final_decision"],
            "final_risk": rules["final_risk"],
            "scores": {k: score[k] for k in ("layer_percentiles", "raw_anomaly_score", "raw_percentile",
                                              "context_factor", "adjusted_score", "risk_percentile")},
            "context_adjustments": score["context_adjustments"],
            "top_features": top_features,
            "rules": {k: rules[k] for k in ("winning_rule", "decision_reason", "conflict_strategy", "fired_rules")},
            "explanation": inv,
            "plan": r["plan"],
            "agent_trace": r["trace"],
        }
