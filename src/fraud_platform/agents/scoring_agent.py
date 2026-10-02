"""ScoringAgent: 4 anomali katmanı + skor birleştirme + context düzeltmesi (Adım 4-6)."""
from __future__ import annotations

from typing import Any

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.services.scoring_service import ScoringService


class ScoringAgent(BaseAgent):
    name = "scoring_agent"
    description = "Dört anomali katmanını çalıştırır, skoru birleştirir ve bağlama göre düzeltir"

    def __init__(self, service: ScoringService):
        super().__init__()
        self.service = service

    def capabilities(self) -> dict[str, Capability]:
        return {
            "score": Capability("Anomali skoru, katman yüzdelikleri, context düzeltmesi ve dedektör gerekçeleri",
                                requires=("build_features",), inputs=("features",), provides=("score", "assessment")),
        }

    def on_score(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        results, frame = self.service.score_features(payload["features"], explain=payload.get("explain", True))
        return {"score": results[0], "assessment": frame.iloc[0]}
