"""RuleAgent: iş kurallarını değerlendirir, kararı ve çakışma çözümünü üretir (Adım 7)."""
from __future__ import annotations

from typing import Any

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.rules.engine import RuleEngine


class RuleAgent(BaseAgent):
    name = "rule_agent"
    description = "YAML iş kurallarını skorla birlikte değerlendirip BLOCK/REVIEW/FLAG/ALLOW kararı verir"

    def __init__(self, engine: RuleEngine):
        super().__init__()
        self.engine = engine

    def capabilities(self) -> dict[str, Capability]:
        return {
            "evaluate_rules": Capability("Kural kararı, tetiklenen kurallar, kazanan kural ve çakışma çözümü",
                                         requires=("score",), inputs=("assessment", "strategy"), provides=("rules",)),
        }

    def on_evaluate_rules(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        return {"rules": self.engine.evaluate(payload["assessment"], strategy=payload.get("strategy"))}
