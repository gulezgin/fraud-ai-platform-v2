"""Kural dosyasını (YAML veya JSON) okuyup doğrular."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from fraud_platform.rules.models import RuleSet
from fraud_platform.rules.resolver import RESOLVERS


class RuleLoader:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> RuleSet:
        text = self.path.read_text(encoding="utf-8")
        raw = json.loads(text) if self.path.suffix.lower() == ".json" else yaml.safe_load(text)
        ruleset = RuleSet.model_validate(raw)
        if ruleset.conflict_strategy not in RESOLVERS:
            raise ValueError(f"bilinmeyen çakışma stratejisi '{ruleset.conflict_strategy}', geçerli: {sorted(RESOLVERS)}")
        return ruleset
