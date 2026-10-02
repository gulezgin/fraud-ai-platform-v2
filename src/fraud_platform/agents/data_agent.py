"""DataAgent: gelen işlemi doğrular, veri seti hakkında özet verir (Adım 1-2 çıktıları)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.data.schema import Schema, SemanticType


class DataAgent(BaseAgent):
    name = "data_agent"
    description = "Gelen işlemi şemaya göre doğrular; veri seti kalite özetini verir"

    def __init__(self, schema: Schema, quality_report: dict, critical_fields: list[str], pipeline_fields: list[str]):
        super().__init__()
        self.schema = schema
        self.quality = quality_report
        self.critical = critical_fields     # olmadan skorlanamaz (zaman, tutar, kart)
        self.pipeline_fields = pipeline_fields

    def capabilities(self) -> dict[str, Capability]:
        return {
            "validate_transaction": Capability("İşlemin zorunlu alanlarını ve tiplerini kontrol eder", inputs=("transaction",),
                                               provides=("validation",), needs_transaction=True),
            "dataset_summary": Capability("Eğitim verisinin boyutu, eksiklik ve kalite özetini verir", provides=("dataset",)),
        }

    def on_validate_transaction(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        tx = payload["transaction"]
        errors, warnings = [], []
        for f in self.critical:
            if tx.get(f) is None:
                errors.append(f"zorunlu alan eksik: {f}")
        for f, info in self.schema.columns.items():
            v = tx.get(f)
            if v is None or info.semantic_type not in (SemanticType.NUMERIC, SemanticType.DATETIME):
                continue
            if pd.to_numeric(pd.Series([v]), errors="coerce").isna().iloc[0]:
                errors.append(f"{f} sayısal olmalı, gelen: {v!r}")
        missing = [f for f in self.pipeline_fields if tx.get(f) is None and f not in self.critical]
        if missing:
            warnings.append(f"{len(missing)} feature alanı boş (boş kabul edilecek): {', '.join(missing[:6])}"
                            + (" ..." if len(missing) > 6 else ""))
        unknown = [f for f in tx if f not in self.schema.columns]
        if unknown:
            warnings.append(f"şemada olmayan {len(unknown)} alan yok sayılacak: {', '.join(unknown[:5])}")
        return {"validation": {"valid": not errors, "errors": errors, "warnings": warnings,
                               "n_fields": sum(v is not None for v in tx.values())}}

    def on_dataset_summary(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        q = self.quality
        return {"dataset": {
            "rows": q["shape"]["rows"], "columns": q["shape"]["columns"],
            "missing_groups": q["missing"]["groups"],
            "high_cardinality": [h["column"] for h in q["high_cardinality"]],
            "duplicate_rows": q["duplicates"]["duplicate_rows_excluding_id"],
        }}
