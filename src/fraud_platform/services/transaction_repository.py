"""İşlem deposu (Repository): kayıtlı bir işlemi TransactionID ile getirir.

Prototipte kaynak merged.parquet; gerçek sistemde bir veritabanı olur, API ve agent'lar bu arayüzü kullanmaya devam eder.
Etiket (isFraud) skorlamaya girmesin diye döndürülmez.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


class TransactionNotFound(KeyError):
    pass


class TransactionRepository:
    def __init__(self, path: Path, id_column: str, target_column: str):
        self.dataset = ds.dataset(path)
        self.id_column = id_column
        self.target_column = target_column

    def get(self, transaction_id: int) -> dict[str, Any]:
        table = self.dataset.to_table(filter=ds.field(self.id_column) == transaction_id)
        if table.num_rows == 0:
            raise TransactionNotFound(f"işlem bulunamadı: {transaction_id}")
        row = table.to_pandas().drop(columns=[self.target_column], errors="ignore").iloc[0]
        return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
