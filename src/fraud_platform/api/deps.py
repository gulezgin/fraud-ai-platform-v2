"""Router'ların ortak yardımcıları."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fraud_platform.api.schemas import TransactionIn
from fraud_platform.services.transaction_repository import TransactionRepository


def resolve_transaction(transaction: TransactionIn | None, transaction_id: int | None,
                        repo: TransactionRepository) -> dict[str, Any]:
    """İstekteki işlem JSON'u ya da kayıtlı işlemin tüm alanları."""
    return transaction.to_record() if transaction is not None else repo.get(transaction_id)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    out: dict[str, float] = {}
    start = time.time()
    yield out
    out["latency_ms"] = round((time.time() - start) * 1000, 1)
