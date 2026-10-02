"""Çakışma çözümü (Strategy pattern): aynı işlemde birden çok kural tetiklenirse hangisi karar verir?

Her strateji kurallar için bir sıralama anahtarı tanımlar; tetiklenenler içinde anahtarı en büyük olan kazanır.
Aynı anahtar hem tek işlemde (resolve) hem toplu değerlendirmede (winner_index, vektörel) kullanılır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fraud_platform.rules.models import Rule


class ConflictResolver(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def key(self, rule: Rule, order: int) -> tuple: ...

    def resolve(self, fired: list[tuple[int, Rule]]) -> tuple[int, Rule] | None:
        """fired: (YAML'daki sıra, kural) listesi."""
        return max(fired, key=lambda x: self.key(x[1], x[0]), default=None)

    def winner_index(self, hits: pd.DataFrame, rules: list[Rule]) -> np.ndarray:
        """Her satır için kazanan kuralın listedeki indeksi; hiç kural tetiklenmediyse -1."""
        keys = [self.key(r, i) for i, r in enumerate(rules)]
        rank = np.empty(len(rules))
        rank[sorted(range(len(rules)), key=lambda i: keys[i])] = np.arange(len(rules))
        masked = np.where(hits[[r.id for r in rules]].to_numpy(), rank, -1)
        winner = masked.argmax(axis=1)
        return np.where(masked.max(axis=1) >= 0, winner, -1)


class PriorityThenSeverity(ConflictResolver):
    name = "priority_then_severity"
    description = "En yüksek öncelikli kural kazanır; eşitlikte daha ağır aksiyon (BLOCK > REVIEW > FLAG > ALLOW)"

    def key(self, rule: Rule, order: int) -> tuple:
        return (rule.priority, rule.severity, -order)


class MostSevere(ConflictResolver):
    name = "most_severe"
    description = "En ağır aksiyon kazanır (güvenli taraf); eşitlikte daha yüksek öncelik"

    def key(self, rule: Rule, order: int) -> tuple:
        return (rule.severity, rule.priority, -order)


class FirstMatch(ConflictResolver):
    name = "first_match"
    description = "Dosyadaki sırada ilk tetiklenen kural kazanır (klasik if-elif zinciri)"

    def key(self, rule: Rule, order: int) -> tuple:
        return (-order,)


RESOLVERS: dict[str, type[ConflictResolver]] = {r.name: r for r in (PriorityThenSeverity, MostSevere, FirstMatch)}
