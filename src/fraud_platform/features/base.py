"""Feature üreticilerin ortak arayüzü.

Her builder sklearn'deki gibi fit/transform ayrımı yapar: frekans tabloları, eşikler gibi veriden öğrenilen
durum fit'te hesaplanır, transform sadece uygular. Böylece aynı nesne eğitimde ve API'de kullanılabilir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    layers: tuple[str, ...] = ()   # hangi dedektör / motor kullanıyor: column, multivariate, entity, temporal, context, rules


class FeatureBuilder(ABC):
    group: str = ""

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def specs(self) -> list[FeatureSpec]: ...

    def required_columns(self) -> list[str]:
        """Builder'ın okuduğu ham kolonlar. Profil deposu geçmiş işlemlerin sadece bu kolonlarını saklar."""
        return []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.specs()]


def add_columns(df: pd.DataFrame, new: dict[str, pd.Series]) -> pd.DataFrame:
    """Yeni kolonları tek concat ile ekler; geniş tabloda tek tek atama DataFrame'i parçalıyor."""
    new_df = pd.DataFrame(new, index=df.index)
    return pd.concat([df.drop(columns=new_df.columns, errors="ignore"), new_df], axis=1)
