"""Metin -> vektör. İki strateji: Ollama embedding modeli (bge-m3, çok dilli) ve indirme gerektirmeyen TF-IDF taban çizgisi.

İkisi de L2 normalize vektör döndürür; iç çarpım = kosinüs benzerliği (FAISS IndexFlatIP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer

from fraud_platform.llm.base import LLMError


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norm == 0, 1, norm)


class BaseEmbedder(ABC):
    name: str = ""

    def fit(self, texts: list[str]) -> BaseEmbedder:
        return self

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder(BaseEmbedder):
    name = "ollama"

    def __init__(self, model: str, host: str, timeout: float = 180):
        self.model = model
        self.client = ollama.Client(host=host, timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            return _normalize(self.client.embed(model=self.model, input=texts)["embeddings"])
        except Exception as e:
            raise LLMError(f"Ollama embedding hatası ({self.model}): {e}") from e


class TfidfEmbedder(BaseEmbedder):
    """Karakter n-gram TF-IDF: Türkçe eklerde ("işlemler", "işlemin") kelime bazlıdan dayanıklı. Anlamı değil yazımı eşler."""

    name = "tfidf"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        self.vectorizer.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.vectorizer.transform(texts).toarray())
