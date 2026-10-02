"""RAG bileşenlerini settings'ten oluşturur (container gelene kadar script'ler ve notebook'lar bunu kullanır)."""
from __future__ import annotations

from fraud_platform.config import Settings
from fraud_platform.llm.base import BaseLLM
from fraud_platform.llm.ollama_client import OllamaLLM
from fraud_platform.rag.embedder import BaseEmbedder, OllamaEmbedder, TfidfEmbedder
from fraud_platform.rag.pipeline import RAGPipeline
from fraud_platform.rag.vector_store import FaissVectorStore


def build_embedder(settings: Settings, kind: str | None = None) -> BaseEmbedder:
    kind = kind or settings.rag.embedder
    if kind == "tfidf":
        return TfidfEmbedder()
    return OllamaEmbedder(settings.rag.embedding_model, settings.llm.host, settings.llm.timeout)


def build_llm(settings: Settings) -> BaseLLM:
    c = settings.llm
    return OllamaLLM(c.model, c.host, c.temperature, c.num_ctx, c.timeout, c.max_tokens)


def load_pipeline(settings: Settings, llm: BaseLLM | None = None) -> RAGPipeline:
    store = FaissVectorStore.load(settings.rag.index_dir, build_embedder(settings))
    return RAGPipeline(store, llm or build_llm(settings), settings.rag.top_k, settings.rag.max_context_chunks)
