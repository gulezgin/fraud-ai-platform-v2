"""Dependency injection container (dependency_injector).

Bütün nesnelerin nasıl oluşturulacağı tek yerde. Ağır nesneler (modeller, profil deposu, FAISS index, LLM istemcisi) Singleton:
API açılışında bir kez yüklenir, her istekte yeniden yüklenmez. Ayarlar Pydantic Settings'ten okunur (doğrulama korunur);
FRAUD_SETTINGS ortam değişkeniyle başka bir settings.yaml verilebilir.

Testlerde bileşenler override edilir: ör. container.llm.override(providers.Object(SahteLLM())); kod değişmez.
"""
from __future__ import annotations

import os
from typing import Any

from dependency_injector import containers, providers

from fraud_platform.agents.factory import build_agent_system
from fraud_platform.config import load_settings
from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.detection.aggregator import ScoreAggregator
from fraud_platform.detection.factory import load_detectors
from fraud_platform.features.online import OnlineFeatureBuilder
from fraud_platform.features.pipeline import FeaturePipeline
from fraud_platform.llm.ollama_client import OllamaLLM
from fraud_platform.rag.embedder import OllamaEmbedder, TfidfEmbedder
from fraud_platform.rag.pipeline import RAGPipeline
from fraud_platform.rag.vector_store import FaissVectorStore
from fraud_platform.rules.engine import RuleEngine
from fraud_platform.services.explain_service import ExplainService
from fraud_platform.services.scoring_service import ScoringService
from fraud_platform.services.transaction_repository import TransactionRepository
from fraud_platform.store.profile_store import EntityProfileStore


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["fraud_platform.api.routes"])

    settings = providers.Singleton(load_settings, providers.Callable(os.environ.get, "FRAUD_SETTINGS"))
    s = settings.provided

    # ------------------------------------------------ veri ve feature (Adım 1-3)
    transactions = providers.Singleton(TransactionRepository, s.merged_path, s.data.id_column, s.data.target_column)
    feature_pipeline = providers.Singleton(FeaturePipeline.load, s.feature_pipeline_path)
    profile_store = providers.Singleton(EntityProfileStore.load, s.profile_store_path)
    online_features = providers.Singleton(OnlineFeatureBuilder, feature_pipeline, profile_store)

    # ------------------------------------------------ skor (Adım 4-6)
    detectors = providers.Singleton(load_detectors, s.detectors_path)
    aggregator = providers.Singleton(ScoreAggregator.load, s.aggregator_path)
    context = providers.Singleton(ContextAdjuster, s.context.rules_file)
    scoring_service = providers.Singleton(ScoringService, online_features, detectors, aggregator, context, s.data.id_column)

    # ------------------------------------------------ kurallar (Adım 7)
    rule_engine = providers.Singleton(RuleEngine, s.rules.rules_file, s.rules.base_risk_field)

    # ------------------------------------------------ LLM ve RAG (Adım 8)
    llm = providers.Singleton(OllamaLLM, model=s.llm.model, host=s.llm.host, temperature=s.llm.temperature,
                              num_ctx=s.llm.num_ctx, timeout=s.llm.timeout, max_tokens=s.llm.max_tokens)
    embedder = providers.Selector(
        s.rag.embedder,
        ollama=providers.Singleton(OllamaEmbedder, s.rag.embedding_model, s.llm.host, s.llm.timeout),
        tfidf=providers.Singleton(TfidfEmbedder),
    )
    vector_store = providers.Singleton(FaissVectorStore.load, s.rag.index_dir, embedder)
    rag = providers.Singleton(RAGPipeline, vector_store, llm, s.rag.top_k, s.rag.max_context_chunks)

    # ------------------------------------------------ agent'lar ve açıklama (Adım 9-10)
    orchestrator = providers.Singleton(build_agent_system, settings, scoring_service, rule_engine, rag, llm)
    explain_service = providers.Singleton(ExplainService, orchestrator)


COMPONENTS = ("settings", "transactions", "scoring_service", "rule_engine", "llm", "rag", "orchestrator", "explain_service")


def warm_up(container: Container) -> dict[str, Any]:
    """Bileşenleri açılışta yükler; biri yüklenemezse (ör. index yok) API yine açılır, durum /health'te görünür."""
    status = {}
    for name in COMPONENTS:
        try:
            getattr(container, name)()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001 - açılış: eksik bileşen API'yi düşürmemeli, hata raporlanır
            status[name] = f"{type(e).__name__}: {e}"
    return status
