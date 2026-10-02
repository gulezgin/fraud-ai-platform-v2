"""InvestigatorAgent: kararı politikalara dayanarak açıklar (RAG, Adım 8); politika sorularını cevaplar.

Açıklamadan önce FeatureAgent'tan kullanıcı profilini ister (agent-to-agent): profil LLM'e ek bağlam olarak gider.
"""
from __future__ import annotations

from typing import Any

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.rag.pipeline import RAGPipeline


class InvestigatorAgent(BaseAgent):
    name = "investigator_agent"
    description = "Kararı politika bilgi tabanına dayanarak açıklar (RAG + yerel LLM); politika sorularını cevaplar"

    def __init__(self, rag: RAGPipeline, profile_agent: str = "feature_agent"):
        super().__init__()
        self.rag = rag
        self.profile_agent = profile_agent

    def capabilities(self) -> dict[str, Capability]:
        return {
            "investigate": Capability("Verilen işlemin kararını politikalara dayanarak açıklar (neden bu karar)",
                                      requires=("evaluate_rules",), inputs=("score", "rules"), provides=("investigation",)),
            "answer_policy_question": Capability("İşlemden bağımsız genel politika sorusunu cevaplar "
                                                 "(bir işlemin açıklaması için değil, onun için investigate)",
                                                 inputs=("question",), provides=("answer",)),
        }

    def on_investigate(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        notes = []
        reply = self.ask(self.profile_agent, "get_entity_profile", {"uid": payload["score"]["uid"]}, parent=msg)
        if reply.status == "ok" and reply.payload["profile"].get("known"):
            p = reply.payload["profile"]
            notes.append(f"Kullanıcı profili: geçmişte {p['n_tx']} işlem, ortalama {p['avg_amount']} USD, "
                         f"en yüksek {p['max_amount']} USD, {p['n_devices']} farklı cihaz")
        elif reply.status == "ok":
            notes.append("Kullanıcı profili: geçmiş işlem yok (ilk işlem)")
        return {"investigation": self.rag.explain_transaction(payload["score"], payload["rules"], notes=notes)}

    def on_answer_policy_question(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        return {"answer": self.rag.query(payload["question"])}
