"""RAG pipeline (Adım 8): retrieval -> context injection -> yerel LLM -> atıf doğrulama.

İki akış:
- query(): serbest politika sorusu.
- explain_transaction(): işlem açıklaması (reasoning akışı). Tetiklenen kuralların policy_ref kodlarıyla politikalar
  doğrudan getirilir, kural / context / dedektör gerekçelerinden kurulan sorguyla anlamsal arama ek politika bulur,
  LLM yapılandırılmış JSON üretir, atıflar getirilen parçalara karşı doğrulanır.

Halüsinasyon kontrolü: cevapta geçen her politika kodu getirilen parçalarda olmalı; değilse citations_verified = False.
LLM'e ulaşılamazsa sistem çökmez: kural açıklamaları + politika başlıklarından deterministik açıklama döner.
"""
from __future__ import annotations

import time
from typing import Any

from fraud_platform.llm.base import BaseLLM, LLMError
from fraud_platform.rag import prompts
from fraud_platform.rag.chunker import POLICY_CODE, Chunk
from fraud_platform.rag.vector_store import FaissVectorStore


def verify_citations(cited: list[str], retrieved_codes: set[str]) -> dict[str, Any]:
    cited = list(dict.fromkeys(cited))
    unsupported = [c for c in cited if c not in retrieved_codes]
    return {"citations": cited, "unsupported_citations": unsupported,
            "citations_verified": bool(cited) and not unsupported}


class RAGPipeline:
    def __init__(self, store: FaissVectorStore, llm: BaseLLM, top_k: int = 3, max_context_chunks: int = 5):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks

    # ------------------------------------------------------------ retrieval

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        return [{"chunk": c, "score": round(s, 4), "via": "vector"} for c, s in self.store.search(query, k or self.top_k)]

    @staticmethod
    def _source(item: dict[str, Any]) -> dict[str, Any]:
        c: Chunk = item["chunk"]
        return {"policy_code": c.policy_code, "title": c.title, "source": c.source,
                "score": item["score"], "via": item["via"], "text": c.text}

    # ------------------------------------------------------------ serbest soru

    def query(self, question: str, k: int | None = None) -> dict[str, Any]:
        t0 = time.time()
        hits = self.retrieve(question, k)
        chunks = [h["chunk"] for h in hits]
        result: dict[str, Any] = {"question": question, "sources": [self._source(h) for h in hits]}
        try:
            answer = self.llm.generate(prompts.QA.format(context=prompts.format_context(chunks), question=question),
                                       system=prompts.SYSTEM)
            result["answer"] = answer.strip()
            result.update(verify_citations(POLICY_CODE.findall(answer), {c.policy_code for c in chunks if c.policy_code}))
        except LLMError as e:
            result.update({"answer": None, "llm_error": str(e), "citations": [], "unsupported_citations": [],
                           "citations_verified": False})
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    # ------------------------------------------------------------ işlem açıklaması

    @staticmethod
    def summarize(score: dict[str, Any], rules: dict[str, Any], top_reasons: int = 2) -> str:
        """İşlem değerlendirmesinin LLM'e giden kısa özeti (4096 token bağlama sığacak kadar)."""
        lines = [
            f"Kural motoru kararı: {rules['final_decision']} ({rules['decision_reason']})",
            f"Risk yüzdeliği: {score['risk_percentile']:.4f} (ham {score['raw_percentile']:.4f}, context çarpanı {score['context_factor']})",
            "Tetiklenen kurallar:",
            *[f"- {r['id']} {r['name']} -> {r['action']} [{r['policy_ref']}]: {r['explanation']}" for r in rules["fired_rules"]],
        ]
        if score.get("context_adjustments"):
            lines.append("Context düzeltmeleri:")
            lines += [f"- {a['name']} x{a['factor']}: {a['reason']}" for a in score["context_adjustments"]]
        if score.get("reasons"):
            lines.append("Anomali katmanlarının öne çıkan gerekçeleri:")
            for layer, rs in score["reasons"].items():
                lines += [f"- [{layer}] {r['text']}" for r in rs[:top_reasons]]
        return "\n".join(lines)

    @staticmethod
    def build_query(score: dict[str, Any], rules: dict[str, Any]) -> str:
        parts = [r["name"] for r in rules["fired_rules"]]
        parts += [a["name"].replace("_", " ") for a in score.get("context_adjustments", [])]
        parts += [rs[0]["text"] for rs in score.get("reasons", {}).values() if rs]
        return "; ".join(parts) or rules["final_decision"]

    def gather_context(self, score: dict[str, Any], rules: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        refs = [r["policy_ref"] for r in rules["fired_rules"] if r.get("policy_ref")]
        direct = [{"chunk": c, "score": 1.0, "via": "policy_ref"} for c in self.store.by_codes(refs)]
        query = self.build_query(score, rules)
        seen = {d["chunk"].id for d in direct}
        semantic = [h for h in self.retrieve(query) if h["chunk"].id not in seen]
        return (direct + semantic)[: self.max_context_chunks], query

    def fallback_explanation(self, rules: dict[str, Any], context: list[dict[str, Any]]) -> dict[str, Any]:
        """LLM yokken: kural açıklamaları + politika başlıkları (deterministik)."""
        titles = {i["chunk"].policy_code: i["chunk"].title for i in context if i["chunk"].policy_code}
        sentences = [f"{r['explanation']} [{r['policy_ref']}: {titles.get(r['policy_ref'], '')}]"
                     for r in rules["fired_rules"] if r.get("policy_ref")]
        return {"explanation": " ".join(sentences) or "Hiçbir kural tetiklenmedi.",
                "recommended_action": rules["final_decision"], "next_steps": [],
                "citations": [r["policy_ref"] for r in rules["fired_rules"] if r.get("policy_ref")]}

    def explain_transaction(self, score: dict[str, Any], rules: dict[str, Any],
                            notes: list[str] | None = None) -> dict[str, Any]:
        """notes: başka agent'lardan gelen ek bilgi (ör. kullanıcı profili), özete eklenir."""
        t0 = time.time()
        context, query = self.gather_context(score, rules)
        chunks = [i["chunk"] for i in context]
        retrieved_codes = {c.policy_code for c in chunks if c.policy_code}
        summary = self.summarize(score, rules)
        if notes:
            summary += "\nEk bilgiler:\n" + "\n".join(f"- {n}" for n in notes)

        trace = [
            {"step": "policy_ref", "codes": [i["chunk"].policy_code for i in context if i["via"] == "policy_ref"]},
            {"step": "vector_search", "query": query,
             "codes": [i["chunk"].policy_code for i in context if i["via"] == "vector"]},
        ]
        prompt = prompts.EXPLAIN.format(context=prompts.format_context(chunks), summary=summary,
                                        decision=rules["final_decision"])
        try:
            out = self.llm.generate_json(prompt, system=prompts.SYSTEM)
            mode = "llm"
        except LLMError as e:
            out, mode = self.fallback_explanation(rules, context), "fallback"
            trace.append({"step": "llm_error", "error": str(e)})

        explanation = str(out.get("explanation", ""))
        # sadece politika kodu biçimindeki atıflar doğrulanır; model bazen kural id'si (R002) de listeliyor
        listed = [c for c in (out.get("citations") or []) if isinstance(c, str) and POLICY_CODE.fullmatch(c.strip())]
        cited = [c.strip() for c in listed] + POLICY_CODE.findall(explanation)
        action = str(out.get("recommended_action", "")).upper().strip()
        steps = [s for s in (out.get("next_steps") or []) if isinstance(s, str) and s.strip(" .") and "..." not in s]
        trace.append({"step": "generate", "mode": mode})
        return {
            "explanation": explanation,
            "recommended_action": action,
            "engine_decision": rules["final_decision"],
            # karar kural motorunda; LLM farklı önerirse analist için işaretlenir, karar değişmez
            "llm_agrees_with_engine": action == rules["final_decision"],
            "next_steps": steps,
            **verify_citations(cited, retrieved_codes),
            "sources": [self._source(i) for i in context],
            "mode": mode,
            "trace": trace,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
