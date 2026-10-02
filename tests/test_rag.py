"""RAG testleri Ollama gerektirmez: TF-IDF embedder + sahte LLM."""
import json

import pytest

from fraud_platform.llm.base import BaseLLM, LLMError
from fraud_platform.rag.chunker import chunk_directory, chunk_markdown
from fraud_platform.rag.embedder import TfidfEmbedder
from fraud_platform.rag.pipeline import RAGPipeline, verify_citations
from fraud_platform.rag.vector_store import FaissVectorStore

KB = """# Kart Politikası

Giriş paragrafı.

## FP-07 Yeni Kart Politikası

İlk 7 gün içinde 500 USD üzeri işlemler izlenir. Günlük limit 1500 USD.

## FP-12 Cihaz Paylaşımı

Aynı cihazda 5 farklı kart kart test saldırısıdır. Cihaz 48 saat gözetime alınır.
"""


class FakeLLM(BaseLLM):
    name = "fake"

    def __init__(self, reply: str | dict | Exception):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return json.dumps(self.reply, ensure_ascii=False) if isinstance(self.reply, dict) else self.reply


@pytest.fixture
def kb_dir(tmp_path):
    (tmp_path / "kart.md").write_text(KB, encoding="utf-8")
    return tmp_path


@pytest.fixture
def store(kb_dir):
    return FaissVectorStore(TfidfEmbedder()).build(chunk_directory(kb_dir))


def test_chunker_splits_by_policy_heading(kb_dir):
    chunks = chunk_markdown(kb_dir / "kart.md")
    assert [c.policy_code for c in chunks] == [None, "FP-07", "FP-12"]
    assert chunks[1].title == "Yeni Kart Politikası" and "1500 USD" in chunks[1].text
    assert chunks[1].embedding_text.startswith("Kart Politikası > Yeni Kart Politikası")


def test_long_section_is_split_on_paragraphs(tmp_path):
    body = "\n\n".join(" ".join(["kelime"] * 30) for _ in range(4))
    (tmp_path / "uzun.md").write_text(f"# Doc\n\n## FP-99 Uzun\n\n{body}\n", encoding="utf-8")
    chunks = chunk_markdown(tmp_path / "uzun.md", max_words=70)
    assert len(chunks) == 2 and all(c.policy_code == "FP-99" for c in chunks)


def test_search_and_lookup_by_code(store):
    top_chunk, score = store.search("cihaz kart test saldırısı", k=1)[0]
    assert top_chunk.policy_code == "FP-12" and score > 0
    assert [c.policy_code for c in store.by_codes(["FP-07"])] == ["FP-07"]


def test_store_roundtrip_on_non_ascii_path(tmp_path, store):
    # Windows'ta faiss.write_index ASCII olmayan yolu açamıyordu; serialize ile yazılıyor
    target = tmp_path / "klasör_ı"
    store.save(target)
    loaded = FaissVectorStore.load(target, TfidfEmbedder())
    assert loaded.index.ntotal == len(store.chunks)
    assert loaded.search("yeni kart limit", 1)[0][0].policy_code == "FP-07"


def test_verify_citations():
    assert verify_citations(["FP-07"], {"FP-07", "FP-12"})["citations_verified"] is True
    v = verify_citations(["FP-07", "FP-99"], {"FP-07"})
    assert v["citations_verified"] is False and v["unsupported_citations"] == ["FP-99"]
    assert verify_citations([], {"FP-07"})["citations_verified"] is False   # atıfsız cevap doğrulanmış sayılmaz


def test_query_injects_context_and_verifies(store):
    llm = FakeLLM("Günlük limit 1500 USD'dir [FP-07].")
    out = RAGPipeline(store, llm, top_k=2).query("yeni kartın günlük limiti")
    assert "FP-07" in llm.prompts[0] and "1500 USD" in llm.prompts[0]     # context injection
    assert out["citations"] == ["FP-07"] and out["citations_verified"]


def test_query_flags_hallucinated_citation(store):
    out = RAGPipeline(store, FakeLLM("Limit 3000 USD [FP-77]."), top_k=2).query("yeni kart limiti")
    assert out["citations_verified"] is False and out["unsupported_citations"] == ["FP-77"]


SCORE = {"risk_percentile": 0.99, "raw_percentile": 0.98, "context_factor": 1.2, "context_adjustments": [],
         "reasons": {"entity": [{"text": "Bu cihazda daha önce 7 farklı kart görülmüş"}]}}
RULES = {"final_decision": "REVIEW", "decision_reason": "test", "fired_rules": [
    {"id": "R007", "name": "Cihaz paylaşımı", "action": "FLAG", "policy_ref": "FP-12", "explanation": "7 kart"}]}


def test_explain_transaction_uses_policy_ref_first(store):
    llm = FakeLLM({"explanation": "Kart test şüphesi [FP-12].", "recommended_action": "REVIEW",
                   "next_steps": ["Cihazı 48 saat gözetime al", "..."], "citations": ["FP-12", "R007"]})
    out = RAGPipeline(store, llm, top_k=1).explain_transaction(SCORE, RULES)
    assert out["sources"][0]["policy_code"] == "FP-12" and out["sources"][0]["via"] == "policy_ref"
    assert out["citations_verified"] and out["citations"] == ["FP-12"]      # kural id'si atıf sayılmaz
    assert out["llm_agrees_with_engine"] and out["next_steps"] == ["Cihazı 48 saat gözetime al"]
    assert "REVIEW" in llm.prompts[0]                                        # motorun kararı prompt'ta


def test_explain_transaction_falls_back_without_llm(store):
    out = RAGPipeline(store, FakeLLM(LLMError("bağlantı yok"))).explain_transaction(SCORE, RULES)
    assert out["mode"] == "fallback" and out["recommended_action"] == "REVIEW"
    assert "FP-12" in out["explanation"] and out["citations_verified"]


def test_generate_json_extracts_embedded_object():
    assert FakeLLM('Cevap: {"a": 1} bitti').generate_json("x") == {"a": 1}
    with pytest.raises(LLMError):
        FakeLLM("json yok").generate_json("x")
