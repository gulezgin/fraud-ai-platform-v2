"""RAG değerlendirmesi: retrieval (hit@1, hit@3, MRR; bge-m3 vs TF-IDF) ve üretim (atıf + kurgusal bilgi; RAG vs bağlamsız).

    python scripts/eval_rag.py               # retrieval + üretim (Ollama gerekir, ~2-3 dk)
    python scripts/eval_rag.py --no-llm      # sadece retrieval

Politikalar kurgusal olduğu için LLM'in "72 saat", "gölge mod" gibi bilgileri önceden bilmesi mümkün değil:
cevapta geçiyorsa RAG çalışıyor demektir. Çıktı: artifacts/rag/eval.json
"""
import argparse
import json
import logging

import pandas as pd
import yaml

from fraud_platform.config import get_settings
from fraud_platform.rag import prompts
from fraud_platform.rag.chunker import chunk_directory
from fraud_platform.rag.factory import build_embedder, build_llm, load_pipeline
from fraud_platform.rag.vector_store import FaissVectorStore


def retrieval_metrics(store: FaissVectorStore, questions: list[dict], k: int = 5) -> dict:
    ranks = []
    for q in questions:
        codes = [c.policy_code for c, _ in store.search(q["question"], k)]
        ranks.append(codes.index(q["expected"]) + 1 if q["expected"] in codes else None)
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(bool(r and r <= 3) for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "ranks": ranks,
    }


def has_fact(text: str | None, facts: list[str]) -> bool:
    text = (text or "").lower()
    return any(f.lower() in text for f in facts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    s = get_settings()
    questions = yaml.safe_load(s.rag.eval_file.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q["expected"]]
    chunks = chunk_directory(s.rag.knowledge_base_dir, s.rag.max_chunk_words)

    report = {"retrieval": {}}
    for kind in ("tfidf", "ollama"):
        store = FaissVectorStore(build_embedder(s, kind)).build(chunks)
        report["retrieval"][kind] = retrieval_metrics(store, answerable)
    print(f"retrieval ({len(answerable)} soru):")
    print(pd.DataFrame(report["retrieval"]).T.drop(columns="ranks").round(3).to_string())

    if not args.no_llm:
        rag, llm = load_pipeline(s), build_llm(s)
        rows = []
        for q in questions:
            r = rag.query(q["question"])
            bare = llm.generate(prompts.NO_CONTEXT.format(question=q["question"]))
            rows.append({
                "expected": q["expected"],
                "answerable": bool(q["expected"]),
                "rag_cites_expected": q["expected"] in r["citations"] if q["expected"] else None,
                "rag_citations_verified": r["citations_verified"],
                "rag_fact": has_fact(r["answer"], q["facts"]),
                "no_context_fact": has_fact(bare, q["facts"]),
                "rag_answer": r["answer"],
                "no_context_answer": bare,
                "latency_ms": r["latency_ms"],
            })
        gen = pd.DataFrame(rows)
        ans, unans = gen[gen["answerable"]], gen[~gen["answerable"]]
        report["generation"] = {
            "summary": {
                "cevaplanabilir: doğru politikaya atıf": ans["rag_cites_expected"].mean(),
                "cevaplanabilir: atıflar doğrulandı": ans["rag_citations_verified"].mean(),
                "cevaplanabilir: kurgusal bilgi (RAG)": ans["rag_fact"].mean(),
                "cevaplanabilir: kurgusal bilgi (bağlamsız)": ans["no_context_fact"].mean(),
                "cevaplanamaz: 'bilgi yok' dedi": unans["rag_fact"].mean() if len(unans) else None,
            },
            "median_latency_ms": float(gen["latency_ms"].median()),
            "rows": rows,
        }
        print()
        print(pd.Series(report["generation"]["summary"]).round(3).to_string())
        print(f"medyan gecikme: {report['generation']['median_latency_ms']:.0f} ms")

    out = s.rag.index_dir / "eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nkaydedildi: {out}")


if __name__ == "__main__":
    main()
