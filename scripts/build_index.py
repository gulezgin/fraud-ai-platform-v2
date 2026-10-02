"""Bilgi tabanından (knowledge_base/*.md) parça + embedding + FAISS index üretir.

    python scripts/build_index.py              # settings.yaml'daki embedder (bge-m3)
    python scripts/build_index.py --embedder tfidf

Ollama embedder için `ollama pull bge-m3` ve Ollama servisinin çalışıyor olması gerekir.
"""
import argparse
import logging
import time

from fraud_platform.config import get_settings
from fraud_platform.rag.chunker import chunk_directory
from fraud_platform.rag.factory import build_embedder
from fraud_platform.rag.vector_store import FaissVectorStore

logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["ollama", "tfidf"], default=None)
    args = parser.parse_args()

    s = get_settings()
    chunks = chunk_directory(s.rag.knowledge_base_dir, s.rag.max_chunk_words)
    logger.info("%d parça, %d politika kodu", len(chunks), len({c.policy_code for c in chunks if c.policy_code}))

    start = time.time()
    store = FaissVectorStore(build_embedder(s, args.embedder)).build(chunks)
    store.save(s.rag.index_dir)
    logger.info("index (%s, %d boyut) %.1f sn: %s", store.embedder.name, store.index.d, time.time() - start, s.rag.index_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
