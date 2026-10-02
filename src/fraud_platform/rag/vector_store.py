"""FAISS vektör deposu: parçalar + vektörler diske kaydedilir, API her açılışta embedding hesaplamaz."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import joblib
import numpy as np

from fraud_platform.rag.chunker import Chunk
from fraud_platform.rag.embedder import BaseEmbedder, TfidfEmbedder


class FaissVectorStore:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None

    def build(self, chunks: list[Chunk]) -> FaissVectorStore:
        texts = [c.embedding_text for c in chunks]
        vectors = self.embedder.fit(texts).embed(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])   # normalize vektörde iç çarpım = kosinüs
        self.index.add(vectors)
        self.chunks = chunks
        return self

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def by_codes(self, codes: list[str]) -> list[Chunk]:
        wanted = set(codes)
        return [c for c in self.chunks if c.policy_code in wanted]

    @property
    def codes(self) -> set[str]:
        return {c.policy_code for c in self.chunks if c.policy_code}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # faiss.write_index C++ tarafında dosya açar ve Windows'ta ASCII olmayan yolları (ör. "msı") açamaz;
        # index belleğe serileştirilip Python ile yazılır
        (directory / "faiss.index").write_bytes(faiss.serialize_index(self.index).tobytes())
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "model": getattr(self.embedder, "model", None),
                "dim": self.index.d, "n_chunks": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if isinstance(self.embedder, TfidfEmbedder):
            joblib.dump(self.embedder, directory / "tfidf.joblib")   # kelime dağarcığı sorgu için gerekli

    @classmethod
    def load(cls, directory: Path, embedder: BaseEmbedder) -> FaissVectorStore:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index {meta['embedder']} ile kurulmuş, verilen embedder {embedder.name}")
        if embedder.name == "tfidf":
            embedder = joblib.load(directory / "tfidf.joblib")
        store = cls(embedder)
        store.index = faiss.deserialize_index(np.frombuffer((directory / "faiss.index").read_bytes(), dtype="uint8"))
        store.chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        if store.index.ntotal != len(store.chunks):
            raise ValueError("index ve parça sayısı uyuşmuyor; build_index.py yeniden çalıştırılmalı")
        return store
