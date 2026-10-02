"""Politika markdown'larını başlıklara göre parçalar. Her "## FP-xx Başlık" bölümü bir parça; politika kodu metadata'ya yazılır.

Politika kodu, kural motorundaki policy_ref ile birebir eşleşir: tetiklenen kural, ilgili politika parçasını doğrudan getirir.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

POLICY_HEADING = re.compile(r"^##\s+(?P<code>[A-Z]{2,}-\d+)\s+(?P<title>.+)$")
POLICY_CODE = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")


@dataclass
class Chunk:
    id: str
    source: str
    document: str
    policy_code: str | None
    title: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embedding_text(self) -> str:
        """Embedding'e giden metin: doküman ve bölüm başlığı bağlam olarak eklenir."""
        return f"{self.document} > {self.title}\n{self.text}"


def _split_long(text: str, max_words: int) -> list[str]:
    """Uzun bölümü paragraf sınırlarından böler; tek paragraf max_words'ü aşsa da bölünmez."""
    parts, current = [], []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if current and len(" ".join(current + [para]).split()) > max_words:
            parts.append("\n\n".join(current))
            current = []
        current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(path: Path, max_words: int = 220) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    document = next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem)

    sections: list[tuple[str | None, str, list[str]]] = [(None, document, [])]
    for line in lines:
        m = POLICY_HEADING.match(line)
        if m:
            sections.append((m["code"], m["title"].strip(), []))
        elif not line.startswith("# "):
            sections[-1][2].append(line)

    chunks = []
    for code, title, body in sections:
        text = "\n".join(body).strip()
        if not text:
            continue
        for i, part in enumerate(_split_long(text, max_words)):
            cid = f"{path.stem}:{code or 'giris'}" + (f":{i}" if i else "")
            chunks.append(Chunk(cid, path.name, document, code, title, " ".join(part.split())))
    return chunks


def chunk_directory(directory: Path, max_words: int = 220) -> list[Chunk]:
    return [c for p in sorted(Path(directory).glob("*.md")) for c in chunk_markdown(p, max_words)]
