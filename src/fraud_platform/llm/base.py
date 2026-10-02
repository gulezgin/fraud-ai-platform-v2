"""LLM arayüzü. Uygulama kodu sağlayıcıyı bilmez: testlerde sahte LLM, ileride başka bir sağlayıcı tek sınıfla eklenir."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """LLM'e ulaşılamadı veya cevap üretilemedi."""


class BaseLLM(ABC):
    name: str = ""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """JSON cevap ister; küçük modeller bazen JSON'u metin içine gömer, ilk {...} bloğu ayıklanır."""
        text = self.generate(prompt, system=system, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise LLMError(f"LLM geçerli JSON döndürmedi: {text[:200]}")

    def health(self) -> bool:
        try:
            self.generate("ping")
            return True
        except LLMError:
            return False
