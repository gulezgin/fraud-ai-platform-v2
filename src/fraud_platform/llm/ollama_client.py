"""Yerel LLM: Ollama (case ücretli API gerektirmiyor; model settings.yaml -> llm.model)."""
from __future__ import annotations

import httpx
import ollama

from fraud_platform.llm.base import BaseLLM, LLMError


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str, host: str, temperature: float = 0.1, num_ctx: int = 4096, timeout: float = 180,
                 max_tokens: int = 400):
        self.model = model
        # num_predict sınırı şart: sıcaklık 0'da küçük model tekrar döngüsüne girip bağlam dolana kadar üretebiliyor
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_tokens, "repeat_penalty": 1.1}
        self.client = ollama.Client(host=host, timeout=timeout)

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            r = self.client.chat(model=self.model, messages=messages, options=self.options,
                                 format="json" if json_mode else None)
        except Exception as e:   # bağlantı, model yok, zaman aşımı
            raise LLMError(f"Ollama hatası ({self.model}): {e}") from e
        return r["message"]["content"]

    def health(self) -> bool:
        try:
            names = {m.model for m in self.client.list().models}
        except (ConnectionError, ollama.ResponseError, httpx.HTTPError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)
