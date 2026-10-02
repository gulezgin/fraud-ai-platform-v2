"""Agent temel sınıfı. Agent iş mantığı içermez: önceki adımların modüllerini (servis, kural motoru, RAG) çağırır.

Görev adı -> on_<görev> metodu. Agent sınırında hata yakalanır ve hata cevabı olarak döner; tek bir agent'ın hatası
sistemi çökertmez, izde görünür ve orchestrator karar verir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fraud_platform.agents.message import Message

if TYPE_CHECKING:
    from fraud_platform.agents.bus import MessageBus


@dataclass(frozen=True)
class Capability:
    """Agent'ın bir görevi: orchestrator planı ve görev girdilerini bu beyandan çıkarır (sabit tablo yok)."""
    description: str
    requires: tuple[str, ...] = ()        # önce çalışması gereken görevler
    inputs: tuple[str, ...] = ()          # paylaşılan durumdan (state) okuduğu alanlar
    provides: tuple[str, ...] = ()        # duruma yazdığı alanlar
    needs_transaction: bool = False


class BaseAgent(ABC):
    name: str = ""
    description: str = ""

    def __init__(self):
        self.bus: MessageBus | None = None

    @abstractmethod
    def capabilities(self) -> dict[str, Capability]: ...

    def handle(self, msg: Message) -> Message:
        handler = getattr(self, f"on_{msg.task}", None)
        if handler is None:
            return msg.reply({}, status="error", error=f"{self.name} '{msg.task}' görevini bilmiyor")
        try:
            return msg.reply(handler(msg.payload, msg))
        except Exception as e:  # noqa: BLE001 - agent sınırı: hata cevaba dönüşür, sistem çökmez
            return msg.reply({}, status="error", error=f"{type(e).__name__}: {e}")

    def ask(self, receiver: str, task: str, payload: dict[str, Any], parent: Message) -> Message:
        """Başka bir agent'a aynı konuşma içinde istek gönderir (agent-to-agent)."""
        if self.bus is None:
            raise RuntimeError(f"{self.name} bir bus'a kayıtlı değil")
        return self.bus.send(Message(self.name, receiver, task, payload,
                                     conversation_id=parent.conversation_id, reply_to=parent.id))
