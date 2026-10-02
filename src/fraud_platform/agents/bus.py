"""Message bus (Mediator pattern): agent'lar birbirini doğrudan çağırmaz, bus üzerinden mesajlaşır.

Bus mesajı alıcıya yönlendirir, cevabı döndürür ve her ikisini kaydeder. Kayıt, /explain cevabında "agent izi" olarak
gösterilir. Agent'lar yeteneklerini (görev adı, ön koşullar) kayıt sırasında bildirir; orchestrator görevleri sabit bir
tablodan değil bu kayıttan yönlendirir.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fraud_platform.agents.message import Message, summarize_payload

if TYPE_CHECKING:
    from fraud_platform.agents.base import BaseAgent, Capability


class MessageBus:
    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.log: list[dict[str, Any]] = []

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self.agents:
            raise ValueError(f"'{agent.name}' zaten kayıtlı")
        self.agents[agent.name] = agent
        agent.bus = self

    def capabilities(self) -> dict[str, tuple[str, Capability]]:
        """görev -> (agent adı, yetenek)."""
        caps = {}
        for agent in self.agents.values():
            for task, cap in agent.capabilities().items():
                if task in caps:
                    raise ValueError(f"'{task}' görevi iki agent'ta tanımlı: {caps[task][0]}, {agent.name}")
                caps[task] = (agent.name, cap)
        return caps

    def route(self, task: str) -> str:
        caps = self.capabilities()
        if task not in caps:
            raise KeyError(f"'{task}' görevini yapan agent yok")
        return caps[task][0]

    def _record(self, msg: Message, duration_ms: float | None = None) -> None:
        self.log.append({
            "id": msg.id, "conversation": msg.conversation_id, "reply_to": msg.reply_to,
            "from": msg.sender, "to": msg.receiver, "task": msg.task, "kind": msg.kind, "status": msg.status,
            "error": msg.error, "duration_ms": duration_ms, "payload": summarize_payload(msg.payload),
        })

    def send(self, msg: Message) -> Message:
        if msg.receiver not in self.agents:
            raise KeyError(f"alıcı agent yok: {msg.receiver}")
        self._record(msg)
        start = time.time()
        reply = self.agents[msg.receiver].handle(msg)
        self._record(reply, round((time.time() - start) * 1000, 1))
        return reply

    def trace(self, conversation_id: str) -> list[dict[str, Any]]:
        return [e for e in self.log if e["conversation"] == conversation_id]

    def clear(self) -> None:
        self.log.clear()
