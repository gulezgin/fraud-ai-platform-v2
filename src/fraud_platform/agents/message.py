"""Agent'lar arası mesaj. Her istek bir cevap üretir; cevap reply_to ile isteğe, conversation_id ile konuşmaya bağlanır."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Message:
    sender: str
    receiver: str
    task: str
    payload: dict[str, Any]
    kind: Literal["request", "reply"] = "request"
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    conversation_id: str = field(default_factory=_new_id)
    reply_to: str | None = None
    id: str = field(default_factory=_new_id)
    ts: float = field(default_factory=time.time)

    def reply(self, payload: dict[str, Any], status: Literal["ok", "error"] = "ok", error: str | None = None) -> Message:
        return Message(self.receiver, self.sender, self.task, payload, kind="reply", status=status, error=error,
                       conversation_id=self.conversation_id, reply_to=self.id)


def summarize_payload(payload: dict[str, Any], max_len: int = 80) -> dict[str, Any]:
    """İz (trace) için kısa özet: büyük tablolar ve uzun metinler kaydedilmez."""
    out = {}
    for k, v in payload.items():
        if isinstance(v, pd.DataFrame):
            out[k] = f"DataFrame({v.shape[0]}x{v.shape[1]})"
        elif isinstance(v, pd.Series):
            out[k] = f"Series({len(v)})"
        elif isinstance(v, dict):
            out[k] = f"dict({len(v)} alan)"
        elif isinstance(v, list):
            out[k] = f"list({len(v)})"
        elif isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + "..."
        else:
            out[k] = v
    return out
