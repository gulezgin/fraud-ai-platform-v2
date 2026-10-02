"""Orchestrator (supervisor): isteği planlar, görevleri bus üzerinden ilgili agent'lara dağıtır, sonuçları birleştirir.

Görev -> agent eşlemesi ve görev girdileri agent'ların beyan ettiği yeteneklerden gelir (sabit tablo yok).
Ara sonuca göre dinamik karar: doğrulama başarısızsa akış durur; karar ALLOW ise maliyetli LLM soruşturması atlanır
(istenirse force_investigation ile zorlanır). Bir agent hata verirse kısmi sonuç ve hata döner, sistem çökmez.
"""
from __future__ import annotations

import time
from typing import Any

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.agents.planner import TaskPlanner

HIDDEN_STATE = {"features", "assessment", "transaction"}    # büyük / ham nesneler rapora girmez


class Orchestrator(BaseAgent):
    name = "orchestrator"
    description = "İsteği planlar, görevleri agent'lara dağıtır, sonuçları birleştirir"

    def __init__(self, planner: TaskPlanner, investigate_on: tuple[str, ...] = ("BLOCK", "REVIEW", "FLAG")):
        super().__init__()
        self.planner = planner
        self.investigate_on = investigate_on

    def capabilities(self) -> dict[str, Capability]:
        return {"handle_request": Capability("Kullanıcı isteğini planlayıp ilgili agent'lara dağıtır")}

    def run(self, goal: str, transaction: dict[str, Any] | None = None, question: str | None = None,
            strategy: str | None = None, force_investigation: bool = False) -> dict[str, Any]:
        """Dış giriş noktası: istek de bus'tan geçer, böylece izin ilk satırı kullanıcı isteğidir."""
        msg = Message("user", self.name, "handle_request",
                      {"goal": goal, "transaction": transaction, "question": question,
                       "strategy": strategy, "force_investigation": force_investigation})
        reply = self.bus.send(msg)
        if reply.status == "error":
            return {"conversation_id": msg.conversation_id, "status": "error", "error": reply.error,
                    "trace": self.bus.trace(msg.conversation_id)}
        return {**reply.payload, "trace": self.bus.trace(msg.conversation_id)}

    def on_handle_request(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        caps = {t: c for t, (agent, c) in self.bus.capabilities().items() if agent != self.name}
        has_tx = payload.get("transaction") is not None
        plan = self.planner.plan(payload["goal"], caps, has_tx)

        state: dict[str, Any] = {k: v for k, v in payload.items() if v is not None}
        state.setdefault("question", payload["goal"])
        steps, status, error = [], "ok", None

        for task in plan.tasks:
            if task == "investigate" and not payload.get("force_investigation"):
                decision = state["rules"]["final_decision"]
                if decision not in self.investigate_on:
                    steps.append({"task": task, "status": "skipped",
                                  "reason": f"karar {decision}: LLM soruşturması gerekmiyor (maliyet)"})
                    continue

            agent = self.bus.route(task)
            inputs = {k: state[k] for k in caps[task].inputs if k in state}
            start = time.time()
            reply = self.ask(agent, task, inputs, parent=msg)
            steps.append({"task": task, "agent": agent, "status": reply.status,
                          "duration_ms": round((time.time() - start) * 1000, 1), **({"error": reply.error} if reply.error else {})})
            if reply.status == "error":
                status, error = "error", f"{task} başarısız: {reply.error}"
                break
            state.update(reply.payload)
            if task == "validate_transaction" and not state["validation"]["valid"]:
                status, error = "invalid", "işlem doğrulanamadı: " + "; ".join(state["validation"]["errors"])
                break

        return self._report(msg, plan, steps, state, status, error)

    @staticmethod
    def _report(msg, plan, steps, state, status, error) -> dict[str, Any]:
        rules, score = state.get("rules"), state.get("score")
        report = {
            "conversation_id": msg.conversation_id, "status": status, "error": error,
            "goal": state.get("goal"), "plan": plan.to_dict(), "steps": steps,
            "decision": rules["final_decision"] if rules else None,
            "risk_percentile": score["risk_percentile"] if score else None,
        }
        report.update({k: v for k, v in state.items() if k not in HIDDEN_STATE and k not in report and k != "force_investigation"})
        return report
