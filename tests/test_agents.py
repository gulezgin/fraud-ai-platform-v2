"""Agent katmanı testleri: modeller yüklenmez, basit taklit (stub) agent'lar ve sahte LLM kullanılır."""
import pytest

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.bus import MessageBus
from fraud_platform.agents.investigator_agent import InvestigatorAgent
from fraud_platform.agents.message import Message
from fraud_platform.agents.orchestrator import Orchestrator
from fraud_platform.agents.planner import TaskPlanner
from fraud_platform.llm.base import BaseLLM, LLMError


class FakeLLM(BaseLLM):
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, system=None, json_mode=False):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class StubData(BaseAgent):
    name = "data_agent"

    def capabilities(self):
        return {"validate_transaction": Capability("doğrula", inputs=("transaction",), needs_transaction=True),
                "dataset_summary": Capability("veri seti özeti")}

    def on_validate_transaction(self, payload, msg):
        ok = payload["transaction"].get("TransactionAmt") is not None
        return {"validation": {"valid": ok, "errors": [] if ok else ["zorunlu alan eksik: TransactionAmt"]}}

    def on_dataset_summary(self, payload, msg):
        return {"dataset": {"rows": 10}}


class StubFeature(BaseAgent):
    name = "feature_agent"

    def capabilities(self):
        return {"build_features": Capability("feature", requires=("validate_transaction",), inputs=("transaction",),
                                             needs_transaction=True),
                "get_entity_profile": Capability("profil", inputs=("uid",))}

    def on_build_features(self, payload, msg):
        return {"features": {"amt": payload["transaction"]["TransactionAmt"]}, "uid": "u1"}

    def on_get_entity_profile(self, payload, msg):
        return {"profile": {"uid": payload["uid"], "known": True, "n_tx": 7, "avg_amount": 40.0,
                            "max_amount": 90.0, "n_devices": 2}}


class StubScoring(BaseAgent):
    name = "scoring_agent"

    def capabilities(self):
        return {"score": Capability("skor", requires=("build_features",), inputs=("features",))}

    def on_score(self, payload, msg):
        if payload["features"]["amt"] < 0:
            raise ValueError("negatif tutar")
        risk = 0.999 if payload["features"]["amt"] > 1000 else 0.2
        return {"score": {"uid": "u1", "risk_percentile": risk}, "assessment": {"risk": risk}}


class StubRules(BaseAgent):
    name = "rule_agent"

    def capabilities(self):
        return {"evaluate_rules": Capability("kural", requires=("score",), inputs=("assessment",))}

    def on_evaluate_rules(self, payload, msg):
        return {"rules": {"final_decision": "BLOCK" if payload["assessment"]["risk"] > 0.99 else "ALLOW", "fired_rules": []}}


class StubRAG:
    def __init__(self):
        self.notes = None

    def explain_transaction(self, score, rules, notes=None):
        self.notes = notes
        return {"explanation": f"karar {rules['final_decision']} [FP-01]", "citations_verified": True}

    def query(self, question):
        return {"answer": f"cevap: {question}"}


def build(planner=None, investigate_on=("BLOCK", "REVIEW", "FLAG")):
    bus, rag = MessageBus(), StubRAG()
    orch = Orchestrator(planner or TaskPlanner(None, "rules"), investigate_on)
    for agent in (orch, StubData(), StubFeature(), StubScoring(), StubRules(), InvestigatorAgent(rag)):
        bus.register(agent)
    return orch, bus, rag


def test_bus_rejects_duplicates():
    bus = MessageBus()
    bus.register(StubData())
    with pytest.raises(ValueError, match="zaten kayıtlı"):
        bus.register(StubData())

    class Clash(BaseAgent):
        name = "clash"

        def capabilities(self):
            return {"dataset_summary": Capability("aynı görev")}

    bus.register(Clash())
    with pytest.raises(ValueError, match="iki agent"):
        bus.capabilities()


def test_full_flow_with_agent_to_agent_message():
    orch, _, rag = build()
    out = orch.run("Bu işlemi değerlendir ve nedenini açıkla", transaction={"TransactionAmt": 5000})

    assert out["status"] == "ok" and out["decision"] == "BLOCK"
    assert out["plan"]["tasks"] == ["validate_transaction", "build_features", "score", "evaluate_rules", "investigate"]
    requests = [(e["from"], e["to"], e["task"]) for e in out["trace"] if e["kind"] == "request"]
    assert ("investigator_agent", "feature_agent", "get_entity_profile") in requests    # agent'tan agent'a
    assert requests[0] == ("user", "orchestrator", "handle_request")
    assert "7 işlem" in rag.notes[0]                                                    # profil açıklamaya girdi
    assert "features" not in out and "transaction" not in out                          # büyük nesneler rapora girmez


def test_investigation_skipped_for_allow_unless_forced():
    orch, _, _ = build()
    out = orch.run("nedenini açıkla", transaction={"TransactionAmt": 50})
    assert out["decision"] == "ALLOW" and "investigation" not in out
    assert out["steps"][-1]["status"] == "skipped"

    forced = orch.run("nedenini açıkla", transaction={"TransactionAmt": 50}, force_investigation=True)
    assert forced["investigation"]["explanation"] == "karar ALLOW [FP-01]"


def test_invalid_transaction_stops_flow():
    orch, _, _ = build()
    out = orch.run("risk skoru", transaction={"TransactionAmt": None})
    assert out["status"] == "invalid" and "TransactionAmt" in out["error"]
    assert [s["task"] for s in out["steps"]] == ["validate_transaction"]


def test_agent_error_does_not_crash_system():
    orch, _, _ = build()
    out = orch.run("risk skoru", transaction={"TransactionAmt": -5})
    assert out["status"] == "error" and "negatif tutar" in out["error"]
    assert any(e["status"] == "error" and e["task"] == "score" for e in out["trace"])


def test_unknown_task_message_returns_error():
    bus = MessageBus()
    bus.register(StubData())
    reply = bus.send(Message("test", "data_agent", "yok_boyle", {}))
    assert reply.status == "error" and "bilmiyor" in reply.error


def test_requests_without_transaction():
    orch, _, _ = build()
    assert orch.run("veri setinde kaç satır var")["dataset"] == {"rows": 10}
    assert orch.run("FP-12 ne diyor?")["answer"]["answer"] == "cevap: FP-12 ne diyor?"


# ---------------------------------------------------------------- planlayıcı

def caps():
    agents = [StubData(), StubFeature(), StubScoring(), StubRules(), InvestigatorAgent(None)]
    return {t: c for a in agents for t, c in a.capabilities().items()}


def test_repair_adds_prerequisites_and_drops_infeasible():
    p = TaskPlanner.repair(["evaluate_rules", "uydurma"], caps(), has_transaction=True)
    assert p.tasks == ["validate_transaction", "build_features", "score", "evaluate_rules"]
    assert any("bilinmeyen" in r for r in p.repairs)

    p = TaskPlanner.repair(["score", "answer_policy_question"], caps(), has_transaction=False)
    assert p.tasks == ["answer_policy_question"]            # score'un ön koşulu işlem ister -> düşer


def test_repair_detects_cycle():
    cyc = {"a": Capability("a", requires=("b",)), "b": Capability("b", requires=("a",))}
    with pytest.raises(ValueError, match="döngüsel"):
        TaskPlanner.repair(["a"], cyc, has_transaction=False)


def test_hybrid_uses_rules_first_then_llm():
    llm = FakeLLM('{"task": "score", "reason": "tuhaflık sorusu"}')
    planner = TaskPlanner(llm, "hybrid")
    p = planner.plan("kararın nedenini açıkla", caps(), True)
    assert p.source == "rules" and p.tasks[-1] == "investigate" and llm.calls == 0

    p = planner.plan("bu harcama ne kadar tuhaf?", caps(), True)
    assert p.source == "llm" and p.tasks[-1] == "score" and llm.calls == 1


def test_planner_falls_back_to_default():
    p = TaskPlanner(FakeLLM(LLMError("bağlantı yok")), "llm").plan("bu harcama ne kadar tuhaf?", caps(), True)
    assert p.source == "default" and p.tasks[-1] == "investigate" and "ulaşılamadı" in p.reason

    p = TaskPlanner(FakeLLM('{"task": "uydurma_gorev"}'), "hybrid").plan("tuhaf mı", caps(), False)
    assert p.source == "default" and p.tasks == ["answer_policy_question"]


def test_planner_rejects_unknown_mode():
    with pytest.raises(ValueError):
        TaskPlanner(None, "sihir")
