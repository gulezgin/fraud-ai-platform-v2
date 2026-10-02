"""API testleri. Ağır bileşenler DI container'da override edilir: model, veri ve Ollama gerekmez.

Gerçek artifact'larla çalışan entegrasyon testleri, artifact'lar yoksa atlanır.
"""
import pandas as pd
import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient

from fraud_platform.api.main import create_app
from fraud_platform.container import Container
from fraud_platform.llm.base import LLMError
from fraud_platform.services.transaction_repository import TransactionNotFound

TX = {"TransactionDT": 15055092, "TransactionAmt": 119.21, "ProductCD": "C", "card1": 5812}
SCORE = {"TransactionID": 1, "uid": "5812_NA_174", "layer_scores": {"column": 0.2}, "layer_percentiles": {"column": 0.99},
         "raw_anomaly_score": 0.4, "raw_percentile": 0.995, "context_factor": 1.22, "adjusted_score": 0.49,
         "risk_percentile": 0.9993, "context_adjustments": [], "reasons": {}}


class StubScoring:
    def __init__(self):
        self.seen = []

    def score(self, transactions, explain=True):
        self.seen.append(transactions[0])
        return [SCORE | {"reasons": {} if explain else None}]

    def assessment_frame(self, transactions):
        return pd.DataFrame([{"risk_percentile": 0.9993, "user_tx_count": 3, "tx_count_1h": 1, "TransactionAmt": 119.21}])


class StubRepo:
    def get(self, transaction_id):
        if transaction_id != 3554906:
            raise TransactionNotFound(f"işlem bulunamadı: {transaction_id}")
        return TX | {"TransactionID": 3554906}


class StubRAG:
    def __init__(self, fail=False):
        self.fail = fail

    def query(self, question, k=None):
        if self.fail:
            raise LLMError("Ollama kapalı")
        return {"question": question, "answer": "48 saat [FP-12]", "sources": [], "citations": ["FP-12"],
                "unsupported_citations": [], "citations_verified": True, "latency_ms": 1.0}


class StubLLM:
    def __init__(self, up=True):
        self.up = up

    def health(self):
        return self.up


class StubOrchestrator:
    def run(self, goal, transaction=None, strategy=None, force_investigation=False):
        return {"status": "ok", "plan": {"tasks": ["score"], "source": "rules"}, "steps": [],
                "score": SCORE, "rules": {"final_decision": "BLOCK", "final_risk": 1.0, "winning_rule": "R002",
                                          "decision_reason": "test", "conflict_strategy": "priority_then_severity",
                                          "fired_rules": []},
                "investigation": {"explanation": "açıklama [FP-01]", "citations": ["FP-01"]},
                "trace": [{"from": "user", "to": "orchestrator", "task": "handle_request"}], "conversation_id": "x"}


def make_client(rag=None, llm_up=True, scoring=None):
    c = Container()
    c.scoring_service.override(providers.Object(scoring or StubScoring()))
    c.transactions.override(providers.Object(StubRepo()))
    c.rag.override(providers.Object(rag or StubRAG()))
    c.llm.override(providers.Object(StubLLM(llm_up)))
    c.vector_store.override(providers.Object(type("VS", (), {"chunks": [1, 2, 3]})()))
    c.orchestrator.override(providers.Object(StubOrchestrator()))
    return TestClient(create_app(c))


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_reachable"] and h["knowledge_base_chunks"] == 3 and h["rules"] >= 10


def test_health_degraded_when_llm_down():
    with make_client(llm_up=False) as c:
        h = c.get("/health").json()
    assert h["status"] == "degraded" and not h["llm_reachable"]


def test_score_by_json_and_id(client):
    r = client.post("/score", json={"transaction": TX})
    assert r.status_code == 200 and r.json()["risk_percentile"] == 0.9993 and "latency_ms" in r.json()
    assert client.post("/score", json={"transaction_id": 3554906}).status_code == 200


def test_score_keeps_extra_fields():
    scoring = StubScoring()
    with make_client(scoring=scoring) as c:
        c.post("/score", json={"transaction": TX | {"V258": 3.0, "id_31": "chrome"}})
    assert scoring.seen[0]["V258"] == 3.0 and scoring.seen[0]["id_31"] == "chrome"   # V/id alanları kaybolmaz


@pytest.mark.parametrize("body,status", [
    ({"transaction_id": 1}, 404),                                     # kayıtlı değil
    ({}, 422),                                                        # ikisi de yok
    ({"transaction": TX, "transaction_id": 3554906}, 422),           # ikisi birden
    ({"transaction": TX | {"TransactionAmt": -5}}, 422),            # negatif tutar
    ({"transaction": {k: v for k, v in TX.items() if k != "card1"}}, 422),   # zorunlu alan yok
])
def test_score_errors(client, body, status):
    assert client.post("/score", json=body).status_code == status


def test_rules_list_evaluate_reload(client):
    rules = client.get("/rules").json()
    assert len([r for r in rules["rules"] if r["enabled"]]) >= 10

    fields = {"risk_percentile": 0.995, "user_tx_count": 4, "tx_count_1h": 12, "TransactionAmt": 820.0,
              "is_night": 1, "local_hour": 3}
    r = client.post("/rules/evaluate", json={"fields": fields}).json()
    assert r["final_decision"] == "BLOCK" and r["winning_rule"] == "R001"
    assert {f["id"] for f in r["fired_rules"]} >= {"R001", "R003", "R005"}

    by_id = client.post("/rules/evaluate", json={"transaction_id": 3554906}).json()
    assert by_id["final_decision"] == "BLOCK" and by_id["winning_rule"] == "R002"

    assert client.post("/rules/evaluate", json={"fields": fields, "strategy": "yok"}).status_code == 422
    assert client.post("/rules/reload").json()["rules"] >= 10


def test_rag_query_and_llm_down():
    with make_client() as c:
        r = c.post("/rag/query", json={"question": "Kart test saldırısında cihaz ne kadar gözetimde kalır?"})
        assert r.status_code == 200 and r.json()["citations_verified"]
        assert c.post("/rag/query", json={"question": "a"}).status_code == 422
    with make_client(rag=StubRAG(fail=True)) as c:
        assert c.post("/rag/query", json={"question": "FP-12 ne diyor?"}).status_code == 503


def test_explain_and_agents(client):
    e = client.post("/explain", json={"transaction_id": 3554906}).json()
    assert e["decision"] == "BLOCK" and e["explanation"]["citations"] == ["FP-01"] and e["agent_trace"]
    a = client.post("/agents/run", json={"goal": "Sadece risk skorunu ver", "transaction": TX}).json()
    assert a["status"] == "ok" and a["plan"]["tasks"] == ["score"]


def test_missing_artifacts_return_503():
    def broken():
        raise FileNotFoundError("artifacts/aggregator.joblib")

    c = Container()
    c.scoring_service.override(providers.Callable(broken))
    c.transactions.override(providers.Object(StubRepo()))
    with TestClient(create_app(c, warm=False)) as client:
        r = client.post("/score", json={"transaction": TX})
    assert r.status_code == 503 and "train.py" in r.json()["detail"]


# ---------------------------------------------------------------- gerçek artifact'larla entegrasyon

@pytest.fixture(scope="module")
def real_client(settings):
    needed = [settings.aggregator_path, settings.profile_store_path, settings.feature_pipeline_path, settings.merged_path]
    if not all(p.exists() for p in needed):
        pytest.skip("artifact'lar yok (scripts/prepare_data.py ve scripts/train.py çalıştırılmalı)")
    c = Container()
    c.llm.override(providers.Object(StubLLM()))
    with TestClient(create_app(c, warm=False)) as client:
        yield client


def test_integration_score_and_rules(real_client):
    s = real_client.post("/score", json={"transaction_id": 3554906}).json()
    assert s["risk_percentile"] > 0.999 and set(s["layer_scores"]) == {"column", "multivariate", "entity", "temporal"}
    partial = real_client.post("/score", json={"transaction": TX}).json()        # kısmi JSON da skorlanır
    assert 0 <= partial["risk_percentile"] <= 1
    r = real_client.post("/rules/evaluate", json={"transaction_id": 3554906}).json()
    assert r["final_decision"] == "BLOCK"
