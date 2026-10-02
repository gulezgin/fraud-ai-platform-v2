import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from fraud_platform.rules.engine import RuleEngine, render


def rule(rid, field, action, priority, delta=0.0, value=1, explanation="{x} tetiklendi"):
    return {"id": rid, "name": rid, "priority": priority, "action": action, "risk_delta": delta,
            "condition": {"field": field, "op": "eq", "value": value}, "explanation": explanation}


RULES = [
    rule("WL", "trusted", "ALLOW", 60, -0.1),
    rule("SC", "high_score", "REVIEW", 70, 0.05),
    rule("FL", "device", "FLAG", 58, 0.1),
    rule("BK", "burst", "BLOCK", 95, 0.3),
]


def write(tmp_path, rules=RULES, strategy="priority_then_severity", fmt="yaml"):
    data = {"conflict_strategy": strategy, "default_action": "ALLOW", "rules": rules}
    path = tmp_path / f"rules.{fmt}"
    path.write_text(json.dumps(data) if fmt == "json" else yaml.safe_dump(data), encoding="utf-8")
    return path


def tx(**flags):
    base = {"trusted": 0, "high_score": 0, "device": 0, "burst": 0, "x": "x", "risk_percentile": 0.5}
    return {**base, **flags}


@pytest.mark.parametrize("flags,expected,winner", [
    ({}, "ALLOW", None),                                   # hiçbir kural yok -> varsayılan
    ({"trusted": 1, "high_score": 1}, "REVIEW", "SC"),     # beyaz liste (60) skoru (70) ezemez
    ({"trusted": 1, "device": 1}, "ALLOW", "WL"),          # beyaz liste FLAG'i (58) ezer
    ({"high_score": 1, "device": 1}, "REVIEW", "SC"),      # FLAG skorun REVIEW'unu ezemez
    ({"trusted": 1, "high_score": 1, "burst": 1}, "BLOCK", "BK"),
])
def test_priority_then_severity(tmp_path, flags, expected, winner):
    out = RuleEngine(write(tmp_path)).evaluate(tx(**flags))
    assert out["final_decision"] == expected and out["winning_rule"] == winner
    overridden = {r["id"] for r in out["fired_rules"] if "overridden_by" in r}
    assert overridden == {r["id"] for r in out["fired_rules"]} - ({winner} if winner else set())


def test_strategies_differ(tmp_path):
    rules = [rule("WL", "trusted", "ALLOW", 90), rule("SC", "high_score", "REVIEW", 70)]
    t = tx(trusted=1, high_score=1)
    assert RuleEngine(write(tmp_path, rules, "priority_then_severity")).evaluate(t)["final_decision"] == "ALLOW"
    assert RuleEngine(write(tmp_path, rules, "most_severe")).evaluate(t)["final_decision"] == "REVIEW"
    eng = RuleEngine(write(tmp_path, rules[::-1], "first_match"))
    assert eng.evaluate(t)["winning_rule"] == "SC"
    assert eng.evaluate(t, strategy="priority_then_severity")["winning_rule"] == "WL"   # çağrı başına strateji


@pytest.mark.parametrize("strategy", ["priority_then_severity", "most_severe", "first_match"])
def test_batch_matches_single_evaluation(tmp_path, strategy):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(0, 2, 300) for c in ["trusted", "high_score", "device", "burst"]})
    df["x"], df["risk_percentile"] = "x", rng.random(300)
    eng = RuleEngine(write(tmp_path, strategy=strategy))

    batch = eng.evaluate_batch(df)
    for i in range(len(df)):
        single = eng.evaluate(df.iloc[i])
        assert single["final_decision"] == batch["decision"].iloc[i]
        assert single["final_risk"] == pytest.approx(batch["final_risk"].iloc[i], abs=1e-4)


def test_final_risk_adds_deltas_and_clips(tmp_path):
    eng = RuleEngine(write(tmp_path))
    assert eng.evaluate(tx(burst=1, device=1, risk_percentile=0.5))["final_risk"] == pytest.approx(0.9)
    assert eng.evaluate(tx(burst=1, device=1, risk_percentile=0.95))["final_risk"] == 1.0
    assert eng.evaluate(tx(trusted=1, risk_percentile=0.05))["final_risk"] == 0.0


def test_yaml_and_json_give_same_result(tmp_path):
    t = tx(trusted=1, high_score=1, device=1)
    a = RuleEngine(write(tmp_path, fmt="yaml")).evaluate(t)
    b = RuleEngine(write(tmp_path, fmt="json")).evaluate(t)
    assert a == b


def test_explanation_and_matched_values(tmp_path):
    rules = [{**rule("A", "amt_flag", "REVIEW", 80), "explanation": "Tutar {amt:.2f} USD, saat {hour}"}]
    out = RuleEngine(write(tmp_path, rules)).evaluate({"amt_flag": 1, "amt": 1234.5, "hour": None})
    r = out["fired_rules"][0]
    assert r["explanation"] == "Tutar 1234.50 USD, saat ?"
    assert r["matched_values"] == {"amt_flag": 1}
    assert "kazanan kural A" in out["decision_reason"]


def test_render_handles_missing_and_bad_format():
    assert render("{a:.2f} / {b}", {"a": None}) == "? / ?"
    assert render("{a:.1f}", {"a": 2.345}) == "2.3"


def test_reload_picks_up_new_rules(tmp_path):
    path = write(tmp_path, RULES[:1])
    eng = RuleEngine(path)
    assert eng.evaluate(tx(burst=1))["final_decision"] == "ALLOW"
    write(tmp_path, RULES)
    assert eng.reload() == 4
    assert eng.evaluate(tx(burst=1))["final_decision"] == "BLOCK"


def test_disabled_rules_are_skipped_but_measured(tmp_path):
    rules = [{**rule("BK", "burst", "BLOCK", 95), "enabled": False}, rule("SC", "high_score", "REVIEW", 70)]
    eng = RuleEngine(write(tmp_path, rules))
    assert eng.evaluate(tx(burst=1))["final_decision"] == "ALLOW"
    df = pd.DataFrame([tx(burst=1), tx(high_score=1)]).assign(y=[1, 0])
    stats = eng.stats(df, "y")
    assert stats.loc["BK", "fired"] == 1 and stats.loc["BK", "won"] == 0


@pytest.mark.parametrize("bad", [
    {"conflict_strategy": "yok", "rules": RULES},
    {"conflict_strategy": "first_match", "rules": [RULES[0], RULES[0]]},
    {"conflict_strategy": "first_match", "rules": [{**RULES[0], "action": "DELETE"}]},
    {"conflict_strategy": "first_match", "rules": [{**RULES[0], "explanation": "{eksik"}]},
    {"conflict_strategy": "first_match", "rules": [{**RULES[0], "priority": -1}]},
])
def test_invalid_rulesets_are_rejected(tmp_path, bad):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises((ValidationError, ValueError)):
        RuleEngine(path)


def test_project_rules_file(settings):
    eng = RuleEngine(settings.rules.rules_file)
    assert len(eng.rules) >= 10
    assert {"ALLOW", "FLAG", "REVIEW", "BLOCK"} <= {r.action for r in eng.rules}
    assert all(r.policy_ref for r in eng.rules)
