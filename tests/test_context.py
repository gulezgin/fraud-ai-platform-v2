import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.context.calibration import conditional_lift


def write_rules(path, rules, lo=0.5, hi=2.0):
    path.write_text(yaml.safe_dump({"min_total_factor": lo, "max_total_factor": hi, "rules": rules}), encoding="utf-8")
    return path


def rule(rid, field, factor, enabled=True):
    return {"id": rid, "name": rid, "category": "test", "when": {"field": field, "op": "eq", "value": 1},
            "factor": factor, "reason": "test", "enabled": enabled}


@pytest.fixture
def df():
    return pd.DataFrame({"night": [1, 1, 0, 0], "trusted": [0, 1, 1, 0], "foreign": [1, 0, 0, 0]})


def test_factors_multiply_and_clamp(tmp_path, df):
    path = write_rules(tmp_path / "r.yaml", [rule("N", "night", 1.5), rule("T", "trusted", 0.6), rule("F", "foreign", 1.5)])
    out = ContextAdjuster(path).adjust(df, pd.Series([0.4, 0.4, 0.4, 0.4]))

    # 1.5 * 1.5 = 2.25 -> 2.0 ile sınırlanır; 1.5 * 0.6 = 0.9; 0.6; hiç kural yok = 1
    assert out["context_factor"].round(4).tolist() == [2.0, 0.9, 0.6, 1.0]
    assert out["adjusted_score"].round(4).tolist() == [0.8, 0.36, 0.24, 0.4]


def test_adjusted_score_is_capped_at_one(tmp_path, df):
    path = write_rules(tmp_path / "r.yaml", [rule("N", "night", 2.0)])
    out = ContextAdjuster(path).adjust(df, pd.Series([0.9, 0.9, 0.9, 0.9]))
    assert out["adjusted_score"].max() == 1.0


def test_disabled_rule_has_no_effect(tmp_path, df):
    path = write_rules(tmp_path / "r.yaml", [rule("N", "night", 1.5, enabled=False)])
    adj = ContextAdjuster(path)
    assert adj.active_rules == []
    assert (adj.adjust(df, pd.Series([0.4] * 4))["context_factor"] == 1).all()


def test_explain_and_reload(tmp_path, df):
    path = write_rules(tmp_path / "r.yaml", [rule("N", "night", 1.5)])
    adj = ContextAdjuster(path)
    applied = adj.explain(df.iloc[0])
    assert [a["id"] for a in applied] == ["N"] and applied[0]["condition"] == "night = 1"

    write_rules(path, [rule("N", "night", 1.5), rule("T", "trusted", 0.6)])
    assert adj.reload() == 2   # kod değişmeden yeni kural devrede
    assert [a["id"] for a in adj.explain(df.iloc[1])] == ["N", "T"]


def test_invalid_ruleset_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="tekrarlanan"):
        ContextAdjuster(write_rules(tmp_path / "a.yaml", [rule("X", "night", 1.2), rule("X", "trusted", 0.8)]))
    with pytest.raises(ValidationError, match="1'i kapsamalı"):
        ContextAdjuster(write_rules(tmp_path / "b.yaml", [rule("X", "night", 1.2)], lo=1.1, hi=2.0))
    with pytest.raises(ValidationError):
        ContextAdjuster(write_rules(tmp_path / "c.yaml", [rule("X", "night", -1)]))


def test_conditional_lift_ignores_risk_already_in_score():
    # kural yüksek skorlu işlemlerde daha sık görülüyor (ham lift > 1), ama skor bilindiğinde etiketle ilişkisi yok
    # -> koşullu lift ~1: çarpan, skorun zaten gördüğü riski ikinci kez saymamalı
    rng = np.random.default_rng(0)
    n = 200_000
    score = pd.Series(rng.random(n))
    y = pd.Series((rng.random(n) < score**4).astype(int))
    mask = pd.Series(rng.random(n) < score)

    raw_lift = y[mask].mean() / y.mean()
    assert raw_lift > 1.3
    assert conditional_lift(mask, score, y, region=1.0, bins=50) == pytest.approx(1.0, abs=0.05)


def test_conditional_lift_detects_risk_missing_from_score():
    rng = np.random.default_rng(1)
    n = 200_000
    score = pd.Series(rng.random(n))
    mask = pd.Series(rng.random(n) < 0.1)
    p = 0.3 * score**4 * np.where(mask, 2.0, 1.0)   # kural riski skordan bağımsız 2 katına çıkarıyor (p <= 0,6)
    y = pd.Series((rng.random(n) < p).astype(int))
    # beklenen oran kuralın kendi işlemlerini de içerir: dilim ortalaması 0,9 + 0,1 x 2 = 1,1 -> 2 / 1,1 = 1,82
    # (yöntem gerçek etkiyi biraz küçümser; çarpanlar temkinli tarafta kalır)
    assert conditional_lift(mask, score, y, region=1.0, bins=50) == pytest.approx(2 / 1.1, rel=0.05)


def test_project_context_rules_are_valid(settings):
    adj = ContextAdjuster(settings.context.rules_file)
    assert len(adj.active_rules) >= 8
    assert {"zaman", "entity", "islem_tipi", "cografi"} <= {r.category for r in adj.active_rules}
