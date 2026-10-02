"""Görev planlayıcı değerlendirmesi: hybrid / llm / rules modları (16 istek, 4'ü holdout; tam plan eşleşmesi).

    python scripts/eval_planner.py

Planlayıcı sadece agent yetenek beyanlarını kullanır; skorlama servisleri yüklenmez.
"""
import json
import logging

import pandas as pd
import yaml

from fraud_platform.agents.data_agent import DataAgent
from fraud_platform.agents.feature_agent import FeatureAgent
from fraud_platform.agents.investigator_agent import InvestigatorAgent
from fraud_platform.agents.planner import TaskPlanner
from fraud_platform.agents.rule_agent import RuleAgent
from fraud_platform.agents.scoring_agent import ScoringAgent
from fraud_platform.config import PROJECT_ROOT, get_settings
from fraud_platform.rag.factory import build_llm


def capabilities() -> dict:
    agents = [DataAgent(None, {}, [], []), FeatureAgent(None), ScoringAgent(None), RuleAgent(None), InvestigatorAgent(None)]
    return {t: c for a in agents for t, c in a.capabilities().items()}


def main() -> None:
    logging.disable(logging.INFO)
    s = get_settings()
    goals = yaml.safe_load((PROJECT_ROOT / "knowledge_base" / "eval" / "planner_goals.yaml").read_text(encoding="utf-8"))
    caps = capabilities()
    llm = build_llm(s)
    planners = {mode: TaskPlanner(llm, mode) for mode in TaskPlanner.MODES}

    rows = []
    for g in goals:
        row = {"goal": g["goal"], "işlem": g["transaction"], "holdout": g.get("holdout", False), "beklenen": g["expected"][-1]}
        for mode, planner in planners.items():
            p = planner.plan(g["goal"], caps, g["transaction"])
            row[mode] = p.tasks[-1] if p.tasks else None
            row[f"{mode}_ok"] = p.tasks == g["expected"]
            row[f"{mode}_kaynak"] = p.source
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 55)
    print(df[["goal", "holdout", "beklenen", "llm", "rules", "hybrid", "hybrid_kaynak"]].to_string(index=False))
    summary = {
        f"{mode}_{part}": float(df.loc[mask, f"{mode}_ok"].mean())
        for mode in TaskPlanner.MODES
        for part, mask in [("ayarlanan_12", ~df["holdout"]), ("holdout_4", df["holdout"]), ("tümü", df["holdout"] | ~df["holdout"])]
    }
    print("\n", pd.Series(summary).round(3).to_string())
    out = s.rag.index_dir.parent / "planner_eval.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
