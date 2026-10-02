"""Agent sistemini kurar: servisler -> agent'lar -> bus kaydı. (Adım 10'da DI container bu işi devralır.)"""
from __future__ import annotations

import json

from fraud_platform.agents.bus import MessageBus
from fraud_platform.agents.data_agent import DataAgent
from fraud_platform.agents.feature_agent import FeatureAgent
from fraud_platform.agents.investigator_agent import InvestigatorAgent
from fraud_platform.agents.orchestrator import Orchestrator
from fraud_platform.agents.planner import TaskPlanner
from fraud_platform.agents.rule_agent import RuleAgent
from fraud_platform.agents.scoring_agent import ScoringAgent
from fraud_platform.config import Settings
from fraud_platform.data.schema import Schema
from fraud_platform.llm.base import BaseLLM
from fraud_platform.rag.pipeline import RAGPipeline
from fraud_platform.rules.engine import RuleEngine
from fraud_platform.services.scoring_service import ScoringService


def build_agent_system(settings: Settings, scoring: ScoringService, rules: RuleEngine, rag: RAGPipeline,
                       llm: BaseLLM | None) -> Orchestrator:
    d = settings.data
    critical = [d.time_column, d.amount_column, settings.entity.uid_columns[0]]
    schema = Schema.load(settings.schema_path)
    quality = json.loads(settings.quality_report_path.read_text(encoding="utf-8"))
    pipeline_fields = [c for c in scoring.online.pipeline.required_columns() if c != d.id_column]

    bus = MessageBus()
    a = settings.agents
    orchestrator = Orchestrator(TaskPlanner(llm, a.planner_mode), tuple(a.investigate_on))
    for agent in (
        orchestrator,
        DataAgent(schema, quality, critical, pipeline_fields),
        FeatureAgent(scoring.online),
        ScoringAgent(scoring),
        RuleAgent(rules),
        InvestigatorAgent(rag),
    ):
        bus.register(agent)
    return orchestrator
