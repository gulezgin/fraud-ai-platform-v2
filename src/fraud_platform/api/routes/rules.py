from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from fraud_platform.api.deps import resolve_transaction
from fraud_platform.api.schemas import (
    RulesEvaluateRequest,
    RulesEvaluateResponse,
    RulesReloadResponse,
)
from fraud_platform.conditions.evaluator import describe
from fraud_platform.container import Container
from fraud_platform.context.adjuster import ContextAdjuster
from fraud_platform.rules.engine import RuleEngine
from fraud_platform.services.scoring_service import ScoringService
from fraud_platform.services.transaction_repository import TransactionRepository

router = APIRouter(prefix="/rules", tags=["kurallar"])


@router.get("", summary="Yüklü kural seti")
@inject
def list_rules(engine: RuleEngine = Depends(Provide[Container.rule_engine])) -> dict:
    rs = engine.ruleset
    return {"conflict_strategy": rs.conflict_strategy, "default_action": rs.default_action, "rules": [
        {"id": r.id, "name": r.name, "priority": r.priority, "action": r.action, "condition": describe(r.condition),
         "policy_ref": r.policy_ref, "enabled": r.enabled} for r in rs.rules]}


@router.post("/evaluate", response_model=RulesEvaluateResponse,
             summary="Kuralları değerlendirir: karar, tetiklenen kurallar, çakışma çözümü, açıklama")
@inject
def evaluate(req: RulesEvaluateRequest,
             engine: RuleEngine = Depends(Provide[Container.rule_engine]),
             service: ScoringService = Depends(Provide[Container.scoring_service]),
             repo: TransactionRepository = Depends(Provide[Container.transactions])) -> dict:
    if req.fields is not None:
        return engine.evaluate(req.fields, strategy=req.strategy)
    tx = resolve_transaction(req.transaction, req.transaction_id, repo)
    return engine.evaluate(service.assessment_frame([tx]).iloc[0], strategy=req.strategy)


@router.post("/reload", response_model=RulesReloadResponse,
             summary="rules.yaml ve context_rules.yaml'ı kod değişmeden yeniden yükler")
@inject
def reload(engine: RuleEngine = Depends(Provide[Container.rule_engine]),
           context: ContextAdjuster = Depends(Provide[Container.context])) -> dict:
    return {"rules": engine.reload(), "context_rules": context.reload()}
