from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from fraud_platform.agents.orchestrator import Orchestrator
from fraud_platform.api.deps import timer
from fraud_platform.api.schemas import AgentRunRequest
from fraud_platform.container import Container
from fraud_platform.services.transaction_repository import TransactionRepository

router = APIRouter(prefix="/agents", tags=["agent'lar"])


@router.post("/run", summary="Doğal dildeki isteği multi-agent sisteme verir: plan, adımlar, sonuç ve mesaj izi")
@inject
def run(req: AgentRunRequest,
        orchestrator: Orchestrator = Depends(Provide[Container.orchestrator]),
        repo: TransactionRepository = Depends(Provide[Container.transactions])) -> dict:
    tx = req.transaction.to_record() if req.transaction else (repo.get(req.transaction_id) if req.transaction_id else None)
    with timer() as t:
        result = orchestrator.run(req.goal, transaction=tx, strategy=req.strategy,
                                  force_investigation=req.force_investigation)
    return {**result, **t}
