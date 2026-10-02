from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from fraud_platform.api.deps import resolve_transaction, timer
from fraud_platform.api.schemas import ExplainRequest, ExplainResponse
from fraud_platform.container import Container
from fraud_platform.services.explain_service import ExplainService
from fraud_platform.services.transaction_repository import TransactionRepository

router = APIRouter(tags=["açıklama"])


@router.post("/explain", response_model=ExplainResponse,
             summary="Skor + en etkili feature'lar + kurallar + RAG açıklaması + agent izi")
@inject
def explain(req: ExplainRequest,
            service: ExplainService = Depends(Provide[Container.explain_service]),
            repo: TransactionRepository = Depends(Provide[Container.transactions])) -> dict:
    tx = resolve_transaction(req.transaction, req.transaction_id, repo)
    with timer() as t:
        result = service.explain(tx, strategy=req.strategy)
    return {**result, **t}
