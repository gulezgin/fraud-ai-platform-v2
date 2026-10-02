from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from fraud_platform.api.deps import resolve_transaction, timer
from fraud_platform.api.schemas import ScoreRequest, ScoreResponse
from fraud_platform.container import Container
from fraud_platform.services.scoring_service import ScoringService
from fraud_platform.services.transaction_repository import TransactionRepository

router = APIRouter(tags=["skorlama"])


@router.post("/score", response_model=ScoreResponse,
             summary="4 anomali katmanı + skor birleştirme + context düzeltmesi")
@inject
def score(req: ScoreRequest,
          service: ScoringService = Depends(Provide[Container.scoring_service]),
          repo: TransactionRepository = Depends(Provide[Container.transactions])) -> dict:
    tx = resolve_transaction(req.transaction, req.transaction_id, repo)
    with timer() as t:
        result = service.score([tx], explain=req.explain)[0]
    return {**result, **t}
