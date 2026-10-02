from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from fraud_platform.api.schemas import RAGQueryRequest, RAGQueryResponse
from fraud_platform.container import Container
from fraud_platform.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=RAGQueryResponse,
             summary="Politika bilgi tabanında arama + yerel LLM cevabı + atıf doğrulama")
@inject
def query(req: RAGQueryRequest, rag: RAGPipeline = Depends(Provide[Container.rag])) -> dict:
    return rag.query(req.question, k=req.top_k)
