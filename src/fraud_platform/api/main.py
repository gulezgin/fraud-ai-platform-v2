"""FastAPI uygulaması.

    uvicorn fraud_platform.api.main:app            # http://localhost:8000/docs (Swagger)

Bağımlılıklar DI container'dan gelir; ağır bileşenler açılışta bir kez yüklenir (warm_up). Bir bileşen yüklenemezse
(ör. Ollama kapalı, index yok) uygulama yine açılır: ilgili endpoint 503 döner, /health durumu gösterir.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraud_platform.api.routes import agents, explain, health, rag, rules, scoring
from fraud_platform.container import Container, warm_up
from fraud_platform.llm.base import LLMError
from fraud_platform.services.transaction_repository import TransactionNotFound

DESCRIPTION = """
IEEE-CIS verisi üzerinde çok katmanlı anomali tespiti, context düzeltmesi, YAML kural motoru, RAG ve multi-agent orkestrasyonu.

İşlem iki yoldan verilir: `transaction` (JSON; verilmeyen alanlar boş sayılır) veya `transaction_id` (kayıtlı işlemin tüm 435 alanı
veri setinden okunur). Skorlamada sadece işlemden önceki geçmiş kullanılır (profil deposu).
"""


def create_app(container: Container | None = None, warm: bool = True) -> FastAPI:
    container = container or Container()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.components = warm_up(container) if warm else {}
        yield

    app = FastAPI(title="Fraud AI Platform", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
    app.state.container = container
    container.wire(packages=["fraud_platform.api.routes"])

    for module in (health, scoring, explain, rules, rag, agents):
        app.include_router(module.router)

    @app.exception_handler(TransactionNotFound)
    async def _not_found(request: Request, exc: TransactionNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc.args[0])})

    @app.exception_handler(LLMError)
    async def _llm_down(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"LLM kullanılamıyor: {exc}"})

    @app.exception_handler(FileNotFoundError)
    async def _artifact_missing(request: Request, exc: FileNotFoundError) -> JSONResponse:
        hint = "önce scripts/prepare_data.py, scripts/train.py ve scripts/build_index.py çalıştırılmalı"
        return JSONResponse(status_code=503, content={"detail": f"gerekli dosya yok ({exc}); {hint}"})

    return app


app = create_app()
