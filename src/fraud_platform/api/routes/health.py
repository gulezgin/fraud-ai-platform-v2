from fastapi import APIRouter, Request

from fraud_platform.api.schemas import HealthResponse

router = APIRouter(tags=["sistem"])


@router.get("/health", response_model=HealthResponse, summary="Bileşenler yüklendi mi, LLM erişilebilir mi")
def health(request: Request) -> dict:
    container = request.app.state.container
    components = getattr(request.app.state, "components", {})

    def safe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - sağlık kontrolü hiçbir koşulda hata fırlatmamalı
            return None

    llm_ok = bool(safe(lambda: container.llm().health()))
    ok = all(v == "ok" for v in components.values()) and llm_ok
    return {
        "status": "ok" if ok else "degraded",
        "components": components,
        "llm_reachable": llm_ok,
        "rules": safe(lambda: len(container.rule_engine().rules)),
        "knowledge_base_chunks": safe(lambda: len(container.vector_store().chunks)),
    }
