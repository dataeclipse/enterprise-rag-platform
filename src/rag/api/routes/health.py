from fastapi import APIRouter, Request, Response

from rag import __version__
from rag.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    payload = request.app.state.container.metrics.render()
    return Response(content=payload, media_type="text/plain; version=0.0.4")
