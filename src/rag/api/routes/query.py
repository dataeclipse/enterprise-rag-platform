from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from rag.api.auth import get_current_subject
from rag.api.container import Container
from rag.api.schemas import AnswerResponse, QueryRequest
from rag.observability.logging import get_logger

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


def _container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def _apply_guardrails(container: Container, query: str) -> tuple[str, bool]:
    settings = container.settings.guardrails
    if settings.injection_detection:
        assessment = container.injection.scan(query)
        if assessment.blocked:
            container.metrics.guardrail_blocks_total.labels(guardrail="injection").inc()
            logger.warning("query_blocked", rules=assessment.matched_rules)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="query rejected by safety filters",
            )
    if settings.pii_redaction:
        redaction = container.redactor.redact(query)
        if redaction.matches:
            container.metrics.guardrail_blocks_total.labels(guardrail="pii").inc()
        return redaction.text, bool(redaction.matches)
    return query, False


@router.post("/query", response_model=AnswerResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    subject: str = Depends(get_current_subject),
) -> AnswerResponse:
    container = _container(request)
    safe_query, redacted = _apply_guardrails(container, payload.query)
    answer = await container.query_service.answer(safe_query)
    return AnswerResponse.from_answer(answer, pii_redacted=redacted)


@router.post("/query/stream")
async def query_stream(
    payload: QueryRequest,
    request: Request,
    subject: str = Depends(get_current_subject),
) -> EventSourceResponse:
    container = _container(request)
    safe_query, _ = _apply_guardrails(container, payload.query)

    async def event_source() -> AsyncIterator[dict[str, str]]:
        async for event in container.query_service.stream(safe_query):
            yield {"event": event.event, "data": event.data}

    return EventSourceResponse(event_source())
