import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.observability.metrics import Metrics


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id", "path")
        response.headers["x-request-id"] = request_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        container = getattr(request.app.state, "container", None)
        metrics: Metrics | None = container.metrics if container is not None else None
        started = time.perf_counter()
        response = await call_next(request)
        if metrics is not None:
            endpoint = request.url.path
            metrics.requests_total.labels(endpoint=endpoint, status=str(response.status_code)).inc()
            metrics.request_duration_seconds.labels(endpoint=endpoint).observe(
                time.perf_counter() - started
            )
        return response
