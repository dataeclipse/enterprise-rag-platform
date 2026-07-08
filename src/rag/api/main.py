from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag import __version__
from rag.api.container import Container, build_container
from rag.api.middleware import MetricsMiddleware, RequestContextMiddleware
from rag.api.routes import documents, health, query
from rag.config import Settings, get_settings
from rag.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or get_settings()
        configure_logging(level=resolved_settings.log_level, json_output=resolved_settings.log_json)
        active = container or await build_container(resolved_settings)
        app.state.container = active
        logger.info("app_started", env=resolved_settings.env)
        yield
        await active.aclose()
        logger.info("app_stopped")

    app = FastAPI(
        title="enterprise-rag-platform",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    return app


app = create_app()
