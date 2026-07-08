import asyncio

from rag.api.container import build_container
from rag.config import get_settings
from rag.ingestion.queue import IngestionConsumer
from rag.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def run() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    container = await build_container(settings)
    consumer = IngestionConsumer(settings.rabbitmq, container.pipeline)
    logger.info("worker_started", queue=settings.rabbitmq.ingestion_queue)
    try:
        await consumer.run()
    finally:
        await container.aclose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
