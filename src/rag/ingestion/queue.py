import base64
from collections.abc import Awaitable, Callable

import aio_pika
from pydantic import BaseModel, ValidationError

from rag.config import RabbitMQConfig
from rag.ingestion.pipeline import IngestionPipeline
from rag.observability.logging import get_logger

logger = get_logger(__name__)


class IngestionJob(BaseModel):
    filename: str
    source: str | None = None
    payload_b64: str

    @classmethod
    def from_bytes(cls, filename: str, data: bytes, source: str | None = None) -> "IngestionJob":
        return cls(
            filename=filename,
            source=source,
            payload_b64=base64.b64encode(data).decode("ascii"),
        )

    def payload(self) -> bytes:
        return base64.b64decode(self.payload_b64)


class IngestionPublisher:
    def __init__(self, config: RabbitMQConfig) -> None:
        self._config = config
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._config.url.get_secret_value())

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def publish(self, job: IngestionJob) -> None:
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        channel = await self._connection.channel()
        await channel.declare_queue(self._config.ingestion_queue, durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=job.model_dump_json().encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            ),
            routing_key=self._config.ingestion_queue,
        )
        logger.info("job_published", filename=job.filename)


JobHandler = Callable[[IngestionJob], Awaitable[None]]


async def handle_message(
    message: aio_pika.abc.AbstractIncomingMessage, handler: JobHandler
) -> None:
    try:
        job = IngestionJob.model_validate_json(message.body)
    except ValidationError:
        logger.error("job_invalid", body_size=len(message.body))
        await message.reject(requeue=False)
        return
    try:
        await handler(job)
    except Exception:
        logger.exception("job_failed", filename=job.filename)
        await message.reject(requeue=False)
        return
    await message.ack()


class IngestionConsumer:
    def __init__(self, config: RabbitMQConfig, pipeline: IngestionPipeline) -> None:
        self._config = config
        self._pipeline = pipeline

    async def _handle(self, job: IngestionJob) -> None:
        await self._pipeline.ingest(job.filename, job.payload(), source=job.source)

    async def run(self) -> None:
        connection = await aio_pika.connect_robust(self._config.url.get_secret_value())
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=self._config.prefetch_count)
            queue = await channel.declare_queue(self._config.ingestion_queue, durable=True)
            logger.info("consumer_started", queue=self._config.ingestion_queue)
            async with queue.iterator() as messages:
                async for message in messages:
                    await handle_message(message, self._handle)
