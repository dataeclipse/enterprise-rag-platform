from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from rag.agents.graph import build_agent_graph
from rag.api.service import QueryService
from rag.config import Settings
from rag.guardrails.injection import InjectionDetector
from rag.guardrails.pii import PIIRedactor
from rag.ingestion.chunkers import ChunkerFactory
from rag.ingestion.embedders import BytesCache, EmbedderFactory
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.storage import DocumentRepository, create_schema
from rag.llm.factory import build_llm_provider
from rag.observability.metrics import Metrics
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker, NoopReranker, Reranker
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import QdrantVectorStore


class RedisBytesCache:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        return cast(list[bytes | None], await self._client.mget(list(keys)))

    async def setex(self, name: str, time: int, value: bytes) -> Any:
        return await self._client.setex(name, time, value)


@dataclass
class Container:
    settings: Settings
    metrics: Metrics
    query_service: QueryService
    pipeline: IngestionPipeline
    redactor: PIIRedactor
    injection: InjectionDetector
    closers: list[Any] = field(default_factory=list)

    async def aclose(self) -> None:
        for closer in self.closers:
            await closer()


async def build_container(settings: Settings) -> Container:
    metrics = Metrics()
    engine: AsyncEngine = create_async_engine(
        settings.postgres.dsn.get_secret_value(), pool_size=settings.postgres.pool_size
    )
    await create_schema(engine)
    repository = DocumentRepository(async_sessionmaker(engine, expire_on_commit=False))

    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    vector_store = QdrantVectorStore(
        qdrant, collection=settings.qdrant.collection, vector_size=settings.qdrant.vector_size
    )
    redis_client = aioredis.from_url(settings.redis.url.get_secret_value())
    cache: BytesCache = RedisBytesCache(redis_client)
    embedder = EmbedderFactory.from_config(
        settings.embeddings, redis_config=settings.redis, cache=cache
    )
    bm25 = BM25Index()
    reranker: Reranker = (
        CrossEncoderReranker(settings.reranker.model)
        if settings.reranker.enabled
        else NoopReranker()
    )
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25,
        embedder=embedder,
        config=settings.retrieval,
        reranker=reranker,
    )
    llm = build_llm_provider(settings.llm)
    graph = build_agent_graph(llm, retriever, settings.agents)
    pipeline = IngestionPipeline(
        repository=repository,
        chunker=ChunkerFactory.from_config(settings.chunking),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        metrics=metrics,
    )
    await pipeline.warm_up()
    return Container(
        settings=settings,
        metrics=metrics,
        query_service=QueryService(graph, metrics),
        pipeline=pipeline,
        redactor=PIIRedactor(),
        injection=InjectionDetector(threshold=settings.guardrails.injection_threshold),
        closers=[engine.dispose, qdrant.close],
    )
