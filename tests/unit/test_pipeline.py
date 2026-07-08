from collections.abc import AsyncIterator, Sequence

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rag.exceptions import LoaderError
from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.embedders import Embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.storage import DocumentRepository, create_schema
from rag.models import DocumentStatus
from rag.observability.metrics import Metrics
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import InMemoryVectorStore

DOCUMENT_TEXT = (
    "The vacation policy grants twenty five days of paid leave per year. "
    "Employees may carry over up to five unused days into the next year. "
    "Approval from a direct manager is required for absences longer than "
    "two consecutive weeks. Sick leave is tracked separately from vacation."
)

REVISED_TEXT = DOCUMENT_TEXT.replace("twenty five", "thirty")

UNRELATED_TEXT = (
    "Network security requires multi factor authentication for all remote "
    "access. VPN certificates rotate every ninety days without exception. "
    "Incident response procedures are documented in the security handbook."
)


class HashEmbedder(Embedder):
    @property
    def model_id(self) -> str:
        return "hash"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0 + (sum(map(ord, text)) % 97) / 97.0, float(len(text) % 31)] for text in texts]


@pytest.fixture
async def pipeline() -> AsyncIterator[IngestionPipeline]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    repository = DocumentRepository(async_sessionmaker(engine, expire_on_commit=False))
    instance = IngestionPipeline(
        repository=repository,
        chunker=RecursiveChunker(chunk_size=120, chunk_overlap=20),
        embedder=HashEmbedder(),
        vector_store=InMemoryVectorStore(),
        bm25_index=BM25Index(),
        dedup_threshold=0.8,
        metrics=Metrics(),
    )
    await instance.warm_up()
    yield instance
    await engine.dispose()


async def test_ingest_indexes_document(pipeline: IngestionPipeline) -> None:
    meta = await pipeline.ingest("policy.txt", DOCUMENT_TEXT.encode())
    assert meta.status is DocumentStatus.INDEXED
    assert meta.version == 1
    stored = await pipeline._repository.get_by_id(meta.id)
    assert stored is not None
    assert stored.status is DocumentStatus.INDEXED
    chunks = await pipeline._repository.all_chunks()
    assert chunks
    assert all(chunk.metadata["source"] == "policy.txt" for chunk in chunks)
    assert len(pipeline._bm25_index) == len(chunks)


async def test_ingest_exact_duplicate_returns_existing(pipeline: IngestionPipeline) -> None:
    first = await pipeline.ingest("policy.txt", DOCUMENT_TEXT.encode())
    second = await pipeline.ingest("copy.txt", DOCUMENT_TEXT.encode())
    assert second.id == first.id
    assert len(await pipeline._repository.list_documents()) == 1


async def test_ingest_near_duplicate_skipped(pipeline: IngestionPipeline) -> None:
    first = await pipeline.ingest("policy.txt", DOCUMENT_TEXT.encode())
    near = DOCUMENT_TEXT.replace("direct manager", "line manager")
    second = await pipeline.ingest("other.txt", near.encode())
    assert second.id == first.id


async def test_ingest_new_version_supersedes(pipeline: IngestionPipeline) -> None:
    first = await pipeline.ingest("policy.txt", DOCUMENT_TEXT.encode())
    second = await pipeline.ingest("policy.txt", REVISED_TEXT.encode())
    assert second.version == 2
    old = await pipeline._repository.get_by_id(first.id)
    assert old is not None
    assert old.status is DocumentStatus.SUPERSEDED
    chunks = await pipeline._repository.all_chunks()
    assert all(chunk.document_id == second.id for chunk in chunks)
    results = await pipeline._vector_store.search([1.5, 10.0], top_k=50)
    assert all(item.chunk.document_id == second.id for item in results)


async def test_ingest_unrelated_documents_coexist(pipeline: IngestionPipeline) -> None:
    first = await pipeline.ingest("policy.txt", DOCUMENT_TEXT.encode())
    second = await pipeline.ingest("security.txt", UNRELATED_TEXT.encode())
    assert first.id != second.id
    assert len(await pipeline._repository.list_documents()) == 2


async def test_ingest_unsupported_extension(pipeline: IngestionPipeline) -> None:
    with pytest.raises(LoaderError, match="unsupported file extension"):
        await pipeline.ingest("data.xlsx", b"payload")


async def test_ingest_empty_document_fails_cleanly(pipeline: IngestionPipeline) -> None:
    with pytest.raises(LoaderError, match="no text content"):
        await pipeline.ingest("empty.txt", b"   ")
    assert await pipeline._repository.list_documents() == []
