from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rag.ingestion.storage import DocumentRepository, create_schema
from rag.models import Chunk, DocumentFormat, DocumentMeta, DocumentStatus


@pytest.fixture
async def repository() -> AsyncIterator[DocumentRepository]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    yield DocumentRepository(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


def make_meta(source: str = "a.pdf", version: int = 1, content_hash: str = "h1") -> DocumentMeta:
    return DocumentMeta(
        source=source,
        format=DocumentFormat.PDF,
        content_hash=content_hash,
        version=version,
        status=DocumentStatus.INDEXED,
    )


async def test_save_and_get_by_hash(repository: DocumentRepository) -> None:
    meta = make_meta()
    await repository.save_document(meta, signature=(1, 2, 3))
    loaded = await repository.get_by_content_hash("h1")
    assert loaded is not None
    assert loaded.id == meta.id
    assert loaded.source == "a.pdf"
    assert await repository.get_by_content_hash("missing") is None


async def test_latest_for_source_skips_superseded(repository: DocumentRepository) -> None:
    old = make_meta(version=1, content_hash="h1")
    new = make_meta(version=2, content_hash="h2")
    await repository.save_document(old)
    await repository.save_document(new)
    await repository.update_status(old.id, DocumentStatus.SUPERSEDED)
    latest = await repository.latest_for_source("a.pdf")
    assert latest is not None
    assert latest.version == 2


async def test_chunks_roundtrip(repository: DocumentRepository) -> None:
    meta = make_meta()
    await repository.save_document(meta)
    chunks = [
        Chunk(
            id=f"{meta.id}:{i}",
            document_id=meta.id,
            text=f"chunk {i}",
            index=i,
            metadata={"source": "a.pdf"},
        )
        for i in range(3)
    ]
    await repository.add_chunks(chunks)
    loaded = await repository.all_chunks()
    assert len(loaded) == 3
    assert loaded[0].metadata == {"source": "a.pdf"}
    await repository.delete_chunks(meta.id)
    assert await repository.all_chunks() == []


async def test_signatures_exclude_superseded(repository: DocumentRepository) -> None:
    active = make_meta(content_hash="h1")
    stale = make_meta(content_hash="h2", version=2)
    await repository.save_document(active, signature=(10, 20))
    await repository.save_document(stale, signature=(30, 40))
    await repository.update_status(stale.id, DocumentStatus.SUPERSEDED)
    signatures = await repository.all_signatures()
    assert signatures == [(str(active.id), "h1", (10, 20))]


async def test_list_documents(repository: DocumentRepository) -> None:
    await repository.save_document(make_meta(content_hash="h1"))
    await repository.save_document(make_meta(source="b.pdf", content_hash="h2"))
    documents = await repository.list_documents()
    assert {doc.source for doc in documents} == {"a.pdf", "b.pdf"}


async def test_get_by_id_missing(repository: DocumentRepository) -> None:
    assert await repository.get_by_id(uuid4()) is None
