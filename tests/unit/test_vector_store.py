from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from rag.exceptions import VectorStoreError
from rag.models import Chunk
from rag.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreFactory,
    _point_id,
)


def make_chunk(text: str, index: int = 0, document_id: object = None) -> Chunk:
    doc_id = document_id or uuid4()
    return Chunk(id=f"{doc_id}:{index}", document_id=doc_id, text=text, index=index)


async def test_in_memory_upsert_and_search_ranking() -> None:
    store = InMemoryVectorStore()
    await store.ensure_ready()
    doc_id = uuid4()
    chunks = [
        make_chunk("about cats", 0, doc_id),
        make_chunk("about finance", 1, doc_id),
    ]
    await store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = await store.search([0.9, 0.1], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.text == "about cats"
    assert results[0].score > results[1].score
    assert results[0].origin == "dense"


async def test_in_memory_top_k_limits() -> None:
    store = InMemoryVectorStore()
    doc_id = uuid4()
    chunks = [make_chunk(f"t{i}", i, doc_id) for i in range(5)]
    await store.upsert(chunks, [[1.0, float(i)] for i in range(5)])
    results = await store.search([1.0, 0.0], top_k=3)
    assert len(results) == 3


async def test_in_memory_delete_document() -> None:
    store = InMemoryVectorStore()
    keep_id, drop_id = uuid4(), uuid4()
    await store.upsert(
        [make_chunk("keep", 0, keep_id), make_chunk("drop", 0, drop_id)],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    await store.delete_document(drop_id)
    results = await store.search([1.0, 0.0], top_k=10)
    assert [r.chunk.text for r in results] == ["keep"]


async def test_in_memory_length_mismatch() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(VectorStoreError, match="length mismatch"):
        await store.upsert([make_chunk("a")], [])


async def test_in_memory_zero_query_rejected() -> None:
    store = InMemoryVectorStore()
    await store.upsert([make_chunk("a")], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError, match="zero norm"):
        await store.search([0.0, 0.0], top_k=1)


async def test_qdrant_ensure_ready_creates_missing_collection() -> None:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    store = QdrantVectorStore(client, collection="docs", vector_size=4)
    await store.ensure_ready()
    client.create_collection.assert_awaited_once()
    kwargs = client.create_collection.await_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["vectors_config"].size == 4


async def test_qdrant_upsert_maps_payload() -> None:
    client = MagicMock()
    client.upsert = AsyncMock()
    store = QdrantVectorStore(client, collection="docs", vector_size=2)
    chunk = make_chunk("hello world")
    await store.upsert([chunk], [[0.1, 0.2]])
    points = client.upsert.await_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["chunk_id"] == chunk.id
    assert points[0].payload["text"] == "hello world"
    assert points[0].id == _point_id(chunk.id)


async def test_qdrant_search_restores_chunks() -> None:
    doc_id = uuid4()
    point = MagicMock()
    point.score = 0.87
    point.payload = {
        "chunk_id": f"{doc_id}:0",
        "document_id": str(doc_id),
        "text": "restored",
        "index": 0,
        "metadata": {"source": "a.pdf"},
    }
    response = MagicMock()
    response.points = [point]
    client = MagicMock()
    client.query_points = AsyncMock(return_value=response)
    store = QdrantVectorStore(client, collection="docs", vector_size=2)
    results = await store.search([0.1, 0.2], top_k=5)
    assert len(results) == 1
    assert results[0].chunk.text == "restored"
    assert results[0].chunk.document_id == doc_id
    assert results[0].score == pytest.approx(0.87)


async def test_qdrant_delete_document_filters_by_id() -> None:
    client = MagicMock()
    client.delete = AsyncMock()
    store = QdrantVectorStore(client, collection="docs", vector_size=2)
    doc_id = uuid4()
    await store.delete_document(doc_id)
    selector = client.delete.await_args.kwargs["points_selector"]
    assert selector.filter.must[0].match.value == str(doc_id)


def test_factory_creates_registered_backends() -> None:
    assert isinstance(VectorStoreFactory.create("memory"), InMemoryVectorStore)
    with pytest.raises(VectorStoreError, match="unknown vector store backend"):
        VectorStoreFactory.create("pinecone")
