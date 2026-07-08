import socket
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from rag.models import Chunk
from rag.retrieval.vector_store import QdrantVectorStore

pytestmark = pytest.mark.integration


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


requires_qdrant = pytest.mark.skipif(
    not _reachable("localhost", 6333), reason="qdrant is not running on localhost:6333"
)


@requires_qdrant
async def test_qdrant_roundtrip() -> None:
    client = AsyncQdrantClient(url="http://localhost:6333")
    collection = f"it_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(client, collection=collection, vector_size=4)
    try:
        await store.ensure_ready()
        document_id = uuid.uuid4()
        chunks = [
            Chunk(id=f"{document_id}:0", document_id=document_id, text="alpha", index=0),
            Chunk(id=f"{document_id}:1", document_id=document_id, text="beta", index=1),
        ]
        await store.upsert(chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].chunk.text == "alpha"
        await store.delete_document(document_id)
    finally:
        await client.delete_collection(collection)
        await client.close()
