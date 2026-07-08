from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from rag.exceptions import VectorStoreError
from rag.models import Chunk, ScoredChunk


class VectorStore(ABC):
    @abstractmethod
    async def ensure_ready(self) -> None: ...

    @abstractmethod
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    async def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> None: ...


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(self, client: AsyncQdrantClient, collection: str, vector_size: int) -> None:
        self._client = client
        self._collection = collection
        self._vector_size = vector_size

    async def ensure_ready(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.id),
                vector=list(vector),
                payload={
                    "chunk_id": chunk.id,
                    "document_id": str(chunk.document_id),
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                id=str(payload["chunk_id"]),
                document_id=UUID(str(payload["document_id"])),
                text=str(payload["text"]),
                index=int(payload["index"]),
                metadata=dict(payload.get("metadata") or {}),
            )
            results.append(ScoredChunk(chunk=chunk, score=float(point.score), origin="dense"))
        return results

    async def delete_document(self, document_id: UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = [float(value) for value in vector]

    async def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        if not self._vectors:
            return []
        query = np.asarray(vector, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise VectorStoreError("query vector has zero norm")
        scored: list[ScoredChunk] = []
        for chunk_id, stored in self._vectors.items():
            candidate = np.asarray(stored, dtype=np.float64)
            denominator = float(np.linalg.norm(candidate) * query_norm)
            score = float(np.dot(candidate, query) / denominator) if denominator else 0.0
            scored.append(ScoredChunk(chunk=self._chunks[chunk_id], score=score, origin="dense"))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def delete_document(self, document_id: UUID) -> None:
        stale = [
            chunk_id for chunk_id, chunk in self._chunks.items() if chunk.document_id == document_id
        ]
        for chunk_id in stale:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]


class VectorStoreFactory:
    _builders: ClassVar[dict[str, Callable[..., VectorStore]]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[..., VectorStore]) -> None:
        cls._builders[name] = builder

    @classmethod
    def create(cls, name: str, **kwargs: object) -> VectorStore:
        builder = cls._builders.get(name)
        if builder is None:
            raise VectorStoreError(f"unknown vector store backend: {name}")
        return builder(**kwargs)


VectorStoreFactory.register("qdrant", QdrantVectorStore)
VectorStoreFactory.register("memory", InMemoryVectorStore)
