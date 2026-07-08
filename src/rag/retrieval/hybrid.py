from collections.abc import Sequence

from rag.config import RetrievalConfig
from rag.ingestion.embedders import Embedder
from rag.models import Chunk, ScoredChunk
from rag.retrieval.reranker import Reranker
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import VectorStore


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    fused_scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result.chunk.id
            chunks[chunk_id] = result.chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredChunk(chunk=chunks[chunk_id], score=score, origin="fused")
        for chunk_id, score in ordered
    ]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._embedder = embedder
        self._config = config
        self._reranker = reranker

    async def retrieve(self, query: str) -> list[ScoredChunk]:
        vectors = await self._embedder.embed([query])
        dense = await self._vector_store.search(vectors[0], self._config.dense_top_k)
        sparse = self._bm25_index.search(query, self._config.sparse_top_k)
        fused = reciprocal_rank_fusion([dense, sparse], k=self._config.rrf_k)
        if self._reranker is None:
            return fused[: self._config.final_top_k]
        candidates = fused[: self._config.dense_top_k]
        return await self._reranker.rerank(query, candidates, self._config.final_top_k)
