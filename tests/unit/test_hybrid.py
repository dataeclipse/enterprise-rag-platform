from collections.abc import Sequence
from uuid import uuid4

import pytest

from rag.config import RetrievalConfig
from rag.ingestion.embedders import Embedder
from rag.models import Chunk, ScoredChunk
from rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.retrieval.reranker import NoopReranker, Reranker
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import InMemoryVectorStore

DOC_ID = uuid4()


def make_chunk(text: str, index: int) -> Chunk:
    return Chunk(id=f"{DOC_ID}:{index}", document_id=DOC_ID, text=text, index=index)


def scored(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, origin="dense")


class KeywordEmbedder(Embedder):
    def __init__(self, vocabulary: list[str]) -> None:
        self._vocabulary = vocabulary

    @property
    def model_id(self) -> str:
        return "keyword"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [1.0 if word in text.lower() else 0.01 for word in self._vocabulary] for text in texts
        ]


class ReverseReranker(Reranker):
    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        reranked = [
            ScoredChunk(chunk=c.chunk, score=float(i), origin="reranked")
            for i, c in enumerate(reversed(list(candidates)))
        ]
        return reranked[:top_k]


def test_rrf_prefers_chunk_present_in_both_lists() -> None:
    shared = make_chunk("shared", 0)
    dense_only = make_chunk("dense", 1)
    sparse_only = make_chunk("sparse", 2)
    fused = reciprocal_rank_fusion(
        [
            [scored(dense_only, 0.9), scored(shared, 0.8)],
            [scored(shared, 5.0), scored(sparse_only, 4.0)],
        ],
        k=60,
    )
    assert fused[0].chunk.id == shared.id
    assert fused[0].origin == "fused"
    assert len(fused) == 3


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def make_retriever(reranker: Reranker | None) -> HybridRetriever:
    vocabulary = ["cats", "finance", "security"]
    store = InMemoryVectorStore()
    bm25 = BM25Index()
    config = RetrievalConfig(dense_top_k=5, sparse_top_k=5, rrf_k=60, final_top_k=2)
    return HybridRetriever(
        vector_store=store,
        bm25_index=bm25,
        embedder=KeywordEmbedder(vocabulary),
        config=config,
        reranker=reranker,
    )


async def seed(retriever: HybridRetriever) -> list[Chunk]:
    chunks = [
        make_chunk("cats are wonderful pets", 0),
        make_chunk("finance quarterly report", 1),
        make_chunk("security training policy", 2),
    ]
    embedder = retriever._embedder
    vectors = await embedder.embed([chunk.text for chunk in chunks])
    await retriever._vector_store.upsert(chunks, vectors)
    retriever._bm25_index.rebuild(chunks)
    return chunks


async def test_hybrid_retrieve_returns_relevant_first() -> None:
    retriever = make_retriever(reranker=None)
    await seed(retriever)
    results = await retriever.retrieve("cats")
    assert len(results) == 2
    assert "cats" in results[0].chunk.text
    assert results[0].origin == "fused"


async def test_hybrid_respects_final_top_k() -> None:
    retriever = make_retriever(reranker=NoopReranker())
    await seed(retriever)
    results = await retriever.retrieve("finance report")
    assert len(results) <= 2


async def test_hybrid_applies_reranker() -> None:
    retriever = make_retriever(reranker=ReverseReranker())
    await seed(retriever)
    results = await retriever.retrieve("cats")
    assert all(result.origin == "reranked" for result in results)


async def test_noop_reranker_slices() -> None:
    chunk = make_chunk("a", 0)
    results = await NoopReranker().rerank("q", [scored(chunk, 1.0)] * 3, top_k=2)
    assert len(results) == 2


@pytest.mark.parametrize("query", ["cats", "finance", "security"])
async def test_hybrid_each_topic_found(query: str) -> None:
    retriever = make_retriever(reranker=None)
    await seed(retriever)
    results = await retriever.retrieve(query)
    assert results
    assert query in results[0].chunk.text
