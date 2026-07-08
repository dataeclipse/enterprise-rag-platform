import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPORT_PATH, GoldenExample, build_indexes, load_golden
from rag.config import RetrievalConfig
from rag.ingestion.embedders import LocalEmbedder
from rag.models import ScoredChunk
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker

TOP_K = 5

RetrieveFn = Callable[[str], Awaitable[list[ScoredChunk]]]


def score_mode(
    examples: list[GoldenExample], results: dict[str, list[ScoredChunk]]
) -> tuple[float, float, float]:
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = 0.0
    for example in examples:
        retrieved = results[example.question]
        rank = next(
            (
                position + 1
                for position, item in enumerate(retrieved[:TOP_K])
                if item.chunk.metadata.get("source") == example.source
            ),
            None,
        )
        if rank is not None:
            reciprocal_ranks += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
    total = len(examples)
    return hits_at_1 / total, hits_at_3 / total, reciprocal_ranks / total


async def collect(
    examples: list[GoldenExample], retrieve: RetrieveFn
) -> dict[str, list[ScoredChunk]]:
    return {example.question: await retrieve(example.question) for example in examples}


async def main() -> None:
    model_name = os.environ.get("RAG_EVAL_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
    use_reranker = os.environ.get("RAG_EVAL_RERANK", "0") == "1"
    embedder = LocalEmbedder(model_name=model_name)
    store, bm25, chunks = await build_indexes(embedder, chunk_size=256, chunk_overlap=32)
    examples = load_golden()
    config = RetrievalConfig(dense_top_k=10, sparse_top_k=10, rrf_k=60, final_top_k=TOP_K)

    async def sparse_retrieve(query: str) -> list[ScoredChunk]:
        return bm25.search(query, TOP_K)

    async def dense_retrieve(query: str) -> list[ScoredChunk]:
        vectors = await embedder.embed([query])
        return await store.search(vectors[0], TOP_K)

    hybrid = HybridRetriever(vector_store=store, bm25_index=bm25, embedder=embedder, config=config)
    modes: dict[str, RetrieveFn] = {
        "bm25": sparse_retrieve,
        "dense": dense_retrieve,
        "hybrid_rrf": hybrid.retrieve,
    }
    if use_reranker:
        reranked = HybridRetriever(
            vector_store=store,
            bm25_index=bm25,
            embedder=embedder,
            config=config,
            reranker=CrossEncoderReranker(
                os.environ.get("RAG_EVAL_RERANKER", "BAAI/bge-reranker-v2-m3")
            ),
        )
        modes["hybrid_rerank"] = reranked.retrieve

    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Embedding model: `{model_name}`",
        f"Corpus: {len(chunks)} chunks from 3 documents, "
        f"{len(examples)} golden questions, top-k = {TOP_K}",
        "",
        "| Mode | Hit@1 | Hit@3 | MRR@5 |",
        "|------|-------|-------|-------|",
    ]
    for mode_name, retrieve in modes.items():
        results = await collect(examples, retrieve)
        hit_1, hit_3, mrr = score_mode(examples, results)
        lines.append(f"| {mode_name} | {hit_1:.3f} | {hit_3:.3f} | {mrr:.3f} |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
