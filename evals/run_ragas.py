import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPORT_PATH, build_indexes, load_golden
from rag.agents.graph import build_agent_graph
from rag.config import AgentsConfig, LLMConfig, RetrievalConfig
from rag.ingestion.embedders import LocalEmbedder
from rag.llm.factory import build_llm_provider
from rag.retrieval.hybrid import HybridRetriever


async def main() -> None:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        sys.stderr.write("ragas is not installed, run: uv sync --extra evals --extra ml\n")
        raise SystemExit(1) from None

    embedder = LocalEmbedder(model_name=os.environ.get("RAG_EVAL_EMBEDDER", "BAAI/bge-m3"))
    store, bm25, _ = await build_indexes(embedder)
    retriever = HybridRetriever(
        vector_store=store,
        bm25_index=bm25,
        embedder=embedder,
        config=RetrievalConfig(final_top_k=5),
    )
    llm_config = LLMConfig(
        base_url=os.environ.get("RAG_LLM__BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("RAG_LLM__MODEL", "qwen3:8b"),
    )
    graph = build_agent_graph(build_llm_provider(llm_config), retriever, AgentsConfig())

    rows = []
    for example in load_golden():
        state = await graph.ainvoke({"query": example.question})
        answer = state["answer"]
        rows.append(
            {
                "user_input": example.question,
                "response": answer.text,
                "retrieved_contexts": [scored.chunk.text for scored in state.get("context", [])],
                "reference": example.answer,
            }
        )

    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = result.to_pandas().mean(numeric_only=True)
    lines = ["", "## RAGAS Metrics", "", "| Metric | Score |", "|--------|-------|"]
    for metric_name, value in scores.items():
        lines.append(f"| {metric_name} | {value:.3f} |")
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    REPORT_PATH.write_text(report + "\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
