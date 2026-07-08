from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest

from rag.agents.citation import CitationNode
from rag.agents.graph import build_agent_graph
from rag.agents.nodes import make_route_after_critic, route_after_router
from rag.agents.state import CritiqueResult, GraphState
from rag.config import AgentsConfig, RetrievalConfig
from rag.ingestion.embedders import Embedder
from rag.llm.base import ChatMessage, LLMProvider
from rag.models import Chunk, QueryCategory, ScoredChunk
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import InMemoryVectorStore

DOC_ID = uuid4()


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(messages)
        return self._responses.pop(0)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield await self.complete(messages, temperature, max_tokens)


class UniformEmbedder(Embedder):
    @property
    def model_id(self) -> str:
        return "uniform"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.5] for _ in texts]


def make_scored(text: str, index: int, source: str = "policy.pdf") -> ScoredChunk:
    chunk = Chunk(
        id=f"{DOC_ID}:{index}",
        document_id=DOC_ID,
        text=text,
        index=index,
        metadata={"source": source},
    )
    return ScoredChunk(chunk=chunk, score=1.0, origin="fused")


async def make_retriever() -> HybridRetriever:
    store = InMemoryVectorStore()
    bm25 = BM25Index()
    chunks = [
        Chunk(
            id=f"{DOC_ID}:0",
            document_id=DOC_ID,
            text="The vacation policy grants 25 days of paid leave per year.",
            index=0,
            metadata={"source": "hr-handbook.pdf"},
        ),
        Chunk(
            id=f"{DOC_ID}:1",
            document_id=DOC_ID,
            text="Remote work is allowed up to three days per week.",
            index=1,
            metadata={"source": "hr-handbook.pdf"},
        ),
    ]
    embedder = UniformEmbedder()
    await store.upsert(chunks, await embedder.embed([c.text for c in chunks]))
    bm25.rebuild(chunks)
    config = RetrievalConfig(dense_top_k=5, sparse_top_k=5, final_top_k=2)
    return HybridRetriever(vector_store=store, bm25_index=bm25, embedder=embedder, config=config)


ROUTER_FACTUAL = '{"category": "factual", "reasoning": "fact lookup"}'
ROUTER_OOS = '{"category": "out_of_scope", "reasoning": "chit-chat"}'
CRITIC_APPROVE = '{"verdict": "approve", "grounded": true, "issues": []}'
CRITIC_REVISE = (
    '{"verdict": "revise", "grounded": false, "issues": ["claim about 30 days is wrong"]}'
)


async def test_full_graph_happy_path() -> None:
    llm = ScriptedLLM(
        [
            ROUTER_FACTUAL,
            "Employees get 25 days of paid leave [1].",
            CRITIC_APPROVE,
        ]
    )
    graph = build_agent_graph(llm, await make_retriever(), AgentsConfig())
    result = await graph.ainvoke({"query": "How many vacation days do employees get?"})
    answer = result["answer"]
    assert "25 days" in answer.text
    assert answer.category is QueryCategory.FACTUAL
    assert answer.correction_rounds == 0
    assert len(answer.citations) == 1
    assert answer.citations[0].source == "hr-handbook.pdf"


async def test_graph_self_correction_loop() -> None:
    llm = ScriptedLLM(
        [
            ROUTER_FACTUAL,
            "Employees get 30 days of paid leave [1].",
            CRITIC_REVISE,
            "Employees get 25 days of paid leave [1].",
            CRITIC_APPROVE,
        ]
    )
    graph = build_agent_graph(llm, await make_retriever(), AgentsConfig())
    result = await graph.ainvoke({"query": "How many vacation days?"})
    answer = result["answer"]
    assert "25 days" in answer.text
    assert answer.correction_rounds == 1
    revision_prompt = llm.calls[3][0].content
    assert "claim about 30 days is wrong" in revision_prompt
    assert "30 days of paid leave" in revision_prompt


async def test_graph_correction_rounds_capped() -> None:
    llm = ScriptedLLM(
        [
            ROUTER_FACTUAL,
            "Draft one [1].",
            CRITIC_REVISE,
            "Draft two [1].",
            CRITIC_REVISE,
            "Draft three [1].",
            CRITIC_REVISE,
        ]
    )
    config = AgentsConfig(max_correction_rounds=2)
    graph = build_agent_graph(llm, await make_retriever(), config)
    result = await graph.ainvoke({"query": "How many vacation days?"})
    answer = result["answer"]
    assert answer.correction_rounds == 2
    assert answer.text == "Draft three [1]."


async def test_graph_out_of_scope_short_circuits() -> None:
    llm = ScriptedLLM([ROUTER_OOS])
    graph = build_agent_graph(llm, await make_retriever(), AgentsConfig())
    result = await graph.ainvoke({"query": "Tell me a joke"})
    answer = result["answer"]
    assert answer.category is QueryCategory.OUT_OF_SCOPE
    assert answer.citations == []
    assert len(llm.calls) == 1


async def test_router_falls_back_to_factual_on_garbage() -> None:
    llm = ScriptedLLM(
        [
            "not json at all",
            "Answer [1].",
            CRITIC_APPROVE,
        ]
    )
    graph = build_agent_graph(llm, await make_retriever(), AgentsConfig())
    result = await graph.ainvoke({"query": "vacation days?"})
    assert result["answer"].category is QueryCategory.FACTUAL


def test_route_after_router() -> None:
    assert route_after_router({"category": QueryCategory.OUT_OF_SCOPE}) == "reject"
    assert route_after_router({"category": QueryCategory.FACTUAL}) == "retrieve"


def test_route_after_critic_respects_cap() -> None:
    route = make_route_after_critic(AgentsConfig(max_correction_rounds=1))
    revise = CritiqueResult(verdict="revise", grounded=False, issues=["x"])
    approve = CritiqueResult(verdict="approve")
    assert route({"critique": revise, "correction_rounds": 0}) == "revise"
    assert route({"critique": revise, "correction_rounds": 1}) == "cite"
    assert route({"critique": approve, "correction_rounds": 0}) == "cite"


async def test_citation_node_dedupes_and_ignores_invalid_markers() -> None:
    state: GraphState = {
        "query": "q",
        "draft": "Fact [1]. Same fact [1]. Missing [9]. Other [2].",
        "context": [make_scored("first passage", 0), make_scored("second passage", 1)],
        "category": QueryCategory.ANALYTICAL,
        "correction_rounds": 0,
    }
    result = await CitationNode()(state)
    answer = result["answer"]
    assert [c.chunk_id for c in answer.citations] == [f"{DOC_ID}:0", f"{DOC_ID}:1"]
    assert answer.citations[0].quote == "first passage"


async def test_citation_node_long_quote_truncated() -> None:
    long_text = "x" * 500
    state: GraphState = {
        "query": "q",
        "draft": "See [1].",
        "context": [make_scored(long_text, 0)],
        "category": QueryCategory.FACTUAL,
        "correction_rounds": 0,
    }
    result = await CitationNode()(state)
    assert len(result["answer"].citations[0].quote) == 200


async def test_reasoner_receives_numbered_context() -> None:
    llm = ScriptedLLM([ROUTER_FACTUAL, "Answer [1].", CRITIC_APPROVE])
    graph = build_agent_graph(llm, await make_retriever(), AgentsConfig())
    await graph.ainvoke({"query": "vacation days?"})
    reasoner_system = llm.calls[1][0].content
    assert "[1]" in reasoner_system
    assert "[2]" in reasoner_system
    assert "vacation policy" in reasoner_system


@pytest.mark.parametrize(
    "category",
    [QueryCategory.FACTUAL, QueryCategory.ANALYTICAL, QueryCategory.SUMMARY],
)
async def test_all_in_scope_categories_retrieve(category: QueryCategory) -> None:
    assert route_after_router({"category": category}) == "retrieve"
