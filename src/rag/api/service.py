import json
from collections.abc import AsyncIterator
from typing import cast

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from rag.agents.state import GraphState
from rag.api.schemas import AnswerResponse
from rag.exceptions import LLMError
from rag.models import Answer
from rag.observability.metrics import Metrics


class QueryEvent(BaseModel):
    event: str
    data: str


class QueryService:
    def __init__(
        self,
        graph: CompiledStateGraph[GraphState, None, GraphState, GraphState],
        metrics: Metrics | None = None,
    ) -> None:
        self._graph = graph
        self._metrics = metrics

    async def answer(self, query: str) -> Answer:
        result = cast(GraphState, await self._graph.ainvoke({"query": query}))
        return self._finalize(result)

    async def stream(self, query: str) -> AsyncIterator[QueryEvent]:
        final: Answer | None = None
        async for update in self._graph.astream({"query": query}, stream_mode="updates"):
            for node_name, node_state in update.items():
                yield QueryEvent(event="stage", data=node_name)
                if node_state and "answer" in node_state:
                    final = node_state["answer"]
        if final is None:
            raise LLMError("agent graph finished without an answer")
        self._observe(final)
        payload = AnswerResponse.from_answer(final).model_dump(mode="json")
        yield QueryEvent(event="answer", data=json.dumps(payload))

    def _finalize(self, result: GraphState) -> Answer:
        answer = result.get("answer")
        if answer is None:
            raise LLMError("agent graph finished without an answer")
        if self._metrics is not None:
            self._metrics.retrieval_chunks.observe(len(result.get("context", [])))
        self._observe(answer)
        return answer

    def _observe(self, answer: Answer) -> None:
        if self._metrics is not None:
            self._metrics.correction_rounds.observe(answer.correction_rounds)
