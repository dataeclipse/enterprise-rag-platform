from collections.abc import Callable

from rag.agents import prompts
from rag.agents.state import CritiqueResult, GraphState, RouteDecision
from rag.config import AgentsConfig
from rag.exceptions import LLMError
from rag.llm.base import ChatMessage, LLMProvider
from rag.models import Answer, QueryCategory
from rag.observability.logging import get_logger
from rag.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


class RouterNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: GraphState) -> GraphState:
        try:
            decision = await self._llm.complete_structured(
                [
                    ChatMessage(role="system", content=prompts.ROUTER_SYSTEM),
                    ChatMessage(role="user", content=state["query"]),
                ],
                RouteDecision,
                temperature=0.0,
            )
            category = decision.category
        except LLMError:
            logger.warning("router_fallback", query=state["query"])
            category = QueryCategory.FACTUAL
        return {"category": category}


class RetrieveNode:
    def __init__(self, retriever: HybridRetriever, max_context_chunks: int = 8) -> None:
        self._retriever = retriever
        self._max_context_chunks = max_context_chunks

    async def __call__(self, state: GraphState) -> GraphState:
        results = await self._retriever.retrieve(state["query"])
        return {"context": results[: self._max_context_chunks]}


class ReasonNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: GraphState) -> GraphState:
        context = prompts.format_context([scored.chunk.text for scored in state.get("context", [])])
        system = prompts.REASONER_SYSTEM.format(context=context)
        critique = state.get("critique")
        rounds = state.get("correction_rounds", 0)
        if critique is not None and critique.verdict == "revise":
            system += prompts.REASONER_REVISION_SUFFIX.format(
                issues="\n".join(f"- {issue}" for issue in critique.issues),
                draft=state.get("draft", ""),
            )
            rounds += 1
        draft = await self._llm.complete(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=state["query"]),
            ]
        )
        return {"draft": draft, "correction_rounds": rounds}


class CriticNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: GraphState) -> GraphState:
        context = prompts.format_context([scored.chunk.text for scored in state.get("context", [])])
        try:
            critique = await self._llm.complete_structured(
                [
                    ChatMessage(role="system", content=prompts.CRITIC_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=prompts.CRITIC_USER.format(
                            query=state["query"],
                            context=context,
                            draft=state.get("draft", ""),
                        ),
                    ),
                ],
                CritiqueResult,
                temperature=0.0,
            )
        except LLMError:
            logger.warning("critic_fallback", query=state["query"])
            critique = CritiqueResult(verdict="approve", grounded=True)
        return {"critique": critique}


class RejectNode:
    async def __call__(self, state: GraphState) -> GraphState:
        return {
            "answer": Answer(
                text=prompts.OUT_OF_SCOPE_ANSWER,
                citations=[],
                category=QueryCategory.OUT_OF_SCOPE,
                correction_rounds=0,
            )
        }


def route_after_router(state: GraphState) -> str:
    if state.get("category") is QueryCategory.OUT_OF_SCOPE:
        return "reject"
    return "retrieve"


def make_route_after_critic(config: AgentsConfig) -> Callable[[GraphState], str]:
    def route_after_critic(state: GraphState) -> str:
        critique = state.get("critique")
        rounds = state.get("correction_rounds", 0)
        if (
            critique is not None
            and critique.verdict == "revise"
            and rounds < config.max_correction_rounds
        ):
            return "revise"
        return "cite"

    return route_after_critic
