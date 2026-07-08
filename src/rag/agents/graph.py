from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag.agents.citation import CitationNode
from rag.agents.nodes import (
    CriticNode,
    ReasonNode,
    RejectNode,
    RetrieveNode,
    RouterNode,
    make_route_after_critic,
    route_after_router,
)
from rag.agents.state import GraphState
from rag.config import AgentsConfig
from rag.llm.base import LLMProvider
from rag.retrieval.hybrid import HybridRetriever


def build_agent_graph(
    llm: LLMProvider,
    retriever: HybridRetriever,
    config: AgentsConfig,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node("router", RouterNode(llm))
    graph.add_node("retrieve", RetrieveNode(retriever, config.max_context_chunks))
    graph.add_node("reason", ReasonNode(llm))
    graph.add_node("critic", CriticNode(llm))
    graph.add_node("cite", CitationNode())
    graph.add_node("reject", RejectNode())
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", route_after_router, {"retrieve": "retrieve", "reject": "reject"}
    )
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "critic")
    graph.add_conditional_edges(
        "critic", make_route_after_critic(config), {"revise": "reason", "cite": "cite"}
    )
    graph.add_edge("cite", END)
    graph.add_edge("reject", END)
    return graph.compile()
