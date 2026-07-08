import re

from rag.agents.state import GraphState
from rag.models import Answer, Citation, QueryCategory

_MARKER = re.compile(r"\[(\d+)\]")
_QUOTE_LENGTH = 200


class CitationNode:
    async def __call__(self, state: GraphState) -> GraphState:
        draft = state.get("draft", "")
        context = state.get("context", [])
        citations: list[Citation] = []
        seen: set[int] = set()
        for match in _MARKER.finditer(draft):
            position = int(match.group(1)) - 1
            if position in seen or not 0 <= position < len(context):
                continue
            seen.add(position)
            chunk = context[position].chunk
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    source=chunk.metadata.get("source", str(chunk.document_id)),
                    chunk_id=chunk.id,
                    quote=chunk.text[:_QUOTE_LENGTH],
                )
            )
        answer = Answer(
            text=draft,
            citations=citations,
            category=state.get("category", QueryCategory.FACTUAL),
            correction_rounds=state.get("correction_rounds", 0),
        )
        return {"answer": answer}
