from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from rag.models import Answer, QueryCategory, ScoredChunk


class RouteDecision(BaseModel):
    category: QueryCategory
    reasoning: str = ""


class CritiqueResult(BaseModel):
    verdict: Literal["approve", "revise"]
    grounded: bool = True
    issues: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    query: str
    category: QueryCategory
    context: list[ScoredChunk]
    draft: str
    critique: CritiqueResult
    correction_rounds: int
    answer: Answer
