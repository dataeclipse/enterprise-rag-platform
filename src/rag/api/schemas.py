from uuid import UUID

from pydantic import BaseModel, Field

from rag.models import Answer, QueryCategory


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class CitationSchema(BaseModel):
    document_id: UUID
    source: str
    chunk_id: str
    quote: str


class AnswerResponse(BaseModel):
    text: str
    citations: list[CitationSchema]
    category: QueryCategory
    correction_rounds: int
    pii_redacted: bool = False

    @classmethod
    def from_answer(cls, answer: Answer, pii_redacted: bool = False) -> "AnswerResponse":
        return cls(
            text=answer.text,
            citations=[
                CitationSchema(
                    document_id=citation.document_id,
                    source=citation.source,
                    chunk_id=citation.chunk_id,
                    quote=citation.quote,
                )
                for citation in answer.citations
            ],
            category=answer.category,
            correction_rounds=answer.correction_rounds,
            pii_redacted=pii_redacted,
        )


class UploadResponse(BaseModel):
    document_id: UUID
    status: str
    version: int
    deduplicated: bool


class HealthResponse(BaseModel):
    status: str
    version: str
