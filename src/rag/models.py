from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class DocumentMeta(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: str
    format: DocumentFormat
    content_hash: str
    version: int = 1
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    extra: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: UUID
    text: str
    index: int
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    origin: Literal["dense", "sparse", "fused", "reranked"]


class Citation(BaseModel):
    document_id: UUID
    source: str
    chunk_id: str
    quote: str


class QueryCategory(StrEnum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    SUMMARY = "summary"
    OUT_OF_SCOPE = "out_of_scope"


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    category: QueryCategory
    correction_rounds: int = 0
