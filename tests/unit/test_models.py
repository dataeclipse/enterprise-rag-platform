from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag.models import (
    Answer,
    Chunk,
    DocumentFormat,
    DocumentMeta,
    DocumentStatus,
    QueryCategory,
    ScoredChunk,
)


def test_document_meta_defaults() -> None:
    doc = DocumentMeta(source="a.pdf", format=DocumentFormat.PDF, content_hash="abc")
    assert doc.status is DocumentStatus.PENDING
    assert doc.version == 1
    assert doc.created_at.tzinfo is not None


def test_chunk_requires_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({"id": "c1"})


def test_scored_chunk_origin_literal() -> None:
    chunk = Chunk(id="c1", document_id=uuid4(), text="t", index=0)
    ScoredChunk(chunk=chunk, score=0.5, origin="dense")
    with pytest.raises(ValidationError):
        ScoredChunk.model_validate({"chunk": chunk, "score": 0.5, "origin": "magic"})


def test_answer_defaults() -> None:
    answer = Answer(text="hi", category=QueryCategory.FACTUAL)
    assert answer.citations == []
    assert answer.correction_rounds == 0
