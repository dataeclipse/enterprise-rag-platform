from collections.abc import Sequence
from uuid import uuid4

import pytest

from rag.config import ChunkingConfig
from rag.exceptions import ChunkingError
from rag.ingestion.chunkers import (
    ChunkerFactory,
    RecursiveChunker,
    SemanticChunker,
    split_sentences,
)


class FakeEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._mapping[text] for text in texts]


def test_split_sentences() -> None:
    text = "First sentence. Second one! Third?\n\nParagraph break"
    assert split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Third?",
        "Paragraph break",
    ]


def test_recursive_chunker_sizes_and_ids() -> None:
    document_id = uuid4()
    text = " ".join(f"word{i}" for i in range(200))
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=20).split(document_id, text)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].id == f"{document_id}:0"
    assert all(chunk.document_id == document_id for chunk in chunks)


def test_recursive_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ChunkingError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_semantic_chunker_splits_at_topic_boundary() -> None:
    document_id = uuid4()
    sentences = {
        "Cats purr.": [1.0, 0.0],
        "Cats meow.": [0.99, 0.05],
        "Stocks fell.": [0.0, 1.0],
        "Markets crashed.": [0.05, 0.99],
    }
    text = "Cats purr. Cats meow. Stocks fell. Markets crashed."
    chunker = SemanticChunker(FakeEncoder(sentences), breakpoint_percentile=60.0)
    chunks = chunker.split(document_id, text)
    assert len(chunks) == 2
    assert "Cats" in chunks[0].text
    assert "Stocks" in chunks[1].text
    assert "Cats" not in chunks[1].text


def test_semantic_chunker_single_sentence() -> None:
    document_id = uuid4()
    chunker = SemanticChunker(FakeEncoder({"Only one.": [1.0, 0.0]}))
    chunks = chunker.split(document_id, "Only one.")
    assert len(chunks) == 1
    assert chunks[0].text == "Only one."


def test_semantic_chunker_empty_text() -> None:
    chunker = SemanticChunker(FakeEncoder({}))
    assert chunker.split(uuid4(), "   ") == []


def test_semantic_chunker_validates_percentile() -> None:
    with pytest.raises(ChunkingError):
        SemanticChunker(FakeEncoder({}), breakpoint_percentile=100.0)


def test_factory_recursive() -> None:
    config = ChunkingConfig(strategy="recursive", chunk_size=256, chunk_overlap=32)
    assert isinstance(ChunkerFactory.from_config(config), RecursiveChunker)


def test_factory_semantic_requires_encoder() -> None:
    config = ChunkingConfig(strategy="semantic")
    with pytest.raises(ChunkingError, match="requires a sentence encoder"):
        ChunkerFactory.from_config(config)
    assert isinstance(ChunkerFactory.from_config(config, encoder=FakeEncoder({})), SemanticChunker)
