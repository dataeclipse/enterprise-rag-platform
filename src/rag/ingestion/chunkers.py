import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

import numpy as np
import numpy.typing as npt
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import ChunkingConfig
from rag.exceptions import ChunkingError
from rag.models import Chunk

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class SentenceEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _build_chunks(document_id: UUID, parts: Sequence[str]) -> list[Chunk]:
    return [
        Chunk(id=f"{document_id}:{index}", document_id=document_id, text=text, index=index)
        for index, text in enumerate(parts)
    ]


class Chunker(ABC):
    @abstractmethod
    def split(self, document_id: UUID, text: str) -> list[Chunk]: ...


class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than chunk_size")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, document_id: UUID, text: str) -> list[Chunk]:
        return _build_chunks(document_id, self._splitter.split_text(text))


class SemanticChunker(Chunker):
    def __init__(
        self,
        encoder: SentenceEncoder,
        breakpoint_percentile: float = 90.0,
        max_chunk_size: int = 2048,
    ) -> None:
        if not 0.0 < breakpoint_percentile < 100.0:
            raise ChunkingError("breakpoint_percentile must be in (0, 100)")
        self._encoder = encoder
        self._breakpoint_percentile = breakpoint_percentile
        self._fallback = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=0)
        self._max_chunk_size = max_chunk_size

    def split(self, document_id: UUID, text: str) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return _build_chunks(document_id, self._cap(sentences[0]))
        distances = self._consecutive_distances(sentences)
        threshold = float(np.percentile(distances, self._breakpoint_percentile))
        groups: list[list[str]] = [[sentences[0]]]
        for sentence, distance in zip(sentences[1:], distances, strict=True):
            if distance > threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        parts: list[str] = []
        for group in groups:
            parts.extend(self._cap(" ".join(group)))
        return _build_chunks(document_id, parts)

    def _consecutive_distances(self, sentences: list[str]) -> npt.NDArray[np.float64]:
        vectors = np.asarray(self._encoder.encode(sentences), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, a_min=1e-12, a_max=None)
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
        return 1.0 - similarities

    def _cap(self, text: str) -> list[str]:
        if len(text) <= self._max_chunk_size:
            return [text]
        return self._fallback.split_text(text)


class ChunkerFactory:
    @staticmethod
    def from_config(config: ChunkingConfig, encoder: SentenceEncoder | None = None) -> Chunker:
        if config.strategy == "recursive":
            return RecursiveChunker(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
        if encoder is None:
            raise ChunkingError("semantic chunking requires a sentence encoder")
        return SemanticChunker(
            encoder=encoder,
            breakpoint_percentile=config.semantic_breakpoint_percentile,
            max_chunk_size=config.chunk_size * 4,
        )
