import re
from collections.abc import Sequence
from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi

from rag.models import Chunk, ScoredChunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if self._chunks:
            self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self._chunks])
        else:
            self._bm25 = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        self.rebuild([*self._chunks, *chunks])

    def remove_document(self, document_id: UUID) -> None:
        self.rebuild([chunk for chunk in self._chunks if chunk.document_id != document_id])

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[index], score=float(scores[index]), origin="sparse")
            for index in order
            if scores[index] > 0.0
        ]
