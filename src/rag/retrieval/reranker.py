import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from rag.exceptions import EmbeddingError
from rag.models import ScoredChunk


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoopReranker(Reranker):
    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_k])


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed, use the ml extra"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def _score(self, query: str, texts: list[str]) -> list[float]:
        model = self._ensure_model()
        predictions = model.predict([(query, text) for text in texts])
        return [float(value) for value in predictions]

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        async with self._lock:
            scores = await asyncio.to_thread(
                self._score, query, [candidate.chunk.text for candidate in candidates]
            )
        reranked = [
            ScoredChunk(chunk=candidate.chunk, score=score, origin="reranked")
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
