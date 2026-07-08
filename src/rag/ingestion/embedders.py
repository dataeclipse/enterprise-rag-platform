import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

from openai import AsyncOpenAI

from rag.config import EmbeddingsConfig, RedisConfig
from rag.exceptions import EmbeddingError


class Embedder(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self._model_name

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed, use the ml extra"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(list(texts), batch_size=self._batch_size, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self.encode, texts)


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str = "text-embedding-3-large",
        batch_size: int = 128,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self._model_name

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = await self._client.embeddings.create(model=self._model_name, input=batch)
            except Exception as exc:
                raise EmbeddingError("embedding request failed") from exc
            vectors.extend([item.embedding for item in response.data])
        return vectors


class BytesCache(Protocol):
    async def mget(self, keys: Sequence[str]) -> list[bytes | None]: ...

    async def setex(self, name: str, time: int, value: bytes) -> Any: ...


class CachedEmbedder(Embedder):
    def __init__(
        self,
        inner: Embedder,
        cache: BytesCache,
        ttl_seconds: int = 86400,
        namespace: str = "emb",
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl_seconds = ttl_seconds
        self._namespace = namespace

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._inner.model_id}:{text}".encode()).hexdigest()
        return f"{self._namespace}:{digest}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [self._key(text) for text in texts]
        cached = await self._cache.mget(keys)
        results: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, payload in enumerate(cached):
            if payload is None:
                missing.append(index)
            else:
                results[index] = json.loads(payload)
        if missing:
            fresh = await self._inner.embed([texts[index] for index in missing])
            for index, vector in zip(missing, fresh, strict=True):
                results[index] = vector
                await self._cache.setex(keys[index], self._ttl_seconds, json.dumps(vector).encode())
        return [results[index] for index in range(len(texts))]


class EmbedderFactory:
    @staticmethod
    def from_config(
        config: EmbeddingsConfig,
        redis_config: RedisConfig | None = None,
        cache: BytesCache | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> Embedder:
        embedder: Embedder
        if config.provider == "local":
            embedder = LocalEmbedder(model_name=config.model, batch_size=config.batch_size)
        else:
            if openai_client is None:
                raise EmbeddingError("openai provider requires a configured client")
            embedder = OpenAIEmbedder(
                client=openai_client, model_name=config.model, batch_size=config.batch_size
            )
        if cache is not None:
            ttl = redis_config.embedding_cache_ttl_seconds if redis_config else 86400
            embedder = CachedEmbedder(inner=embedder, cache=cache, ttl_seconds=ttl)
        return embedder
