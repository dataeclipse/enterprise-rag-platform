import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.config import EmbeddingsConfig
from rag.exceptions import EmbeddingError
from rag.ingestion.embedders import (
    CachedEmbedder,
    Embedder,
    EmbedderFactory,
    LocalEmbedder,
    OpenAIEmbedder,
)


class StaticEmbedder(Embedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "static"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        return [self.store.get(key) for key in keys]

    async def setex(self, name: str, time: int, value: bytes) -> Any:
        self.store[name] = value


async def test_cached_embedder_miss_then_hit() -> None:
    inner = StaticEmbedder()
    cache = FakeCache()
    embedder = CachedEmbedder(inner=inner, cache=cache, ttl_seconds=60)
    first = await embedder.embed(["alpha", "beta"])
    assert first == [[5.0, 1.0], [4.0, 1.0]]
    assert len(inner.calls) == 1
    second = await embedder.embed(["alpha", "beta"])
    assert second == first
    assert len(inner.calls) == 1


async def test_cached_embedder_partial_hit() -> None:
    inner = StaticEmbedder()
    cache = FakeCache()
    embedder = CachedEmbedder(inner=inner, cache=cache, ttl_seconds=60)
    await embedder.embed(["alpha"])
    result = await embedder.embed(["alpha", "gamma"])
    assert result == [[5.0, 1.0], [5.0, 1.0]]
    assert inner.calls == [["alpha"], ["gamma"]]


async def test_cached_embedder_key_includes_model() -> None:
    cache = FakeCache()
    embedder = CachedEmbedder(inner=StaticEmbedder(), cache=cache, ttl_seconds=60)
    await embedder.embed(["alpha"])
    key = next(iter(cache.store))
    assert key.startswith("emb:")
    assert json.loads(cache.store[key]) == [5.0, 1.0]


async def test_openai_embedder_batches() -> None:
    client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2])]
    client.embeddings.create = AsyncMock(return_value=response)
    embedder = OpenAIEmbedder(client=client, model_name="test-model", batch_size=1)
    result = await embedder.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.1, 0.2]]
    assert client.embeddings.create.await_count == 2


async def test_openai_embedder_wraps_errors() -> None:
    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))
    embedder = OpenAIEmbedder(client=client)
    with pytest.raises(EmbeddingError, match="embedding request failed"):
        await embedder.embed(["a"])


async def test_local_embedder_requires_ml_extra_or_model() -> None:
    embedder = LocalEmbedder(model_name="fake-model")
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.5, 0.5]]
    embedder._model = fake_model
    result = await embedder.embed(["text"])
    assert result == [[0.5, 0.5]]
    fake_model.encode.assert_called_once()


async def test_embedders_return_empty_for_empty_input() -> None:
    assert await StaticEmbedder().embed([]) == []
    assert await OpenAIEmbedder(client=MagicMock()).embed([]) == []
    assert await LocalEmbedder().embed([]) == []


def test_factory_local_with_cache() -> None:
    config = EmbeddingsConfig(provider="local", model="m", batch_size=8)
    embedder = EmbedderFactory.from_config(config, cache=FakeCache())
    assert isinstance(embedder, CachedEmbedder)
    assert embedder.model_id == "m"


def test_factory_openai_requires_client() -> None:
    config = EmbeddingsConfig(provider="openai", model="m")
    with pytest.raises(EmbeddingError, match="requires a configured client"):
        EmbedderFactory.from_config(config)
