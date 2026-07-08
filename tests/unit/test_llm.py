from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from rag.config import LLMConfig
from rag.exceptions import LLMError
from rag.llm.base import ChatMessage, parse_structured
from rag.llm.factory import build_llm_provider
from rag.llm.openai_compat import OpenAICompatProvider

MESSAGES = [ChatMessage(role="user", content="hello")]


class RouteDecision(BaseModel):
    category: str
    confidence: float


def make_completion_response(content: str | None) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def make_provider(client: MagicMock) -> OpenAICompatProvider:
    return OpenAICompatProvider(client=client, model="test-model")


async def test_complete_returns_content() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_completion_response("answer"))
    provider = make_provider(client)
    result = await provider.complete(MESSAGES)
    assert result == "answer"
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["stream"] is False


async def test_complete_empty_content_raises() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_completion_response(None))
    with pytest.raises(LLMError, match="empty completion"):
        await make_provider(client).complete(MESSAGES)


async def test_complete_retries_transient_errors() -> None:
    request = httpx.Request("POST", "http://test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            APIConnectionError(request=request),
            make_completion_response("recovered"),
        ]
    )
    result = await make_provider(client).complete(MESSAGES)
    assert result == "recovered"
    assert client.chat.completions.create.await_count == 2


async def test_complete_wraps_exhausted_retries() -> None:
    request = httpx.Request("POST", "http://test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=APIConnectionError(request=request))
    with pytest.raises(LLMError, match="completion request failed"):
        await make_provider(client).complete(MESSAGES)
    assert client.chat.completions.create.await_count == 3


def make_stream_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return chunk


async def test_stream_yields_deltas() -> None:
    async def fake_stream() -> AsyncIterator[Any]:
        for part in [make_stream_chunk("hel"), make_stream_chunk(None), make_stream_chunk("lo")]:
            yield part

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=fake_stream())
    collected = [part async for part in make_provider(client).stream(MESSAGES)]
    assert collected == ["hel", "lo"]


async def test_complete_structured_parses_model() -> None:
    payload = '{"category": "factual", "confidence": 0.9}'
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_completion_response(payload))
    decision = await make_provider(client).complete_structured(MESSAGES, RouteDecision)
    assert decision.category == "factual"
    assert decision.confidence == pytest.approx(0.9)


def test_parse_structured_from_fenced_block() -> None:
    raw = 'Reasoning here.\n```json\n{"category": "summary", "confidence": 0.7}\n```\nDone.'
    decision = parse_structured(raw, RouteDecision)
    assert decision.category == "summary"


def test_parse_structured_from_embedded_object() -> None:
    raw = 'The decision is {"category": "analytical", "confidence": 1.0} as requested.'
    decision = parse_structured(raw, RouteDecision)
    assert decision.category == "analytical"


def test_parse_structured_invalid_json() -> None:
    with pytest.raises(LLMError, match="not valid json"):
        parse_structured("no json here", RouteDecision)


def test_parse_structured_schema_mismatch() -> None:
    with pytest.raises(LLMError, match="does not match RouteDecision"):
        parse_structured('{"wrong": true}', RouteDecision)


def test_factory_builds_provider() -> None:
    config = LLMConfig(model="custom-model")
    provider = build_llm_provider(config)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider._model == "custom-model"
