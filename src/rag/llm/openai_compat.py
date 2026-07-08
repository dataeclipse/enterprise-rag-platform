from collections.abc import AsyncIterator, Sequence
from typing import cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.exceptions import LLMError
from rag.llm.base import ChatMessage, LLMProvider

_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

_retry_policy = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    reraise=True,
)


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return cast(
        list[ChatCompletionMessageParam],
        [{"role": message.role, "content": message.content} for message in messages],
    )


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @_retry_policy
    async def _create_completion(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(messages),
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            stream=False,
        )

    @_retry_policy
    async def _create_stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncStream[ChatCompletionChunk]:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(messages),
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await self._create_completion(messages, temperature, max_tokens)
        except OpenAIError as exc:
            raise LLMError("llm completion request failed") from exc
        content = response.choices[0].message.content
        if not content:
            raise LLMError("llm returned empty completion")
        return content

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            response = await self._create_stream(messages, temperature, max_tokens)
        except OpenAIError as exc:
            raise LLMError("llm streaming request failed") from exc
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
