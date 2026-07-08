import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal

from pydantic import BaseModel, ValidationError

from rag.exceptions import LLMError

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        temperature: float | None = None,
    ) -> T:
        raw = await self.complete(messages, temperature=temperature)
        return parse_structured(raw, schema)


def parse_structured[T: BaseModel](raw: str, schema: type[T]) -> T:
    candidate = raw.strip()
    match = _JSON_BLOCK.search(candidate)
    if match:
        candidate = match.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError("model output is not valid json") from exc
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"model output does not match {schema.__name__}") from exc
