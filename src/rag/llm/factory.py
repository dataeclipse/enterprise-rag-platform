from openai import AsyncOpenAI

from rag.config import LLMConfig
from rag.llm.base import LLMProvider
from rag.llm.openai_compat import OpenAICompatProvider


def build_llm_provider(config: LLMConfig, client: AsyncOpenAI | None = None) -> LLMProvider:
    if client is None:
        client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value() or "not-required",
            timeout=config.timeout_seconds,
        )
    return OpenAICompatProvider(
        client=client,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
