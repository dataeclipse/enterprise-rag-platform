from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "qwen3:8b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: float = 120.0


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"
    vector_size: int = 1024


class PostgresConfig(BaseModel):
    dsn: SecretStr = SecretStr("postgresql+asyncpg://rag:rag@localhost:5432/rag")
    pool_size: int = 10


class RedisConfig(BaseModel):
    url: SecretStr = SecretStr("redis://localhost:6379/0")
    embedding_cache_ttl_seconds: int = 86400


class RabbitMQConfig(BaseModel):
    url: SecretStr = SecretStr("amqp://rag:rag@localhost:5672/")
    ingestion_queue: str = "ingestion"
    prefetch_count: int = 4


class AuthConfig(BaseModel):
    secret_key: SecretStr
    algorithm: str = "HS256"
    token_ttl_seconds: int = 3600


class EmbeddingsConfig(BaseModel):
    provider: Literal["local", "openai"] = "local"
    model: str = "BAAI/bge-m3"
    batch_size: int = 32


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = True


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_breakpoint_percentile: float = 90.0


class RetrievalConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5


class AgentsConfig(BaseModel):
    max_correction_rounds: int = 2
    max_context_chunks: int = 8


class GuardrailsConfig(BaseModel):
    pii_redaction: bool = True
    injection_detection: bool = True
    injection_threshold: float = 0.5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    llm: LLMConfig = LLMConfig()
    qdrant: QdrantConfig = QdrantConfig()
    postgres: PostgresConfig = PostgresConfig()
    redis: RedisConfig = RedisConfig()
    rabbitmq: RabbitMQConfig = RabbitMQConfig()
    auth: AuthConfig
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    reranker: RerankerConfig = RerankerConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agents: AgentsConfig = AgentsConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()


@lru_cache
def get_settings() -> Settings:
    return Settings()
