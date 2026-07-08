class RagError(Exception):
    pass


class LoaderError(RagError):
    pass


class ChunkingError(RagError):
    pass


class EmbeddingError(RagError):
    pass


class VectorStoreError(RagError):
    pass


class LLMError(RagError):
    pass


class GuardrailViolationError(RagError):
    pass


class AuthError(RagError):
    pass
