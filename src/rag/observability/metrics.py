from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.requests_total = Counter(
            "rag_requests_total",
            "API requests by endpoint and status",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "rag_request_duration_seconds",
            "API request latency",
            ["endpoint"],
            registry=self.registry,
        )
        self.retrieval_chunks = Histogram(
            "rag_retrieval_chunks",
            "Chunks returned per retrieval",
            buckets=(0, 1, 2, 5, 8, 13, 21),
            registry=self.registry,
        )
        self.correction_rounds = Histogram(
            "rag_correction_rounds",
            "Self-correction rounds per answer",
            buckets=(0, 1, 2, 3, 5),
            registry=self.registry,
        )
        self.documents_ingested_total = Counter(
            "rag_documents_ingested_total",
            "Ingested documents by outcome",
            ["status"],
            registry=self.registry,
        )
        self.guardrail_blocks_total = Counter(
            "rag_guardrail_blocks_total",
            "Requests blocked by guardrails",
            ["guardrail"],
            registry=self.registry,
        )
        self.llm_calls_total = Counter(
            "rag_llm_calls_total",
            "LLM calls by outcome",
            ["outcome"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
