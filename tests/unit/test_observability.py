from typing import cast

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag.observability.metrics import Metrics
from rag.observability.tracing import configure_tracing, get_tracer


def test_metrics_isolated_registry() -> None:
    first = Metrics()
    second = Metrics()
    first.requests_total.labels(endpoint="/query", status="200").inc()
    rendered_first = first.render().decode()
    rendered_second = second.render().decode()
    assert 'rag_requests_total{endpoint="/query",status="200"} 1.0' in rendered_first
    assert 'endpoint="/query"' not in rendered_second


def test_metrics_histograms_observe() -> None:
    metrics = Metrics()
    metrics.retrieval_chunks.observe(5)
    metrics.correction_rounds.observe(1)
    metrics.request_duration_seconds.labels(endpoint="/query").observe(0.25)
    rendered = metrics.render().decode()
    assert "rag_retrieval_chunks_count 1.0" in rendered
    assert "rag_correction_rounds_count 1.0" in rendered


def test_metrics_counters_by_label() -> None:
    metrics = Metrics()
    metrics.guardrail_blocks_total.labels(guardrail="injection").inc()
    metrics.documents_ingested_total.labels(status="indexed").inc(3)
    rendered = metrics.render().decode()
    assert 'rag_guardrail_blocks_total{guardrail="injection"} 1.0' in rendered
    assert 'rag_documents_ingested_total{status="indexed"} 3.0' in rendered


def test_tracing_records_spans() -> None:
    exporter = InMemorySpanExporter()
    tracer = configure_tracing(service_name="test-service", exporter=exporter)
    with tracer.start_as_current_span("test-span"):
        pass
    provider = cast(TracerProvider, trace.get_tracer_provider())
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test-span"


def test_get_tracer_returns_named_tracer() -> None:
    tracer = get_tracer("component")
    assert tracer is not None
