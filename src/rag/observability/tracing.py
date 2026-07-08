from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter


def configure_tracing(
    service_name: str = "enterprise-rag-platform",
    exporter: SpanExporter | None = None,
) -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
