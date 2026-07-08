# enterprise-rag-platform

Multi-agent RAG platform for enterprise document search: hybrid retrieval (dense + BM25 + cross-encoder reranking), a LangGraph agent graph with self-correction, streaming SSE API, guardrails and full observability.

![CI](https://github.com/dataeclipse/enterprise-rag-platform/actions/workflows/ci.yml/badge.svg)

> Demo GIF placeholder

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        U[Upload API] --> MQ[RabbitMQ]
        MQ --> W[Ingestion worker]
        W --> CH[Chunkers]
        CH --> EMB[Embedder bge-m3]
        EMB --> Q[(Qdrant)]
        W --> PG[(PostgreSQL)]
    end
    subgraph Query
        C[Client] -->|SSE| API[FastAPI]
        API --> G{LangGraph}
        G --> R1[Router]
        R1 --> R2[Retriever]
        R2 --> HS[Hybrid search + rerank]
        HS --> Q
        R2 --> R3[Reasoner]
        R3 --> R4[Critic]
        R4 -->|revise| R3
        R4 --> R5[Citations]
        R5 --> API
    end
    API --> GR[Guardrails]
    API --> OBS[Prometheus / OTel / LangSmith]
```

## Quickstart

```bash
git clone https://github.com/dataeclipse/enterprise-rag-platform
cd enterprise-rag-platform
cp .env.example .env
make up        # qdrant, postgres, redis, rabbitmq, prometheus, grafana
make dev       # uv sync with ml extras
uv run uvicorn rag.api.main:app --reload
```

## Status

Work in progress. Module-by-module build; see commit history.

## Design Decisions

See [docs/adr](docs/adr) for full records.

## License

MIT
