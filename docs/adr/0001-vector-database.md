# ADR 0001: Qdrant as Primary Vector Database

## Status

Accepted

## Context

The platform needs a vector store for dense retrieval over document chunks.
Candidates: Qdrant, pgvector, Chroma, Pinecone. Requirements: self-hosted
deployment, payload filtering for document-level deletes, cosine similarity,
async client, and a migration path if the team later standardizes on a
managed service.

## Decision

Qdrant is the primary backend. All access goes through the `VectorStore`
abstract base class with a registry-based `VectorStoreFactory`, so adding
pgvector, Chroma or Pinecone is a single adapter class. An `InMemoryVectorStore`
implements the same contract for tests and local development.

## Consequences

- Document deletion and versioning use payload filters (`document_id`),
  which Qdrant supports natively without a secondary index.
- Point IDs are UUIDv5 hashes of chunk IDs, so re-ingesting the same chunk
  is an idempotent upsert.
- pgvector remains available later without touching retrieval code; the cost
  today is one extra service in docker-compose.
- Unit tests never require a running Qdrant: the in-memory adapter plus a
  mocked async client cover the contract.
